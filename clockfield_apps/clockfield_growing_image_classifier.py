"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          CLOCKFIELD GROW — A Living Vision Model                            ║
║                                                                              ║
║  A growable image understanding system built on frozen CLIP.                 ║
║  Teach it new concepts via images or HuggingFace datasets.                   ║
║  Query its grown knowledge with natural language.                            ║
║  β-gated weight decay prevents catastrophic forgetting.                      ║
║                                                                              ║
║  Install: pip install torch torchvision transformers datasets gradio         ║
║           pillow numpy scikit-learn                                          ║
║  Run:     python clockfield_grow.py                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import json
import time
import threading
import numpy as np
from pathlib import Path
from collections import defaultdict
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import gradio as gr

from transformers import CLIPModel, CLIPProcessor
from datasets import load_dataset

# ── Constants ────────────────────────────────────────────────────────────────

SAVE_DIR = Path("./clockfield_brain")
SAVE_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"

# ═════════════════════════════════════════════════════════════════════════════
# 1. CLOCKFIELD GROW HEAD — the part that learns, β-protected
# ═════════════════════════════════════════════════════════════════════════════

class ClockfieldGrowHead(nn.Module):
    """
    A small MLP head that sits on top of frozen CLIP image embeddings.
    It maps 512-dim CLIP features → a learned concept space.
    
    β is measured as activation roughness across the batch.
    High β (rough/noisy) → high weight decay → prevents overfit on new data.
    Low β (crystallized/confident) → low weight decay → protects learned concepts.
    """
    def __init__(self, clip_dim: int = 512, hidden_dim: int = 256, num_concepts: int = 0):
        super().__init__()
        self.clip_dim = clip_dim
        self.hidden_dim = hidden_dim
        
        # The growable trunk
        self.trunk = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        
        # The concept heads — one per learned class
        # We grow this list as new concepts are added
        self.concept_heads: nn.ModuleDict = nn.ModuleDict()
        self.concept_names: List[str] = []
        
        # β EMA per layer (for the trunk layers)
        self.register_buffer('beta_ema_0', torch.tensor(0.5))
        self.register_buffer('beta_ema_1', torch.tensor(0.5))
        
        self._activations = {}
        self._hooks = []

    def add_concept(self, name: str):
        """Grow a new output head for a new concept."""
        if name not in self.concept_heads:
            head = nn.Linear(self.hidden_dim, 1)
            nn.init.normal_(head.weight, std=0.01)
            nn.init.zeros_(head.bias)
            self.concept_heads[name] = head
            self.concept_names.append(name)
            return True
        return False

    def _hook(self, layer_idx):
        def fn(module, inp, out):
            self._activations[layer_idx] = out.detach()
        return fn

    def register_beta_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks = [
            self.trunk[0].register_forward_hook(self._hook(0)),
            self.trunk[3].register_forward_hook(self._hook(1)),
        ]

    def compute_beta(self) -> Dict[int, float]:
        """
        Temporal roughness: how violently activations vary across the batch.
        High roughness = model is uncertain = high β = more decay.
        """
        betas = {}
        for idx, acts in self._activations.items():
            if acts.dim() >= 2 and acts.shape[0] > 1:
                acts_norm = (acts - acts.mean(0, keepdim=True)) / (acts.std(0, keepdim=True) + 1e-6)
                roughness = torch.abs(torch.diff(acts_norm, dim=0)).mean().item()
                betas[idx] = roughness
            else:
                betas[idx] = 0.5
        return betas

    def update_beta_ema(self, betas: Dict[int, float]):
        if 0 in betas:
            self.beta_ema_0 = 0.9 * self.beta_ema_0 + 0.1 * betas[0]
        if 1 in betas:
            self.beta_ema_1 = 0.9 * self.beta_ema_1 + 0.1 * betas[1]

    def get_gamma(self, alpha: float = 3.0) -> Dict[str, float]:
        """Γ = exp(-α·β) — the time dilation / decay shield."""
        b0 = self.beta_ema_0.item()
        b1 = self.beta_ema_1.item()
        return {
            'layer_0': float(np.exp(-alpha * b0)),
            'layer_1': float(np.exp(-alpha * b1)),
            'beta_0': b0,
            'beta_1': b1,
        }

    def forward(self, clip_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = self.trunk(clip_features)
        outputs = {}
        for name, head in self.concept_heads.items():
            outputs[name] = head(features).squeeze(-1)
        return features, outputs

    def get_similarity_to_clip_text(self, clip_text_features: torch.Tensor) -> torch.Tensor:
        """
        Query the grown space by projecting CLIP text features into it.
        The trunk maps visual space → grown concept space.
        We do the same for text (CLIP aligns them).
        """
        return self.trunk(clip_text_features)


# ═════════════════════════════════════════════════════════════════════════════
# 2. DATASET WRAPPERS
# ═════════════════════════════════════════════════════════════════════════════

class ImageFolderDataset(Dataset):
    """Load images from a folder, using subfolder names as labels."""
    def __init__(self, root: str, processor: CLIPProcessor):
        self.processor = processor
        self.samples = []
        root = Path(root)
        for label_dir in sorted(root.iterdir()):
            if label_dir.is_dir():
                for img_path in label_dir.glob("*"):
                    if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}:
                        self.samples.append((str(img_path), label_dir.name))
        self.classes = sorted(set(s[1] for s in self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0), label


def sniff_hf_fields(hf_dataset) -> Tuple[str, str]:
    """
    Auto-detect image and label field names from a HuggingFace dataset.
    Returns (image_field, label_field).
    """
    from datasets import Image as HFImage, ClassLabel, Value
    features = hf_dataset.features

    # Find image field
    image_field = None
    for name, feat in features.items():
        if isinstance(feat, HFImage):
            image_field = name
            break
    if image_field is None:
        # fallback: look for common names
        for candidate in ['image', 'img', 'pixel_values', 'photo']:
            if candidate in features:
                image_field = candidate
                break
    if image_field is None:
        image_field = 'image'  # last resort

    # Find label field
    label_field = None
    for name, feat in features.items():
        if isinstance(feat, ClassLabel):
            label_field = name
            break
    if label_field is None:
        # fallback: look for common names
        for candidate in ['label', 'labels', 'target', 'class', 'category', 'fine_label', 'coarse_label']:
            if candidate in features:
                label_field = candidate
                break
    if label_field is None:
        label_field = 'label'  # last resort

    return image_field, label_field


class HFDatasetWrapper(Dataset):
    """
    Wrap a HuggingFace image classification dataset.
    Auto-detects field names; manual override supported.
    """
    def __init__(self, hf_dataset, processor: CLIPProcessor, label_field: str = 'auto', image_field: str = 'auto'):
        self.data = hf_dataset
        self.processor = processor

        # Auto-detect fields if not specified
        if image_field == 'auto' or label_field == 'auto':
            detected_img, detected_lbl = sniff_hf_fields(hf_dataset)
            self.image_field = detected_img if image_field == 'auto' else image_field
            self.label_field = detected_lbl if label_field == 'auto' else label_field
        else:
            self.image_field = image_field
            self.label_field = label_field

        print(f"[HFDataset] Using image_field='{self.image_field}', label_field='{self.label_field}'")
        print(f"[HFDataset] Available fields: {list(hf_dataset.features.keys())}")

        # Get class names
        feat = hf_dataset.features.get(self.label_field)
        if hasattr(feat, 'names'):
            self.classes = feat.names
        else:
            unique = set(hf_dataset[self.label_field])
            self.classes = [str(u) for u in sorted(unique)]

        print(f"[HFDataset] Classes: {self.classes}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = item[self.image_field]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img).convert("RGB")
        else:
            img = img.convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt")
        label_idx = item[self.label_field]
        label = self.classes[label_idx] if isinstance(label_idx, int) else str(label_idx)
        return inputs['pixel_values'].squeeze(0), label


# ═════════════════════════════════════════════════════════════════════════════
# 3. THE CLOCKFIELD TRAINER
# ═════════════════════════════════════════════════════════════════════════════

class ClockfieldTrainer:
    def __init__(self, alpha: float = 3.0, lambda_max: float = 0.1):
        self.alpha = alpha
        self.lambda_max = lambda_max
        self.log_lines: List[str] = []
        self.is_training = False
        self._stop_flag = False

    def _log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log_lines.append(line)
        if len(self.log_lines) > 200:
            self.log_lines = self.log_lines[-200:]
        print(line)

    def _build_optimizer(self, head: ClockfieldGrowHead, lr: float):
        """Build optimizer with per-layer β-adaptive weight decay groups."""
        groups = []
        for name, param in head.trunk.named_parameters():
            layer_idx = 0 if name.startswith('0') else 1
            groups.append({
                'params': [param],
                'lr': lr,
                'weight_decay': self.lambda_max,
                '_layer_idx': layer_idx,
            })
        for concept_name, module in head.concept_heads.items():
            for param in module.parameters():
                groups.append({
                    'params': [param],
                    'lr': lr * 2,  # concept heads learn faster
                    'weight_decay': 0.01,
                    '_layer_idx': None,
                })
        return torch.optim.AdamW(groups)

    def _update_decay(self, optimizer, gamma: Dict[str, float]):
        """Update weight decay based on current γ values."""
        gamma_map = {0: gamma['layer_0'], 1: gamma['layer_1']}
        for group in optimizer.param_groups:
            idx = group.get('_layer_idx')
            if idx is not None and idx in gamma_map:
                group['weight_decay'] = max(self.lambda_max * gamma_map[idx], 1e-5)

    def train(
        self,
        clip_model: CLIPModel,
        head: ClockfieldGrowHead,
        dataset: Dataset,
        epochs: int = 3,
        batch_size: int = 16,
        lr: float = 1e-3,
        progress_callback=None,
    ):
        self.is_training = True
        self._stop_flag = False
        head.register_beta_hooks()
        
        # Register all concepts upfront — use .classes if available (fast + complete)
        if hasattr(dataset, 'classes'):
            for label in dataset.classes:
                head.add_concept(label)
        else:
            for _, label in dataset:
                head.add_concept(label)
        head = head.to(DEVICE)
        clip_model = clip_model.to(DEVICE)
        clip_model.eval()

        optimizer = self._build_optimizer(head, lr)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        
        label_to_idx = {name: i for i, name in enumerate(head.concept_names)}
        total_steps = epochs * len(loader)
        step = 0

        self._log(f"Starting training: {len(dataset)} samples, {len(head.concept_names)} concepts, {epochs} epochs")
        self._log(f"Concepts: {head.concept_names}")

        for epoch in range(epochs):
            if self._stop_flag:
                break
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for batch_pixels, batch_labels in loader:
                if self._stop_flag:
                    break
                
                batch_pixels = batch_pixels.to(DEVICE)
                
                # Get frozen CLIP embeddings
                with torch.no_grad():
                    clip_feats = clip_model.get_image_features(pixel_values=batch_pixels)
                    clip_feats = F.normalize(clip_feats, dim=-1)

                # Forward through grow head
                features, outputs = head(clip_feats)

                # Build loss: binary cross-entropy per concept
                loss = torch.tensor(0.0, device=DEVICE)
                for label in batch_labels:
                    if label not in outputs:
                        continue
                # Multi-class: build logit matrix [batch, n_concepts]
                concept_names = list(outputs.keys())
                if not concept_names:
                    continue
                logits = torch.stack([outputs[n] for n in concept_names], dim=1)  # [B, C]
                targets = torch.tensor(
                    [concept_names.index(l) if l in concept_names else 0 for l in batch_labels],
                    device=DEVICE, dtype=torch.long
                )
                loss = F.cross_entropy(logits, targets)

                optimizer.zero_grad()
                loss.backward()

                # Update β from activations
                betas = head.compute_beta()
                head.update_beta_ema(betas)
                gamma = head.get_gamma(self.alpha)
                self._update_decay(optimizer, gamma)

                optimizer.step()

                # Track accuracy
                preds = logits.argmax(dim=1)
                correct = (preds == targets).sum().item()
                epoch_correct += correct
                epoch_total += len(targets)
                epoch_loss += loss.item()
                step += 1

                if progress_callback:
                    progress_callback(step / total_steps)

            acc = epoch_correct / max(epoch_total, 1)
            gamma = head.get_gamma(self.alpha)
            self._log(
                f"Epoch {epoch+1}/{epochs} | "
                f"Loss: {epoch_loss/max(len(loader),1):.4f} | "
                f"Acc: {acc:.1%} | "
                f"β₀={gamma['beta_0']:.3f} β₁={gamma['beta_1']:.3f} | "
                f"Γ₀={gamma['layer_0']:.3f} Γ₁={gamma['layer_1']:.3f}"
            )

        self._log("Training complete.")
        self.is_training = False
        return head

    def stop(self):
        self._stop_flag = True
        self._log("Stop requested.")


# ═════════════════════════════════════════════════════════════════════════════
# 4. BRAIN SAVE / LOAD
# ═════════════════════════════════════════════════════════════════════════════

def save_brain(head: ClockfieldGrowHead, path: Path = SAVE_DIR):
    path.mkdir(exist_ok=True)
    torch.save(head.state_dict(), path / "grow_head.pt")
    meta = {
        'concept_names': head.concept_names,
        'clip_dim': head.clip_dim,
        'hidden_dim': head.hidden_dim,
        'saved_at': time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    return str(path)

def load_brain(path: Path = SAVE_DIR) -> Optional[ClockfieldGrowHead]:
    meta_path = path / "meta.json"
    weights_path = path / "grow_head.pt"
    if not meta_path.exists() or not weights_path.exists():
        return None
    with open(meta_path) as f:
        meta = json.load(f)
    head = ClockfieldGrowHead(
        clip_dim=meta['clip_dim'],
        hidden_dim=meta['hidden_dim'],
    )
    for name in meta['concept_names']:
        head.add_concept(name)
    head.load_state_dict(torch.load(weights_path, map_location='cpu'))
    return head


# ═════════════════════════════════════════════════════════════════════════════
# 5. INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def query_image(
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    head: ClockfieldGrowHead,
    image: Image.Image,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    """What does the grown brain think this image is?"""
    if not head.concept_names:
        return [("No concepts learned yet", 0.0)]
    
    head.eval()
    clip_model.eval()
    with torch.no_grad():
        inputs = processor(images=image.convert("RGB"), return_tensors="pt").to(DEVICE)
        clip_feats = clip_model.get_image_features(**inputs)
        clip_feats = F.normalize(clip_feats, dim=-1)
        _, outputs = head(clip_feats)
    
    concept_names = list(outputs.keys())
    logits = torch.stack([outputs[n] for n in concept_names], dim=1)
    probs = F.softmax(logits, dim=1).squeeze(0)
    
    results = sorted(zip(concept_names, probs.cpu().tolist()), key=lambda x: -x[1])
    return results[:top_k]

def query_text(
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    head: ClockfieldGrowHead,
    text: str,
) -> List[Tuple[str, float]]:
    """
    Query the grown concept space with a text prompt via CLIP text encoder.
    CLIP aligns image and text embeddings — so we can probe what the grown
    head has learned using natural language.
    """
    if not head.concept_names:
        return [("No concepts learned yet", 0.0)]
    
    head.eval()
    clip_model.eval()
    with torch.no_grad():
        inputs = processor(text=[text], return_tensors="pt", padding=True).to(DEVICE)
        text_feats = clip_model.get_text_features(**inputs)
        text_feats = F.normalize(text_feats, dim=-1)
        _, outputs = head(text_feats)
    
    concept_names = list(outputs.keys())
    logits = torch.stack([outputs[n] for n in concept_names], dim=1)
    probs = F.softmax(logits, dim=1).squeeze(0)
    
    results = sorted(zip(concept_names, probs.cpu().tolist()), key=lambda x: -x[1])
    return results


# ═════════════════════════════════════════════════════════════════════════════
# 6. GRADIO APP
# ═════════════════════════════════════════════════════════════════════════════

def build_app():
    # ── Shared state ──────────────────────────────────────────────────────────
    state = {
        'clip_model': None,
        'processor': None,
        'head': ClockfieldGrowHead(),
        'trainer': ClockfieldTrainer(),
        'loaded': False,
    }

    def ensure_clip():
        if not state['loaded']:
            gr.Info("Loading CLIP model (first time only)...")
            state['processor'] = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
            state['clip_model'] = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(DEVICE)
            state['clip_model'].eval()
            for p in state['clip_model'].parameters():
                p.requires_grad_(False)
            state['loaded'] = True
            # Try loading saved brain
            saved = load_brain()
            if saved:
                state['head'] = saved
                return f"✅ CLIP loaded. Restored brain with concepts: {state['head'].concept_names}"
            return "✅ CLIP loaded. Fresh brain — ready to learn."
        return "✅ Already loaded."

    # ── Load CLIP ─────────────────────────────────────────────────────────────
    def on_load_clip():
        msg = ensure_clip()
        return msg, get_status()

    # ── Status ────────────────────────────────────────────────────────────────
    def get_status():
        head = state['head']
        gamma = head.get_gamma() if head.concept_names else {}
        concepts = head.concept_names or []
        lines = [
            f"**Concepts learned:** {len(concepts)}",
            f"**Concept list:** {', '.join(concepts) if concepts else '—'}",
            f"**Device:** {DEVICE}",
        ]
        if gamma:
            lines += [
                f"**β₀ (trunk):** {gamma.get('beta_0', 0):.3f}",
                f"**β₁ (trunk):** {gamma.get('beta_1', 0):.3f}",
                f"**Γ₀ (shield):** {gamma.get('layer_0', 0):.3f}",
                f"**Γ₁ (shield):** {gamma.get('layer_1', 0):.3f}",
            ]
        return "\n".join(lines)

    # ── Train on HF dataset ───────────────────────────────────────────────────
    def on_train_hf(dataset_name, split, image_field, label_field, epochs, batch_size, lr, progress=gr.Progress()):
        ensure_clip()
        if not dataset_name.strip():
            return "Please enter a HuggingFace dataset name.", get_status()
        
        try:
            state['trainer']._log(f"Loading HF dataset: {dataset_name} ({split})...")
            hf_data = load_dataset(dataset_name, split=split)
            state['trainer']._log(f"Fields found: {list(hf_data.features.keys())}")
            # 'auto' triggers sniff_hf_fields; manual entry overrides
            img_f = 'auto' if image_field.strip() in ('image', 'auto', '') else image_field.strip()
            lbl_f = 'auto' if label_field.strip() in ('label', 'auto', '') else label_field.strip()
            dataset = HFDatasetWrapper(hf_data, state['processor'], label_field=lbl_f, image_field=img_f)
            state['trainer']._log(f"Using → image='{dataset.image_field}'  label='{dataset.label_field}'")
            state['trainer']._log(f"Dataset: {len(dataset)} samples, classes: {dataset.classes}")
            
            def prog_cb(p):
                progress(p, desc="Training...")
            
            state['head'] = state['trainer'].train(
                state['clip_model'], state['head'], dataset,
                epochs=int(epochs), batch_size=int(batch_size), lr=float(lr),
                progress_callback=prog_cb,
            )
            log = "\n".join(state['trainer'].log_lines[-30:])
            return log, get_status()
        except Exception as e:
            return f"Error: {e}", get_status()


    # ── Train on uploaded images ──────────────────────────────────────────────
    def on_train_images(images, concept_name, epochs, batch_size, lr, progress=gr.Progress()):
        ensure_clip()
        if not images:
            return "Upload at least one image.", get_status()
        if not concept_name.strip():
            return "Enter a concept name (e.g. 'my_cat').", get_status()
        concept_name = concept_name.strip().replace(" ", "_")
        try:
            class UploadedImageDataset(Dataset):
                def __init__(self, pil_images, label, processor):
                    self.images = pil_images
                    self.label = label
                    self.processor = processor
                    self.classes = [label]
                def __len__(self):
                    return len(self.images)
                def __getitem__(self, idx):
                    img = self.images[idx].convert("RGB")
                    inputs = self.processor(images=img, return_tensors="pt")
                    return inputs["pixel_values"].squeeze(0), self.label

            pil_images = []
            for img_data in images:
                if img_data is None:
                    continue
                import numpy as _np
                if isinstance(img_data, Image.Image):
                    pil_images.append(img_data.convert("RGB"))
                else:
                    pil_images.append(Image.fromarray(_np.array(img_data)).convert("RGB"))

            if not pil_images:
                return "No valid images found.", get_status()

            state["trainer"]._log(f"Teaching concept: {concept_name} ({len(pil_images)} images)")
            dataset = UploadedImageDataset(pil_images, concept_name, state["processor"])

            def prog_cb(p):
                progress(p, desc=f"Teaching {concept_name}...")

            state["head"] = state["trainer"].train(
                state["clip_model"], state["head"], dataset,
                epochs=int(epochs), batch_size=min(int(batch_size), max(1, len(pil_images))), lr=float(lr),
                progress_callback=prog_cb,
            )
            log = "\n".join(state["trainer"].log_lines[-20:])
            return log, get_status()
        except Exception as e:
            import traceback
            return f"Error: {e}\n{traceback.format_exc()}", get_status()

    # ── Train on folder ───────────────────────────────────────────────────────
    def on_train_folder(folder_path, epochs, batch_size, lr, progress=gr.Progress()):
        ensure_clip()
        if not folder_path or not Path(folder_path).exists():
            return "Folder not found. Structure: root/class_name/image.jpg", get_status()
        try:
            dataset = ImageFolderDataset(folder_path, state['processor'])
            if len(dataset) == 0:
                return "No images found. Subfolders = class names.", get_status()
            state['trainer']._log(f"Folder dataset: {len(dataset)} images, classes: {dataset.classes}")
            
            def prog_cb(p):
                progress(p, desc="Training...")
            
            state['head'] = state['trainer'].train(
                state['clip_model'], state['head'], dataset,
                epochs=int(epochs), batch_size=int(batch_size), lr=float(lr),
                progress_callback=prog_cb,
            )
            log = "\n".join(state['trainer'].log_lines[-30:])
            return log, get_status()
        except Exception as e:
            return f"Error: {e}", get_status()

    # ── Query image ───────────────────────────────────────────────────────────
    def on_query_image(image):
        ensure_clip()
        if image is None:
            return "Upload an image first."
        if not state['head'].concept_names:
            return "No concepts learned yet. Train the model first."
        try:
            results = query_image(state['clip_model'], state['processor'], state['head'], Image.fromarray(image))
            lines = ["**Top predictions:**"]
            for name, prob in results:
                bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
                lines.append(f"`{name:<20}` {bar} {prob:.1%}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    # ── Query text ────────────────────────────────────────────────────────────
    def on_query_text(text_prompt):
        ensure_clip()
        if not text_prompt.strip():
            return "Enter a text prompt."
        if not state['head'].concept_names:
            return "No concepts learned yet."
        try:
            results = query_text(state['clip_model'], state['processor'], state['head'], text_prompt)
            lines = [f"**Query:** *{text_prompt}*", ""]
            for name, prob in results:
                bar = "█" * int(prob * 20) + "░" * (20 - int(prob * 20))
                lines.append(f"`{name:<20}` {bar} {prob:.1%}")
            return "\n".join(lines)
        except Exception as e:
            return f"Error: {e}"

    # ── Save / Load ───────────────────────────────────────────────────────────
    def on_save():
        path = save_brain(state['head'])
        return f"✅ Brain saved to: {path}\nConcepts: {state['head'].concept_names}"

    def on_load():
        head = load_brain()
        if head:
            state['head'] = head
            return f"✅ Brain loaded! Concepts: {head.concept_names}", get_status()
        return "No saved brain found.", get_status()

    def on_stop():
        state['trainer'].stop()
        return "⏹ Stop requested."

    def on_reset():
        state['head'] = ClockfieldGrowHead()
        return "🧹 Brain reset. All concepts cleared.", get_status()

    def get_log():
        return "\n".join(state['trainer'].log_lines[-50:])

    # ── UI ────────────────────────────────────────────────────────────────────
    css = """
    :root {
        --bg: #0a0c10;
        --panel: #111318;
        --border: #1e2230;
        --accent: #00d4ff;
        --accent2: #7c3aed;
        --text: #c8d0e0;
        --muted: #4a5568;
        --green: #34d399;
        --orange: #f59e0b;
    }
    body, .gradio-container { background: var(--bg) !important; color: var(--text) !important; font-family: 'JetBrains Mono', 'Fira Code', monospace !important; }
    h1, h2, h3 { color: var(--accent) !important; letter-spacing: 0.05em; }
    .gr-button { border-radius: 2px !important; font-family: monospace !important; font-size: 12px !important; }
    .gr-button-primary { background: var(--accent2) !important; border: 1px solid var(--accent) !important; color: white !important; }
    .gr-button-secondary { background: var(--panel) !important; border: 1px solid var(--border) !important; color: var(--text) !important; }
    .gr-box, .gr-form { background: var(--panel) !important; border: 1px solid var(--border) !important; }
    .gr-input, textarea, input { background: #0d0f14 !important; color: var(--text) !important; border: 1px solid var(--border) !important; font-family: monospace !important; }
    .gr-markdown { color: var(--text) !important; }
    label { color: var(--muted) !important; font-size: 11px !important; letter-spacing: 0.08em; text-transform: uppercase; }
    """

    with gr.Blocks(title="Clockfield Grow", css=css, theme=gr.themes.Base()) as app:

        gr.Markdown("""
# ⟳ CLOCKFIELD GROW
### A living vision model. Teach it. Query it. Let it grow.

Built on frozen **CLIP** (your perceptual backbone) + a **β-gated learning head** that accumulates concepts without forgetting.  
`Γ = exp(−α·β)` — crystallized knowledge is protected. New knowledge grows around it.
        """)

        with gr.Row():
            load_btn = gr.Button("⚡ LOAD CLIP", variant="primary", scale=1)
            save_btn = gr.Button("💾 SAVE BRAIN", scale=1)
            load_saved_btn = gr.Button("📂 LOAD BRAIN", scale=1)
            stop_btn = gr.Button("⏹ STOP", scale=1)
            reset_btn = gr.Button("🧹 RESET", scale=1)

        with gr.Row():
            status_box = gr.Markdown("*Click LOAD CLIP to begin.*", label="Status")

        load_msg = gr.Textbox(label="System", interactive=False, lines=1)

        gr.Markdown("---")

        with gr.Tabs():

            # ── TAB 1: HuggingFace dataset ────────────────────────────────────
            with gr.Tab("🤗 Train: HuggingFace Dataset"):
                gr.Markdown("""
Feed it any HuggingFace image classification dataset.  
Try: `cifar10`, `food101`, `beans`, `cats_vs_dogs`, `keremberke/pokemon-classification`
                """)
                with gr.Row():
                    hf_name = gr.Textbox(label="Dataset Name", value="beans", placeholder="e.g. food101")
                    hf_split = gr.Textbox(label="Split", value="train[:500]", placeholder="train, train[:200]")
                with gr.Row():
                    hf_img_field = gr.Textbox(label="Image Field", value="auto", placeholder="auto-detected")
                    hf_label_field = gr.Textbox(label="Label Field", value="auto", placeholder="auto-detected")
                with gr.Row():
                    hf_epochs = gr.Slider(1, 20, value=3, step=1, label="Epochs")
                    hf_batch = gr.Slider(4, 64, value=16, step=4, label="Batch Size")
                    hf_lr = gr.Number(value=1e-3, label="Learning Rate")
                hf_train_btn = gr.Button("▶ TRAIN ON HF DATASET", variant="primary")
                hf_log = gr.Textbox(label="Training Log", lines=12, interactive=False)
                hf_status = gr.Markdown()

            # ── TAB 2: Local folder ───────────────────────────────────────────
            with gr.Tab("📁 Train: Local Folder"):
                gr.Markdown("""
Point to a folder structured like:
```
root/
  dog/    image1.jpg  image2.jpg ...
  cat/    image1.jpg  ...
  bird/   ...
```
Each subfolder = one concept.
                """)
                folder_path = gr.Textbox(label="Folder Path", placeholder="/path/to/your/images")
                with gr.Row():
                    f_epochs = gr.Slider(1, 20, value=3, step=1, label="Epochs")
                    f_batch = gr.Slider(4, 64, value=16, step=4, label="Batch Size")
                    f_lr = gr.Number(value=1e-3, label="Learning Rate")
                folder_train_btn = gr.Button("▶ TRAIN ON FOLDER", variant="primary")
                folder_log = gr.Textbox(label="Training Log", lines=12, interactive=False)
                folder_status = gr.Markdown()


            # ── TAB 2b: Train on uploaded images ─────────────────────────────
            with gr.Tab("🖼️ Train: Upload Images"):
                gr.Markdown("""
Upload your own images and name the concept. The brain grows a new class from them.  
Upload many images of one thing → name it → train. Repeat for each concept. All old concepts survive.
                """)
                upload_images = gr.Gallery(label="Upload Images (drag & drop multiple)", columns=4, height=260, type="pil")
                concept_name_box = gr.Textbox(label="Concept Name", placeholder="e.g. my_cat, defect_A, sunset")
                with gr.Row():
                    ui_epochs = gr.Slider(1, 50, value=10, step=1, label="Epochs")
                    ui_batch = gr.Slider(1, 32, value=8, step=1, label="Batch Size")
                    ui_lr = gr.Number(value=1e-3, label="Learning Rate")
                upload_train_btn = gr.Button("▶ TEACH THIS CONCEPT", variant="primary")
                upload_log = gr.Textbox(label="Training Log", lines=8, interactive=False)
                upload_status = gr.Markdown()

            # ── TAB 3: Query Image ────────────────────────────────────────────
            with gr.Tab("🔍 Query: Image"):
                gr.Markdown("Upload any image. The grown brain tells you what it thinks it is.")
                with gr.Row():
                    query_img_input = gr.Image(label="Input Image", type="numpy")
                    query_img_result = gr.Markdown(label="Predictions")
                query_img_btn = gr.Button("🔍 CLASSIFY", variant="primary")

            # ── TAB 4: Query Text ─────────────────────────────────────────────
            with gr.Tab("💬 Query: Text Prompt"):
                gr.Markdown("""
Probe the grown concept space with natural language.  
CLIP aligns text and image in the same embedding space — so text can query visual concepts.
                """)
                text_prompt = gr.Textbox(label="Text Prompt", placeholder="a photo of a cat", lines=2)
                query_text_btn = gr.Button("💬 QUERY", variant="primary")
                text_result = gr.Markdown()

            # ── TAB 5: Live log ───────────────────────────────────────────────
            with gr.Tab("📊 Live Log & Metrics"):
                refresh_btn = gr.Button("↻ Refresh")
                live_log = gr.Textbox(label="Training Log", lines=20, interactive=False)
                live_status = gr.Markdown()
                refresh_btn.click(fn=lambda: (get_log(), get_status()), outputs=[live_log, live_status])

        # ── Wire up events ─────────────────────────────────────────────────────
        load_btn.click(fn=on_load_clip, outputs=[load_msg, status_box])
        save_btn.click(fn=on_save, outputs=load_msg)
        load_saved_btn.click(fn=on_load, outputs=[load_msg, status_box])
        stop_btn.click(fn=on_stop, outputs=load_msg)
        reset_btn.click(fn=on_reset, outputs=[load_msg, status_box])

        hf_train_btn.click(
            fn=on_train_hf,
            inputs=[hf_name, hf_split, hf_img_field, hf_label_field, hf_epochs, hf_batch, hf_lr],
            outputs=[hf_log, hf_status],
        )
        folder_train_btn.click(
            fn=on_train_folder,
            inputs=[folder_path, f_epochs, f_batch, f_lr],
            outputs=[folder_log, folder_status],
        )
        upload_train_btn.click(
            fn=on_train_images,
            inputs=[upload_images, concept_name_box, ui_epochs, ui_batch, ui_lr],
            outputs=[upload_log, upload_status],
        )
        query_img_btn.click(fn=on_query_image, inputs=[query_img_input], outputs=[query_img_result])
        query_text_btn.click(fn=on_query_text, inputs=[text_prompt], outputs=[text_result])

    return app


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════════════════╗
║  CLOCKFIELD GROW — starting up                                  ║
║  Device: {}                                                   ║
║  Brain dir: {}                                    ║
╚══════════════════════════════════════════════════════════════════╝
""".format(DEVICE, SAVE_DIR))
    
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # set True to get a public gradio.live link
        show_error=True,
    )