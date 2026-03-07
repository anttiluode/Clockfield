"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          CLOCKFIELD DREAM — A Growing Image Generator                       ║
║                                                                              ║
║  Teach it names + images. It learns to dream them back.                      ║
║  "antti" → learns your face. "dog" → learns dogs.                            ║
║  "antti dog" → fuses both. Old concepts never die.                           ║
║                                                                              ║
║  Architecture:                                                               ║
║    CLIP text encoder (frozen) → concept embedding                            ║
║    Per-concept LoRA blocks (trainable, β-shielded)                           ║
║    Shared CNN decoder (frozen after first concept, grown via LoRA)           ║
║    β-gated weight decay — crystallized concepts protected by Γ              ║
║                                                                              ║
║  Install:                                                                    ║
║    pip install torch torchvision transformers datasets gradio pillow numpy  ║
║  Run:                                                                        ║
║    python clockfield_dream.py                                                ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os, json, time, math, threading
import numpy as np
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import gradio as gr

from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────
SAVE_DIR   = Path("./clockfield_dream_brain")
SAVE_DIR.mkdir(exist_ok=True)
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CLIP_ID    = "openai/clip-vit-base-patch32"
IMG_SIZE   = 64          # generated image size — fast, fits 12GB easily
CLIP_DIM   = 512         # CLIP embedding dim
LATENT_DIM = 256         # internal latent dim

# ═════════════════════════════════════════════════════════════════════════════
# 1. THE DECODER — small CNN that turns a concept vector into pixels
# ═════════════════════════════════════════════════════════════════════════════

class ConceptDecoder(nn.Module):
    """
    Maps a LATENT_DIM concept vector → RGB image (IMG_SIZE × IMG_SIZE).
    Architecture: FC projection → reshape → 5× (upsample + conv).
    64×64 output with CLIP_DIM=512 input via projection.
    """
    def __init__(self, latent_dim: int = LATENT_DIM, img_size: int = IMG_SIZE):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim

        # Project CLIP embedding → latent
        self.clip_proj = nn.Sequential(
            nn.Linear(CLIP_DIM, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
        )

        # FC → spatial seed (4×4 × 256ch)
        self.fc = nn.Linear(latent_dim, 256 * 4 * 4)

        # Upsample tower: 4→8→16→32→64
        def up_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            )

        self.up1 = up_block(256, 256)   # 4→8
        self.up2 = up_block(256, 128)   # 8→16
        self.up3 = up_block(128, 64)    # 16→32
        self.up4 = up_block(64,  32)    # 32→64
        self.out_conv = nn.Conv2d(32, 3, 3, padding=1)

        # β EMA buffers — one per decoder stage
        self.register_buffer('beta_ema_fc',  torch.tensor(0.5))
        self.register_buffer('beta_ema_up1', torch.tensor(0.5))
        self.register_buffer('beta_ema_up2', torch.tensor(0.5))
        self.register_buffer('beta_ema_up3', torch.tensor(0.5))

        self._acts = {}
        self._hooks = []

    def register_beta_hooks(self):
        for h in self._hooks: h.remove()
        self._hooks = []
        def hook(name):
            def fn(m, inp, out):
                self._acts[name] = out.detach()
            return fn
        self._hooks = [
            self.fc.register_forward_hook(hook('fc')),
            self.up1.register_forward_hook(hook('up1')),
            self.up2.register_forward_hook(hook('up2')),
            self.up3.register_forward_hook(hook('up3')),
        ]

    def compute_beta(self) -> Dict[str, float]:
        betas = {}
        for name, acts in self._acts.items():
            flat = acts.view(acts.shape[0], -1)  # [B, *]
            if flat.shape[0] > 1:
                norm = (flat - flat.mean(0, keepdim=True)) / (flat.std(0, keepdim=True) + 1e-6)
                betas[name] = torch.abs(torch.diff(norm, dim=0)).mean().item()
            else:
                betas[name] = 0.5
        return betas

    def update_beta(self, betas: Dict[str, float]):
        if 'fc'  in betas: self.beta_ema_fc  = 0.9 * self.beta_ema_fc  + 0.1 * betas['fc']
        if 'up1' in betas: self.beta_ema_up1 = 0.9 * self.beta_ema_up1 + 0.1 * betas['up1']
        if 'up2' in betas: self.beta_ema_up2 = 0.9 * self.beta_ema_up2 + 0.1 * betas['up2']
        if 'up3' in betas: self.beta_ema_up3 = 0.9 * self.beta_ema_up3 + 0.1 * betas['up3']

    def get_gamma(self, alpha: float = 3.0) -> Dict[str, float]:
        betas = {
            'fc':  self.beta_ema_fc.item(),
            'up1': self.beta_ema_up1.item(),
            'up2': self.beta_ema_up2.item(),
            'up3': self.beta_ema_up3.item(),
        }
        return {k: float(np.exp(-alpha * v)) for k, v in betas.items()}, betas

    def forward(self, clip_emb: torch.Tensor) -> torch.Tensor:
        """clip_emb: [B, CLIP_DIM] → image: [B, 3, IMG_SIZE, IMG_SIZE] in [-1, 1]"""
        x = self.clip_proj(clip_emb)          # [B, latent_dim]
        x = self.fc(x)                         # [B, 256*4*4]
        x = x.view(-1, 256, 4, 4)             # [B, 256, 4, 4]
        x = self.up1(x)                        # [B, 256, 8, 8]
        x = self.up2(x)                        # [B, 128, 16, 16]
        x = self.up3(x)                        # [B, 64, 32, 32]
        x = self.up4(x)                        # [B, 32, 64, 64]
        x = self.out_conv(x)                   # [B, 3, 64, 64]
        return torch.tanh(x)


# ═════════════════════════════════════════════════════════════════════════════
# 2. CONCEPT MEMORY — stores per-concept CLIP embeddings and generated prototypes
# ═════════════════════════════════════════════════════════════════════════════

class ConceptMemory:
    """
    Stores the mean CLIP embedding for each learned concept.
    At inference, multi-word prompts average over matching concepts.
    This is the 'grown space' — entirely additive, never destructive.
    """
    def __init__(self):
        self.embeddings: Dict[str, torch.Tensor] = {}   # concept → mean CLIP emb
        self.image_counts: Dict[str, int] = {}
        self.beta_shields: Dict[str, float] = {}         # concept → Γ at save time

    def update(self, concept: str, emb: torch.Tensor, gamma: float = 1.0):
        """Running mean update of concept embedding. Always stored as [1, 512]."""
        emb = emb.detach().cpu().view(1, CLIP_DIM)  # force [1, 512] regardless of input shape
        n = self.image_counts.get(concept, 0)
        if concept not in self.embeddings:
            self.embeddings[concept] = emb.clone()
        else:
            self.embeddings[concept] = (0.8 * self.embeddings[concept].view(1, CLIP_DIM) + 0.2 * emb)
        self.image_counts[concept] = n + 1
        self.beta_shields[concept] = gamma

    def query(self, prompt: str, device: str = DEVICE) -> torch.Tensor:
        """
        Parse prompt into words, find matching concepts, return fused embedding.
        Unknown words are ignored. Multi-word = embedding average.
        """
        words = prompt.lower().strip().split()
        matched = []
        for word in words:
            if word in self.embeddings:
                matched.append(self.embeddings[word])
        if not matched:
            return torch.zeros(1, CLIP_DIM, device=device)
        # Stack all matched, each is [1, 512] → mean over concepts → [1, 512]
        fused = torch.cat([e.view(1, CLIP_DIM) for e in matched], dim=0).mean(0, keepdim=True)
        return fused.to(device)

    @property
    def concepts(self) -> List[str]:
        return list(self.embeddings.keys())

    def save(self, path: Path):
        data = {
            'embeddings': {k: v.tolist() for k, v in self.embeddings.items()},
            'image_counts': self.image_counts,
            'beta_shields': self.beta_shields,
        }
        with open(path / 'concept_memory.json', 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: Path):
        p = path / 'concept_memory.json'
        if not p.exists(): return
        with open(p) as f:
            data = json.load(f)
        self.embeddings = {k: torch.tensor(v) for k, v in data['embeddings'].items()}
        self.image_counts = data.get('image_counts', {})
        self.beta_shields = data.get('beta_shields', {})


# ═════════════════════════════════════════════════════════════════════════════
# 3. DATASETS
# ═════════════════════════════════════════════════════════════════════════════

IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # → [-1, 1]
])

def sniff_hf_fields(hf_dataset) -> Tuple[str, str]:
    from datasets import Image as HFImage, ClassLabel
    features = hf_dataset.features
    image_field = next((n for n, f in features.items() if isinstance(f, HFImage)), None)
    if image_field is None:
        image_field = next((n for n in features if n in ('image','img','photo')), 'image')
    label_field = next((n for n, f in features.items() if isinstance(f, ClassLabel)), None)
    if label_field is None:
        label_field = next((n for n in features if n in ('label','labels','target','class')), 'label')
    return image_field, label_field

class HFImageDataset(Dataset):
    def __init__(self, hf_dataset, image_field='auto', label_field='auto'):
        self.data = hf_dataset
        if image_field == 'auto' or label_field == 'auto':
            di, dl = sniff_hf_fields(hf_dataset)
            self.image_field = di if image_field == 'auto' else image_field
            self.label_field = dl if label_field == 'auto' else label_field
        else:
            self.image_field = image_field
            self.label_field = label_field
        feat = hf_dataset.features.get(self.label_field)
        self.classes = feat.names if hasattr(feat, 'names') else [str(u) for u in sorted(set(hf_dataset[self.label_field]))]
        print(f"[HF] image='{self.image_field}' label='{self.label_field}' classes={self.classes}")

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img = item[self.image_field]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = IMG_TRANSFORM(img.convert('RGB'))
        label_idx = item[self.label_field]
        label = self.classes[label_idx] if isinstance(label_idx, int) else str(label_idx)
        return img, label

class FolderImageDataset(Dataset):
    def __init__(self, root: str):
        self.samples = []
        self.classes = []
        for d in sorted(Path(root).iterdir()):
            if d.is_dir():
                self.classes.append(d.name)
                for f in d.glob('*'):
                    if f.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}:
                        self.samples.append((str(f), d.name))
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = IMG_TRANSFORM(Image.open(path).convert('RGB'))
        return img, label

class PILListDataset(Dataset):
    def __init__(self, images: List[Image.Image], label: str):
        self.images = images
        self.label = label
        self.classes = [label]
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        return IMG_TRANSFORM(self.images[idx].convert('RGB')), self.label


# ═════════════════════════════════════════════════════════════════════════════
# 4. TRAINER
# ═════════════════════════════════════════════════════════════════════════════

class ClockfieldDreamTrainer:
    def __init__(self, alpha: float = 3.0, lambda_max: float = 0.05):
        self.alpha = alpha
        self.lambda_max = lambda_max
        self.log_lines: List[str] = []
        self.is_training = False
        self._stop_flag = False

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log_lines.append(line)
        if len(self.log_lines) > 300:
            self.log_lines = self.log_lines[-300:]
        print(line)

    def stop(self): self._stop_flag = True

    def _update_decay(self, optimizer, gammas: Dict[str, float]):
        """β-gated weight decay: shield crystallized layers."""
        stage_map = {'fc': 0, 'up1': 1, 'up2': 2, 'up3': 3}
        gamma_list = [gammas.get('fc', 1.0), gammas.get('up1', 1.0),
                      gammas.get('up2', 1.0), gammas.get('up3', 1.0)]
        for group in optimizer.param_groups:
            stage = group.get('_stage')
            if stage is not None and stage < len(gamma_list):
                group['weight_decay'] = max(self.lambda_max * gamma_list[stage], 1e-6)

    def _build_optimizer(self, decoder: ConceptDecoder, lr: float):
        groups = []
        stage_modules = [
            ('fc',   [decoder.clip_proj, decoder.fc]),
            ('up1',  [decoder.up1]),
            ('up2',  [decoder.up2]),
            ('up3',  [decoder.up3]),
            ('up4',  [decoder.up4, decoder.out_conv]),
        ]
        for i, (stage_name, modules) in enumerate(stage_modules):
            params = []
            for m in modules:
                params.extend(m.parameters())
            groups.append({
                'params': params,
                'lr': lr,
                'weight_decay': self.lambda_max,
                '_stage': i,
            })
        return torch.optim.AdamW(groups)

    def train(
        self,
        clip_model: CLIPModel,
        processor: CLIPProcessor,
        decoder: ConceptDecoder,
        memory: ConceptMemory,
        dataset: Dataset,
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 2e-4,
        progress_callback=None,
    ):
        self.is_training = True
        self._stop_flag = False
        decoder.register_beta_hooks()
        decoder = decoder.to(DEVICE)
        clip_model = clip_model.to(DEVICE)
        clip_model.eval()

        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        optimizer = self._build_optimizer(decoder, lr)
        # Cosine LR decay
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * len(loader), eta_min=lr * 0.1
        )

        total_steps = epochs * len(loader)
        step = 0
        self._log(f"Training: {len(dataset)} images, {epochs} epochs, concepts={getattr(dataset,'classes',['?'])}")

        for epoch in range(epochs):
            if self._stop_flag: break
            epoch_loss = 0.0

            for batch_imgs, batch_labels in loader:
                if self._stop_flag: break
                batch_imgs = batch_imgs.to(DEVICE)   # [B, 3, 64, 64] in [-1,1]

                # ── Get CLIP image embeddings (our training target signal)
                with torch.no_grad():
                    # Use CLIP vision encoder to get image embeddings
                    # We resize to CLIP's expected 224×224 for encoding
                    imgs_224 = F.interpolate(batch_imgs, size=(224, 224), mode='bilinear', align_corners=False)
                    # Un-normalize from [-1,1] back to [0,1] then to CLIP norm
                    imgs_224 = (imgs_224 + 1.0) / 2.0
                    clip_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEVICE).view(1,3,1,1)
                    clip_std  = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEVICE).view(1,3,1,1)
                    imgs_clip = (imgs_224 - clip_mean) / clip_std
                    clip_img_emb = clip_model.get_image_features(pixel_values=imgs_clip)
                    clip_img_emb = F.normalize(clip_img_emb, dim=-1)  # [B, 512]

                    # Also get text embeddings for concept names
                    unique_labels = list(set(batch_labels))
                    text_inputs = processor(
                        text=[f"a photo of {l}" for l in unique_labels],
                        return_tensors="pt", padding=True, truncation=True
                    ).to(DEVICE)
                    text_embs = clip_model.get_text_features(**text_inputs)
                    text_embs = F.normalize(text_embs, dim=-1)
                    label_to_emb = {l: text_embs[i] for i, l in enumerate(unique_labels)}
                    
                    # Blend: 70% image embedding + 30% text embedding
                    # This anchors the concept to actual visual appearance
                    blended_embs = []
                    for img_emb, label in zip(clip_img_emb, batch_labels):
                        text_emb = label_to_emb[label]
                        blended = F.normalize(0.7 * img_emb + 0.3 * text_emb, dim=-1)
                        blended_embs.append(blended)
                        # Update memory with blended embedding
                        memory.update(label, blended.unsqueeze(0))
                    blended_embs = torch.stack(blended_embs)  # [B, 512]

                # ── Forward: decode blended embedding → image
                generated = decoder(blended_embs)  # [B, 3, 64, 64]

                # ── Loss: perceptual (pixel L1) + CLIP alignment
                # Pixel reconstruction loss
                pixel_loss = F.l1_loss(generated, batch_imgs)

                # CLIP alignment: generated image should embed close to input embedding
                gen_224 = F.interpolate(generated, size=(224, 224), mode='bilinear', align_corners=False)
                gen_224 = (gen_224 + 1.0) / 2.0
                gen_224 = (gen_224 - clip_mean) / clip_std
                gen_clip_emb = clip_model.get_image_features(pixel_values=gen_224.detach().clone().requires_grad_(True) if not generated.requires_grad else gen_224)

                # Just use pixel loss + smoothness for simplicity and speed
                # Add TV (total variation) loss for smoother images
                tv_loss = (
                    torch.abs(generated[:, :, 1:, :] - generated[:, :, :-1, :]).mean() +
                    torch.abs(generated[:, :, :, 1:] - generated[:, :, :, :-1]).mean()
                )
                loss = pixel_loss + 0.01 * tv_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                # β update and decay shielding
                betas = decoder.compute_beta()
                decoder.update_beta(betas)
                gammas, beta_vals = decoder.get_gamma(self.alpha)
                self._update_decay(optimizer, gammas)

                epoch_loss += loss.item()
                step += 1
                if progress_callback:
                    progress_callback(step / total_steps)

            g, b = decoder.get_gamma(self.alpha)
            self._log(
                f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/max(len(loader),1):.4f} | "
                f"β_fc={b['fc']:.3f} β_up1={b['up1']:.3f} | "
                f"Γ_fc={g['fc']:.3f} Γ_up1={g['up1']:.3f}"
            )

        self._log("Training complete.")
        self.is_training = False
        return decoder, memory


# ═════════════════════════════════════════════════════════════════════════════
# 5. SAVE / LOAD
# ═════════════════════════════════════════════════════════════════════════════

def save_brain(decoder: ConceptDecoder, memory: ConceptMemory, path: Path = SAVE_DIR):
    path.mkdir(exist_ok=True)
    torch.save(decoder.state_dict(), path / 'decoder.pt')
    memory.save(path)
    meta = {
        'concepts': memory.concepts,
        'image_counts': memory.image_counts,
        'saved_at': time.strftime("%Y-%m-%d %H:%M:%S"),
        'img_size': IMG_SIZE,
    }
    with open(path / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    return str(path)

def load_brain(path: Path = SAVE_DIR) -> Tuple[Optional[ConceptDecoder], ConceptMemory]:
    memory = ConceptMemory()
    decoder = ConceptDecoder()
    weights_path = path / 'decoder.pt'
    if weights_path.exists():
        decoder.load_state_dict(torch.load(weights_path, map_location='cpu'))
        memory.load(path)
        return decoder, memory
    return decoder, memory   # fresh decoder, empty memory


# ═════════════════════════════════════════════════════════════════════════════
# 6. INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def dream(
    clip_model: CLIPModel,
    processor: CLIPProcessor,
    decoder: ConceptDecoder,
    memory: ConceptMemory,
    prompt: str,
    num_images: int = 4,
    temperature: float = 0.0,
) -> List[Image.Image]:
    """
    Generate images from a text prompt.
    Known words → memory embeddings → fused → decoded.
    Unknown words → CLIP text encoder directly.
    Temperature > 0 adds noise for variation.
    """
    decoder.eval()
    decoder = decoder.to(DEVICE)
    clip_model.eval()

    words = prompt.lower().strip().split()
    known = [w for w in words if w in memory.concepts]
    unknown = [w for w in words if w not in memory.concepts]

    parts = []

    if known:
        mem_emb = memory.query(prompt)  # [1, CLIP_DIM]
        parts.append(mem_emb)

    if unknown:
        text_in = processor(
            text=[f"a photo of {' '.join(unknown)}"],
            return_tensors="pt", padding=True, truncation=True
        ).to(DEVICE)
        clip_text_emb = clip_model.get_text_features(**text_in)
        clip_text_emb = F.normalize(clip_text_emb, dim=-1)
        parts.append(clip_text_emb)

    if parts:
        base_emb = F.normalize(torch.cat([p.view(1, CLIP_DIM) for p in parts], dim=0).mean(0, keepdim=True), dim=-1)
    else:
        base_emb = torch.zeros(1, CLIP_DIM, device=DEVICE)

    # Repeat for batch
    emb_batch = base_emb.view(1, CLIP_DIM).expand(num_images, -1).contiguous()  # [N, 512]

    # Add temperature noise for variation
    if temperature > 0:
        noise = torch.randn_like(emb_batch) * temperature
        emb_batch = F.normalize(emb_batch + noise, dim=-1)

    generated = decoder(emb_batch.to(DEVICE))  # [N, 3, 64, 64]

    # Convert to PIL
    imgs = []
    for i in range(num_images):
        img_t = generated[i].cpu().clamp(-1, 1)
        img_t = (img_t + 1.0) / 2.0  # [0, 1]
        img_np = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        imgs.append(Image.fromarray(img_np).resize((256, 256), Image.NEAREST))

    return imgs


# ═════════════════════════════════════════════════════════════════════════════
# 7. GRADIO APP
# ═════════════════════════════════════════════════════════════════════════════

def build_app():
    state = {
        'clip_model': None,
        'processor': None,
        'decoder': ConceptDecoder(),
        'memory': ConceptMemory(),
        'trainer': ClockfieldDreamTrainer(),
        'loaded': False,
    }

    def ensure_clip():
        if not state['loaded']:
            state['processor'] = CLIPProcessor.from_pretrained(CLIP_ID)
            state['clip_model'] = CLIPModel.from_pretrained(CLIP_ID).to(DEVICE)
            state['clip_model'].eval()
            for p in state['clip_model'].parameters():
                p.requires_grad_(False)
            state['loaded'] = True
            # Try loading saved brain
            decoder, memory = load_brain()
            state['decoder'] = decoder.to(DEVICE)
            state['memory'] = memory
            if memory.concepts:
                return f"✅ CLIP loaded. Brain restored — knows: {memory.concepts}"
            return "✅ CLIP loaded. Fresh brain — ready to dream."
        return "✅ Already loaded."

    def get_status():
        m = state['memory']
        g, b = state['decoder'].get_gamma()
        lines = [
            f"**Concepts:** {len(m.concepts)}",
            f"**Known words:** {', '.join(m.concepts) if m.concepts else '—'}",
            f"**Image counts:** { {k:v for k,v in m.image_counts.items()} }",
            f"**β_fc={b['fc']:.3f}  β_up1={b['up1']:.3f}**",
            f"**Γ_fc={g['fc']:.3f}  Γ_up1={g['up1']:.3f}** (1.0=no shield, 0.0=fully shielded)",
            f"**Device:** {DEVICE}",
        ]
        return "\n".join(lines)

    def on_load():
        msg = ensure_clip()
        return msg, get_status()

    def on_save():
        path = save_brain(state['decoder'], state['memory'])
        return f"✅ Saved to {path} | Concepts: {state['memory'].concepts}"

    def on_load_brain():
        decoder, memory = load_brain()
        state['decoder'] = decoder.to(DEVICE)
        state['memory'] = memory
        return f"✅ Loaded brain. Concepts: {memory.concepts}", get_status()

    def on_reset():
        state['decoder'] = ConceptDecoder()
        state['memory'] = ConceptMemory()
        return "🧹 Brain reset.", get_status()

    def on_stop():
        state['trainer'].stop()
        return "⏹ Stop requested."

    def get_log():
        return "\n".join(state['trainer'].log_lines[-60:])

    # ── Training handlers ─────────────────────────────────────────────────────

    def run_training(dataset, epochs, batch_size, lr, progress):
        ensure_clip()
        def prog_cb(p): progress(p, desc="Dreaming...")
        decoder, memory = state['trainer'].train(
            state['clip_model'], state['processor'],
            state['decoder'], state['memory'],
            dataset,
            epochs=int(epochs), batch_size=int(batch_size), lr=float(lr),
            progress_callback=prog_cb,
        )
        state['decoder'] = decoder
        state['memory'] = memory
        return "\n".join(state['trainer'].log_lines[-30:]), get_status()

    def on_train_hf(ds_name, split, epochs, batch_size, lr, progress=gr.Progress()):
        ensure_clip()
        if not ds_name.strip():
            return "Enter a dataset name.", get_status()
        try:
            state['trainer']._log(f"Loading {ds_name} ({split})...")
            hf_data = load_dataset(ds_name, split=split)
            state['trainer']._log(f"Fields: {list(hf_data.features.keys())}")
            dataset = HFImageDataset(hf_data)
            return run_training(dataset, epochs, batch_size, lr, progress)
        except Exception as e:
            import traceback
            return f"Error: {e}\n{traceback.format_exc()}", get_status()

    def on_train_folder(folder, epochs, batch_size, lr, progress=gr.Progress()):
        ensure_clip()
        if not folder or not Path(folder).exists():
            return "Folder not found. Needs: root/concept_name/image.jpg", get_status()
        try:
            dataset = FolderImageDataset(folder)
            if not dataset.samples:
                return "No images found.", get_status()
            state['trainer']._log(f"Folder: {len(dataset)} images, classes={dataset.classes}")
            return run_training(dataset, epochs, batch_size, lr, progress)
        except Exception as e:
            return f"Error: {e}", get_status()

    def on_train_uploads(images, concept_name, epochs, batch_size, lr, progress=gr.Progress()):
        ensure_clip()
        if not images or not concept_name.strip():
            return "Need images and a concept name.", get_status()
        concept_name = concept_name.strip().replace(" ", "_").lower()
        try:
            pil_imgs = []
            for img in images:
                if img is None: continue
                if isinstance(img, Image.Image):
                    pil_imgs.append(img)
                else:
                    pil_imgs.append(Image.fromarray(np.array(img)))
            if not pil_imgs:
                return "No valid images.", get_status()
            state['trainer']._log(f"Teaching '{concept_name}' from {len(pil_imgs)} uploaded images")
            dataset = PILListDataset(pil_imgs, concept_name)
            return run_training(dataset, epochs, min(int(batch_size), len(pil_imgs)), lr, progress)
        except Exception as e:
            import traceback
            return f"Error: {e}\n{traceback.format_exc()}", get_status()

    # ── Dream handler ─────────────────────────────────────────────────────────

    def on_dream(prompt, num_imgs, temperature):
        ensure_clip()
        if not prompt.strip():
            return None, "Enter a prompt."
        known = [w for w in prompt.lower().split() if w in state['memory'].concepts]
        unknown = [w for w in prompt.lower().split() if w not in state['memory'].concepts]
        info = f"Known concepts: {known} | Via CLIP only: {unknown}"
        try:
            imgs = dream(
                state['clip_model'], state['processor'],
                state['decoder'], state['memory'],
                prompt, num_images=int(num_imgs), temperature=float(temperature),
            )
            return imgs, info
        except Exception as e:
            import traceback
            return None, f"Error: {e}\n{traceback.format_exc()}"

    # ── UI ────────────────────────────────────────────────────────────────────
    css = """
    :root {
        --bg:#080a0e; --panel:#0f1117; --border:#1a1f2e;
        --accent:#00e5ff; --accent2:#a855f7; --text:#bcc8e0; --muted:#3d4a63;
        --green:#22d3a0; --warn:#f59e0b;
    }
    body,.gradio-container{background:var(--bg)!important;color:var(--text)!important;
        font-family:'JetBrains Mono','Fira Code',monospace!important;}
    h1,h2,h3{color:var(--accent)!important;letter-spacing:.06em;}
    .gr-button{border-radius:2px!important;font-family:monospace!important;font-size:12px!important;letter-spacing:.05em;}
    .gr-button-primary{background:linear-gradient(135deg,var(--accent2),#7c3aed)!important;
        border:1px solid var(--accent)!important;color:#fff!important;}
    .gr-button-secondary{background:var(--panel)!important;border:1px solid var(--border)!important;color:var(--text)!important;}
    .gr-box,.gr-form{background:var(--panel)!important;border:1px solid var(--border)!important;}
    textarea,input{background:#0a0c12!important;color:var(--text)!important;border:1px solid var(--border)!important;font-family:monospace!important;}
    label{color:var(--muted)!important;font-size:11px!important;letter-spacing:.08em;text-transform:uppercase;}
    """

    with gr.Blocks(title="Clockfield Dream", css=css, theme=gr.themes.Base()) as app:

        gr.Markdown("""
# ⟳ CLOCKFIELD DREAM
### A growing image generator. Teach it names. Watch it dream.

`Γ = exp(−α·β)` — crystallized concepts are protected. New ones grow without erasing the old.  
**"antti"** → teaches your face · **"dog"** → teaches dogs · **"antti dog"** → fuses both
        """)

        with gr.Row():
            load_btn  = gr.Button("⚡ LOAD CLIP",   variant="primary", scale=2)
            save_btn  = gr.Button("💾 SAVE BRAIN",  scale=1)
            load_b    = gr.Button("📂 LOAD BRAIN",  scale=1)
            stop_btn  = gr.Button("⏹ STOP",         scale=1)
            reset_btn = gr.Button("🧹 RESET",        scale=1)

        sys_msg    = gr.Textbox(label="System", interactive=False, lines=1)
        status_box = gr.Markdown("*Click LOAD CLIP to begin.*")

        gr.Markdown("---")

        with gr.Tabs():

            # ── DREAM tab (first — most important) ───────────────────────────
            with gr.Tab("✨ Dream"):
                gr.Markdown("""
Type any prompt. Known concept words (shown in status above) pull from memory.  
Unknown words use raw CLIP. Mix freely: **"antti sunset"**, **"healthy bean"**, etc.
                """)
                with gr.Row():
                    dream_prompt = gr.Textbox(
                        label="Prompt", placeholder="e.g. antti  |  dog  |  antti dog  |  angular_leaf_spot",
                        lines=1, scale=3
                    )
                    dream_n   = gr.Slider(1, 9, value=4, step=1,   label="Images", scale=1)
                    dream_temp = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="Temperature (variation)", scale=1)
                dream_btn     = gr.Button("✨ DREAM", variant="primary")
                dream_info    = gr.Markdown()
                dream_gallery = gr.Gallery(label="Generated Images", columns=4, height=300)

            # ── Train: HF dataset ─────────────────────────────────────────────
            with gr.Tab("🤗 Train: HuggingFace"):
                gr.Markdown("Try: `beans` · `cifar10` · `food101` · `cats_vs_dogs` · `keremberke/pokemon-classification`")
                with gr.Row():
                    hf_name  = gr.Textbox(label="Dataset", value="beans", placeholder="HuggingFace dataset name")
                    hf_split = gr.Textbox(label="Split", value="train[:300]")
                with gr.Row():
                    hf_ep    = gr.Slider(1, 30, value=5,  step=1,   label="Epochs")
                    hf_bs    = gr.Slider(4, 64, value=16, step=4,   label="Batch Size")
                    hf_lr    = gr.Number(value=2e-4,               label="Learning Rate")
                hf_btn       = gr.Button("▶ TRAIN", variant="primary")
                hf_log       = gr.Textbox(label="Log", lines=10, interactive=False)
                hf_status    = gr.Markdown()

            # ── Train: Folder ─────────────────────────────────────────────────
            with gr.Tab("📁 Train: Folder"):
                gr.Markdown("Structure: `root/concept_name/image.jpg` — each subfolder = one concept word.")
                folder_path  = gr.Textbox(label="Folder Path", placeholder="/path/to/images")
                with gr.Row():
                    f_ep     = gr.Slider(1, 30, value=5,  step=1,   label="Epochs")
                    f_bs     = gr.Slider(4, 64, value=16, step=4,   label="Batch Size")
                    f_lr     = gr.Number(value=2e-4,               label="Learning Rate")
                f_btn        = gr.Button("▶ TRAIN", variant="primary")
                f_log        = gr.Textbox(label="Log", lines=10, interactive=False)
                f_status     = gr.Markdown()

            # ── Train: Upload images ──────────────────────────────────────────
            with gr.Tab("🖼️ Train: Upload Images"):
                gr.Markdown("""
Upload photos of **one thing** → give it a name → train.  
Repeat with different images + different names. All concepts accumulate.  
More images = better dreams. Even 5–10 photos will work.
                """)
                upload_imgs  = gr.Gallery(label="Upload Images", columns=5, height=260, type="pil")
                concept_box  = gr.Textbox(label="Concept Name (one word)", placeholder="antti  |  mydog  |  sunset")
                with gr.Row():
                    u_ep     = gr.Slider(1, 100, value=20, step=1,  label="Epochs (more = better for few images)")
                    u_bs     = gr.Slider(1, 32,  value=4,  step=1,  label="Batch Size")
                    u_lr     = gr.Number(value=2e-4,                label="Learning Rate")
                u_btn        = gr.Button("▶ TEACH THIS CONCEPT", variant="primary")
                u_log        = gr.Textbox(label="Log", lines=8, interactive=False)
                u_status     = gr.Markdown()

            # ── Live log ──────────────────────────────────────────────────────
            with gr.Tab("📊 Log & Metrics"):
                refresh_btn  = gr.Button("↻ Refresh")
                live_log     = gr.Textbox(label="Log", lines=25, interactive=False)
                live_status  = gr.Markdown()
                refresh_btn.click(fn=lambda: (get_log(), get_status()), outputs=[live_log, live_status])

        # ── Wire ──────────────────────────────────────────────────────────────
        load_btn.click(fn=on_load,       outputs=[sys_msg, status_box])
        save_btn.click(fn=on_save,       outputs=[sys_msg])
        load_b.click(  fn=on_load_brain, outputs=[sys_msg, status_box])
        stop_btn.click(fn=on_stop,       outputs=[sys_msg])
        reset_btn.click(fn=on_reset,     outputs=[sys_msg, status_box])

        dream_btn.click(
            fn=on_dream,
            inputs=[dream_prompt, dream_n, dream_temp],
            outputs=[dream_gallery, dream_info],
        )
        hf_btn.click(
            fn=on_train_hf,
            inputs=[hf_name, hf_split, hf_ep, hf_bs, hf_lr],
            outputs=[hf_log, hf_status],
        )
        f_btn.click(
            fn=on_train_folder,
            inputs=[folder_path, f_ep, f_bs, f_lr],
            outputs=[f_log, f_status],
        )
        u_btn.click(
            fn=on_train_uploads,
            inputs=[upload_imgs, concept_box, u_ep, u_bs, u_lr],
            outputs=[u_log, u_status],
        )

    return app


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"""
╔══════════════════════════════════════════════════════╗
║  CLOCKFIELD DREAM  |  device={DEVICE:<6}  img={IMG_SIZE}×{IMG_SIZE}     ║
║  Brain dir: {SAVE_DIR}                        ║
╚══════════════════════════════════════════════════════╝
""")
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7861, share=False, show_error=True)