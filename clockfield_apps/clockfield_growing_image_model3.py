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
IMG_SIZE   = 128         # 128×128 — good quality, still fits 12GB
CLIP_DIM   = 512         # CLIP embedding dim
LATENT_DIM = 384         # larger latent for better capacity

# ═════════════════════════════════════════════════════════════════════════════
# 1. THE DECODER — small CNN that turns a concept vector into pixels
# ═════════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """Residual block with β measurement hook."""
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
            nn.GELU(),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(x + self.net(x))


class VGGPerceptual(nn.Module):
    """Frozen VGG16 feature extractor for perceptual loss."""
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        # Use relu1_2, relu2_2, relu3_3 features
        self.slice1 = nn.Sequential(*list(vgg.features)[:4])
        self.slice2 = nn.Sequential(*list(vgg.features)[4:9])
        self.slice3 = nn.Sequential(*list(vgg.features)[9:16])
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        # x expected in [-1,1]; VGG expects ImageNet norm
        mean = torch.tensor([0.485,0.456,0.406], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.229,0.224,0.225], device=x.device).view(1,3,1,1)
        x = ((x + 1.0) / 2.0 - mean) / std
        f1 = self.slice1(x)
        f2 = self.slice2(f1)
        f3 = self.slice3(f2)
        return [f1, f2, f3]


class ConceptDecoder(nn.Module):
    """
    CLIP embedding → 128×128 RGB image.
    Architecture:
      FC proj → 4×4×512 seed
      6× (Upsample + ResBlock) tower: 4→8→16→32→64→128
      Per-stage β measurement → Γ shielding
    """
    def __init__(self, latent_dim: int = LATENT_DIM, img_size: int = IMG_SIZE):
        super().__init__()
        self.img_size  = img_size
        self.latent_dim = latent_dim

        self.clip_proj = nn.Sequential(
            nn.Linear(CLIP_DIM, latent_dim),
            nn.GELU(),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )

        self.fc = nn.Linear(latent_dim, 512 * 4 * 4)

        def up_res(in_ch, out_ch):
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                ResBlock(out_ch),
            )

        # 4→8→16→32→64→128
        self.up1 = up_res(512, 256)
        self.up2 = up_res(256, 128)
        self.up3 = up_res(128,  64)
        self.up4 = up_res( 64,  32)
        self.up5 = up_res( 32,  16)
        self.out_conv = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16,  3, 1),
        )

        # β EMA per stage — more granular than before
        for stage in ['proj','fc','up1','up2','up3','up4','up5']:
            self.register_buffer(f'beta_{stage}', torch.tensor(0.5))

        self._acts  = {}
        self._hooks = []

    def register_beta_hooks(self):
        for h in self._hooks: h.remove()
        self._hooks = []
        def hook(name):
            def fn(m, inp, out):
                self._acts[name] = out.detach()
            return fn
        self._hooks = [
            self.clip_proj.register_forward_hook(hook('proj')),
            self.fc.register_forward_hook(hook('fc')),
            self.up1.register_forward_hook(hook('up1')),
            self.up2.register_forward_hook(hook('up2')),
            self.up3.register_forward_hook(hook('up3')),
            self.up4.register_forward_hook(hook('up4')),
            self.up5.register_forward_hook(hook('up5')),
        ]

    def _roughness(self, acts: torch.Tensor) -> float:
        flat = acts.view(acts.shape[0], -1)
        if flat.shape[0] > 1:
            norm = (flat - flat.mean(0, keepdim=True)) / (flat.std(0, keepdim=True) + 1e-6)
            return torch.abs(torch.diff(norm, dim=0)).mean().item()
        return 0.5

    def compute_beta(self) -> Dict[str, float]:
        return {name: self._roughness(acts) for name, acts in self._acts.items()}

    def update_beta(self, betas: Dict[str, float]):
        for name, val in betas.items():
            buf_name = f'beta_{name}'
            if hasattr(self, buf_name):
                old = getattr(self, buf_name)
                setattr(self, buf_name, 0.9 * old + 0.1 * val)

    def get_gamma(self, alpha: float = 3.0) -> Tuple[Dict[str,float], Dict[str,float]]:
        betas, gammas = {}, {}
        for stage in ['proj','fc','up1','up2','up3','up4','up5']:
            b = getattr(self, f'beta_{stage}').item()
            betas[stage]  = b
            gammas[stage] = float(np.exp(-alpha * b))
        return gammas, betas

    def forward(self, clip_emb: torch.Tensor) -> torch.Tensor:
        x = self.clip_proj(clip_emb)       # [B, latent]
        x = self.fc(x)                     # [B, 512*4*4]
        x = x.view(-1, 512, 4, 4)
        x = self.up1(x)                    # [B, 256,  8,  8]
        x = self.up2(x)                    # [B, 128, 16, 16]
        x = self.up3(x)                    # [B,  64, 32, 32]
        x = self.up4(x)                    # [B,  32, 64, 64]
        x = self.up5(x)                    # [B,  16,128,128]
        return torch.tanh(self.out_conv(x))# [B,   3,128,128]

# ═════════════════════════════════════════════════════════════════════════════
# 2. CONCEPT MEMORY — stores per-concept CLIP embeddings and generated prototypes
# ═════════════════════════════════════════════════════════════════════════════

class ConceptMemory:
    """
    Multi-prototype concept memory.
    Each concept stores up to N_PROTO embeddings (not just a mean).
    At dream time, we sample one prototype (or average all for stability).
    Running centroid also maintained for fast lookup.
    """
    N_PROTO = 8

    def __init__(self):
        self.prototypes:   Dict[str, List[torch.Tensor]] = {}  # concept → list of [1,512]
        self.centroids:    Dict[str, torch.Tensor]        = {}  # concept → mean [1,512]
        self.image_counts: Dict[str, int]                 = {}
        self.beta_shields: Dict[str, float]               = {}

    def update(self, concept: str, emb: torch.Tensor, gamma: float = 1.0):
        emb = emb.detach().cpu().view(1, CLIP_DIM)
        if concept not in self.prototypes:
            self.prototypes[concept]   = []
            self.image_counts[concept] = 0

        # Store prototype (up to N_PROTO, then replace oldest)
        if len(self.prototypes[concept]) < self.N_PROTO:
            self.prototypes[concept].append(emb.clone())
        else:
            idx = self.image_counts[concept] % self.N_PROTO
            self.prototypes[concept][idx] = emb.clone()

        # Update centroid EMA
        if concept not in self.centroids:
            self.centroids[concept] = emb.clone()
        else:
            self.centroids[concept] = (0.85 * self.centroids[concept].view(1,CLIP_DIM)
                                       + 0.15 * emb).view(1, CLIP_DIM)
        self.image_counts[concept] = self.image_counts[concept] + 1
        self.beta_shields[concept] = gamma

    def query(self, prompt: str, device: str = DEVICE, sample: bool = False) -> torch.Tensor:
        """
        Fuse matching concept embeddings.
        sample=True: pick a random prototype per concept (more variation).
        sample=False: use centroid (more stable).
        """
        words   = prompt.lower().strip().split()
        matched = []
        for word in words:
            if word in self.centroids:
                if sample and word in self.prototypes and self.prototypes[word]:
                    proto = self.prototypes[word]
                    matched.append(proto[torch.randint(len(proto),(1,)).item()])
                else:
                    matched.append(self.centroids[word])
        if not matched:
            return torch.zeros(1, CLIP_DIM, device=device)
        fused = torch.cat([e.view(1,CLIP_DIM) for e in matched], dim=0).mean(0, keepdim=True)
        return F.normalize(fused, dim=-1).to(device)

    @property
    def concepts(self) -> List[str]:
        return list(self.centroids.keys())

    def apply_semantic_gravity(self, co_occurrence: Dict[str, float], strength: float = 0.02):
        """
        Semantic gravity: concepts that appear together in prompts
        drift toward each other in prototype space.
        Strength is modulated by Γ — crystallized concepts resist gravity.
        
        co_occurrence: dict of "conceptA+conceptB" -> count
        """
        for pair, count in co_occurrence.items():
            if '+' not in pair: continue
            a, b = pair.split('+', 1)
            if a not in self.centroids or b not in self.centroids: continue
            # Γ of each concept — how much it resists being pulled
            gamma_a = float(np.exp(-3.0 * self.beta_shields.get(a, 0.5)))
            gamma_b = float(np.exp(-3.0 * self.beta_shields.get(b, 0.5)))
            # Pull strength modulated: crystallized (low Γ) resist, fluid (high Γ) bend
            pull = strength * min(count / 10.0, 1.0)
            ca = self.centroids[a].view(1, CLIP_DIM)
            cb = self.centroids[b].view(1, CLIP_DIM)
            # Each drifts toward midpoint, scaled by other's resistance
            mid = F.normalize((ca + cb) / 2.0, dim=-1)
            self.centroids[a] = F.normalize(ca + pull * (1.0 - gamma_a) * (mid - ca), dim=-1)
            self.centroids[b] = F.normalize(cb + pull * (1.0 - gamma_b) * (mid - cb), dim=-1)

    def save(self, path: Path):
        data = {
            'centroids':    {k: v.tolist() for k,v in self.centroids.items()},
            'prototypes':   {k: [p.tolist() for p in v] for k,v in self.prototypes.items()},
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
        self.centroids    = {k: torch.tensor(v) for k,v in data.get('centroids',{}).items()}
        self.prototypes   = {k: [torch.tensor(p) for p in v]
                             for k,v in data.get('prototypes',{}).items()}
        self.image_counts = data.get('image_counts', {})
        self.beta_shields = data.get('beta_shields', {})
        # back-compat: old saves had 'embeddings' key
        if not self.centroids and 'embeddings' in data:
            self.centroids = {k: torch.tensor(v) for k,v in data['embeddings'].items()}

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
# 3b. LATENT DYNAMICS + MEMORY GRAPH
# ═════════════════════════════════════════════════════════════════════════════

class LatentDynamics(nn.Module):
    """
    Learns z_t → z_{t+1}: how the latent space evolves across consecutive
    training samples. Forces the latent to encode *trajectory*, not just
    static appearance.

    Clockfield connection:
    - β is measured on this module independently
    - Low β (smooth dynamics) → Γ → 0 → maximally shielded: stable attractor
    - High β (turbulent dynamics) → Γ → 1 → trainable: still learning structure
    - Residual design: z_out = z_in + f(z_in), so identity is always preserved
    """
    def __init__(self, dim: int = CLIP_DIM, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.norm = nn.LayerNorm(dim)
        # β EMA buffer
        self.register_buffer('beta_ema', torch.tensor(0.5))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Predict next latent. z: [B, CLIP_DIM], normalized."""
        delta = self.net(z)
        return F.normalize(self.norm(z + 0.1 * delta), dim=-1)

    def compute_beta(self) -> float:
        """Roughness of the dynamics module activations."""
        if not hasattr(self, '_acts') or self._acts is None:
            return float(self.beta_ema)
        acts = self._acts
        if acts.numel() < 2:
            return float(self.beta_ema)
        roughness = float((acts - acts.mean()).abs().mean() / (acts.std() + 1e-8))
        new_beta = 0.9 * float(self.beta_ema) + 0.1 * roughness
        self.beta_ema.fill_(new_beta)
        return new_beta

    def register_hook(self):
        self._acts = None
        def hook(m, inp, out):
            self._acts = out.detach()
        self.net[-1].register_forward_hook(hook)

    def prediction_loss(self, z_now: torch.Tensor, z_next: torch.Tensor) -> torch.Tensor:
        """
        MSE between predicted next latent and actual next latent.
        z_now, z_next: [B, CLIP_DIM] normalized embeddings from consecutive samples.
        """
        z_pred = self.forward(z_now)
        return F.mse_loss(z_pred, z_next.detach())


class MemoryGraph:
    """
    Growing cluster graph — the model's unlabeled visual vocabulary.

    Each node:
        centroid    [CLIP_DIM]  — mean embedding of cluster
        stability   float       — Γ of that cluster (high = crystallized)
        count       int         — how many embeddings have strengthened it
        label       str|None    — optional human label if known
        neighbors   Set[int]    — node IDs that frequently co-activate

    Growth rule:
        if min_dist(emb, all_nodes) > BIRTH_THRESH → create new node
        else → strengthen nearest node (EMA centroid update)

    Death rule:
        nodes with count < MIN_COUNT and age > MAX_AGE are pruned
        (prevents noise from creating permanent nodes)

    Clockfield connection:
        stability = exp(-alpha * beta_of_that_cluster)
        Stable nodes resist centroid drift (EMA rate scales with 1 - stability)
        Novel inputs that birth new nodes start with high beta (unstable)
    """
    BIRTH_THRESH  = 0.25   # cosine distance to nearest node to trigger birth
    MAX_NODES     = 128    # hard cap
    MIN_COUNT     = 3      # nodes below this get pruned if old enough
    MAX_AGE       = 200    # steps before an underused node is pruned
    EMA_BASE      = 0.05   # centroid update rate for fully fluid node

    def __init__(self, alpha: float = 3.0):
        self.alpha     = alpha
        self.centroids : List[torch.Tensor] = []   # list of [CLIP_DIM] tensors (CPU)
        self.stabilities: List[float]        = []
        self.counts    : List[int]           = []
        self.ages      : List[int]           = []  # steps since last update
        self.labels    : List[Optional[str]] = []
        self.neighbors : List[set]           = []
        self._last_hit : Optional[int]       = None  # last node touched

    def _cosine_dist(self, a: torch.Tensor, b: torch.Tensor) -> float:
        return 1.0 - float(F.cosine_similarity(a.view(1,-1), b.view(1,-1)))

    def _nearest(self, emb: torch.Tensor):
        """Returns (node_idx, distance) of nearest node, or (None, inf) if empty."""
        if not self.centroids:
            return None, float('inf')
        emb_n = F.normalize(emb.view(1, -1).cpu(), dim=-1)
        dists = [self._cosine_dist(emb_n, F.normalize(c.view(1,-1), dim=-1))
                 for c in self.centroids]
        idx = int(np.argmin(dists))
        return idx, dists[idx]

    def update(self, emb: torch.Tensor, label: Optional[str] = None,
               gamma: float = 1.0):
        """
        Feed one embedding. Either strengthens an existing node or births a new one.
        Returns (node_idx, is_new_birth).
        """
        emb_cpu = F.normalize(emb.detach().cpu().view(CLIP_DIM), dim=-1)
        idx, dist = self._nearest(emb_cpu)

        # ── Age all nodes ────────────────────────────────────────────────────
        for i in range(len(self.ages)):
            self.ages[i] += 1

        if dist > self.BIRTH_THRESH and len(self.centroids) < self.MAX_NODES:
            # ── Birth new node ───────────────────────────────────────────────
            self.centroids.append(emb_cpu.clone())
            self.stabilities.append(0.1)   # starts unstable
            self.counts.append(1)
            self.ages.append(0)
            self.labels.append(label)
            self.neighbors.append(set())
            new_idx = len(self.centroids) - 1
            # Record neighbor relationship
            if idx is not None:
                self.neighbors[new_idx].add(idx)
                self.neighbors[idx].add(new_idx)
            self._last_hit = new_idx
            return new_idx, True
        elif idx is not None:
            # ── Strengthen existing node ─────────────────────────────────────
            stab = self.stabilities[idx]
            ema_rate = self.EMA_BASE * (1.0 - stab)  # stable → slow drift
            self.centroids[idx] = F.normalize(
                (1 - ema_rate) * self.centroids[idx] + ema_rate * emb_cpu, dim=-1
            )
            self.stabilities[idx] = min(0.99,
                0.9 * stab + 0.1 * float(np.exp(-self.alpha * (1.0 - gamma)))
            )
            self.counts[idx]  += 1
            self.ages[idx]     = 0
            if label and not self.labels[idx]:
                self.labels[idx] = label
            # Update neighbor link to last_hit
            if self._last_hit is not None and self._last_hit != idx:
                self.neighbors[idx].add(self._last_hit)
                self.neighbors[self._last_hit].add(idx)
            self._last_hit = idx
            return idx, False
        return None, False

    def prune(self):
        """Remove weak old nodes."""
        keep = [i for i in range(len(self.centroids))
                if self.counts[i] >= self.MIN_COUNT or self.ages[i] < self.MAX_AGE]
        if len(keep) == len(self.centroids):
            return 0
        pruned = len(self.centroids) - len(keep)
        remap = {old: new for new, old in enumerate(keep)}
        self.centroids   = [self.centroids[i]    for i in keep]
        self.stabilities = [self.stabilities[i]  for i in keep]
        self.counts      = [self.counts[i]        for i in keep]
        self.ages        = [self.ages[i]          for i in keep]
        self.labels      = [self.labels[i]        for i in keep]
        self.neighbors   = [
            {remap[n] for n in self.neighbors[i] if n in remap}
            for i in keep
        ]
        return pruned

    def nearest_label(self, emb: torch.Tensor) -> Optional[str]:
        """Return the label of the nearest node (if labeled)."""
        idx, _ = self._nearest(emb)
        if idx is None: return None
        return self.labels[idx]

    def summary(self) -> str:
        n = len(self.centroids)
        labeled = sum(1 for l in self.labels if l)
        avg_stab = float(np.mean(self.stabilities)) if self.stabilities else 0.0
        top = sorted(zip(self.counts, self.labels), reverse=True)[:6]
        top_str = ", ".join(f"{l or '?'}×{c}" for c,l in top)
        return (f"MemoryGraph: {n} nodes ({labeled} labeled), "
                f"avg_stability={avg_stab:.3f} | top: {top_str}")

    def save(self, path: Path):
        data = {
            'centroids':    [c.tolist() for c in self.centroids],
            'stabilities':  self.stabilities,
            'counts':       self.counts,
            'ages':         self.ages,
            'labels':       self.labels,
            'neighbors':    [list(n) for n in self.neighbors],
        }
        with open(path / 'memory_graph.json', 'w') as f:
            json.dump(data, f)

    def load(self, path: Path):
        p = path / 'memory_graph.json'
        if not p.exists(): return
        with open(p) as f:
            d = json.load(f)
        self.centroids   = [torch.tensor(c) for c in d.get('centroids', [])]
        self.stabilities = d.get('stabilities', [])
        self.counts      = d.get('counts', [])
        self.ages        = d.get('ages', [0]*len(self.centroids))
        self.labels      = d.get('labels', [None]*len(self.centroids))
        self.neighbors   = [set(n) for n in d.get('neighbors', [[] for _ in self.centroids])]


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
        # Latent dynamics and memory graph — persist across training sessions
        self.dynamics = LatentDynamics()
        self.graph    = MemoryGraph(alpha=alpha)

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self.log_lines.append(line)
        if len(self.log_lines) > 300:
            self.log_lines = self.log_lines[-300:]
        print(line)

    def stop(self): self._stop_flag = True

    def _update_decay(self, optimizer, gammas: Dict[str, float]):
        """β-gated weight decay — one shield per decoder stage + dynamics."""
        stages = ['proj','fc','up1','up2','up3','up4','up5','dynamics']
        gamma_list = [gammas.get(s, 1.0) for s in stages]
        # dynamics gets its own beta
        dyn_beta = self.dynamics.compute_beta()
        dyn_gamma = float(np.exp(-self.alpha * dyn_beta))
        gamma_list[7] = dyn_gamma
        for group in optimizer.param_groups:
            stage = group.get('_stage')
            if stage is not None and stage < len(gamma_list):
                group['weight_decay'] = max(self.lambda_max * gamma_list[stage], 1e-6)

    def _build_optimizer(self, decoder: ConceptDecoder, lr: float):
        stage_modules = [
            ('proj', [decoder.clip_proj]),
            ('fc',   [decoder.fc]),
            ('up1',  [decoder.up1]),
            ('up2',  [decoder.up2]),
            ('up3',  [decoder.up3]),
            ('up4',  [decoder.up4]),
            ('up5',  [decoder.up5, decoder.out_conv]),
        ]
        groups = []
        for i, (stage_name, modules) in enumerate(stage_modules):
            params = []
            for m in modules:
                params.extend(m.parameters())
            groups.append({
                'params': params,
                'lr': lr,
                'weight_decay': self.lambda_max,
                '_stage': i,
                '_stage_name': stage_name,
            })
        # Dynamics gets its own group — lower LR, separate β shield
        groups.append({
            'params': list(self.dynamics.parameters()),
            'lr': lr * 0.5,
            'weight_decay': self.lambda_max,
            '_stage': 7,
            '_stage_name': 'dynamics',
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
        self.dynamics = self.dynamics.to(DEVICE)
        self.dynamics.register_hook()
        clip_model = clip_model.to(DEVICE)
        clip_model.eval()

        # Perceptual loss — load VGG once per training session
        vgg = VGGPerceptual().to(DEVICE)
        vgg.eval()

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

                    # ── Memory graph update (no grad needed) ──────────────────
                    for emb, label in zip(blended_embs, batch_labels):
                        node_idx, is_new = self.graph.update(
                            emb, label=label,
                            gamma=float(np.exp(-self.alpha * 0.5))  # initial gamma
                        )

                # ── Forward: decode blended embedding → image
                generated = decoder(blended_embs)  # [B, 3, 128, 128]

                # ── Loss: pixel L1 + perceptual (VGG) + TV + dynamics prediction
                pixel_loss = F.l1_loss(generated, batch_imgs)

                with torch.no_grad():
                    real_feats = vgg(batch_imgs)
                gen_feats = vgg(generated)
                perc_loss = sum(F.l1_loss(g, r.detach())
                                for g, r in zip(gen_feats, real_feats)) / 3.0

                tv_loss = (
                    torch.abs(generated[:,:,1:,:] - generated[:,:,:-1,:]).mean() +
                    torch.abs(generated[:,:,:,1:] - generated[:,:,:,:-1]).mean()
                )

                # ── Dynamics prediction loss: z_t → z_{t+1} ─────────────────
                # Pairs: first half predicts second half of batch
                dyn_loss = torch.tensor(0.0, device=DEVICE)
                if blended_embs.shape[0] >= 2:
                    half = blended_embs.shape[0] // 2
                    z_now  = blended_embs[:half]
                    z_next = blended_embs[half:half*2]
                    dyn_loss = self.dynamics.prediction_loss(z_now, z_next)

                loss = pixel_loss + 0.1 * perc_loss + 0.005 * tv_loss + 0.05 * dyn_loss

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

            # Prune weak memory graph nodes each epoch
            pruned = self.graph.prune()
            g, b = decoder.get_gamma(self.alpha)
            dyn_b = self.dynamics.compute_beta()
            self._log(
                f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/max(len(loader),1):.4f} "
                f"(dyn={dyn_loss.item():.4f}) | "
                f"β_proj={b['proj']:.3f} β_up4={b['up4']:.3f} β_dyn={dyn_b:.3f} | "
                f"Γ_proj={g['proj']:.3f} Γ_up4={g['up4']:.3f}"
            )
            self._log(self.graph.summary() + (f" | pruned {pruned}" if pruned else ""))

        self._log("Training complete.")
        self.is_training = False
        return decoder, memory


# ═════════════════════════════════════════════════════════════════════════════
# 4b. DREAM REPLAY — self-training loop (generative replay)
# ═════════════════════════════════════════════════════════════════════════════

class DreamReplay:
    """
    Generates images from existing concepts, then trains the decoder on
    those generated images. The model refines its own visual world.

    Clockfield connection:
    - During replay, β is LOW (model knows these concepts already)
    - So Γ → 0 → weight decay → near zero → those weights are MAXIMALLY shielded
    - Replay reinforces without overwriting — exactly the stability-plasticity ideal
    
    Co-occurrence tracking for semantic gravity:
    - Every prompt used during replay is logged
    - Concepts that dream together drift together
    """
    def __init__(self, alpha: float = 3.0):
        self.alpha = alpha
        self.log_lines: List[str] = []
        self.co_occurrence: Dict[str, float] = {}  # "a+b" -> count
        self._stop_flag = False

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        line = f"[REPLAY {ts}] {msg}"
        self.log_lines.append(line)
        if len(self.log_lines) > 200: self.log_lines = self.log_lines[-200:]
        print(line)

    def stop(self): self._stop_flag = True

    def _track_cooccurrence(self, prompt: str):
        """Log which concepts appeared together in a prompt."""
        words = [w for w in prompt.lower().split() if w in self.co_occurrence or True]
        for i, a in enumerate(words):
            for b in words[i+1:]:
                key = f"{min(a,b)}+{max(a,b)}"
                self.co_occurrence[key] = self.co_occurrence.get(key, 0) + 1.0

    def generate_replay_batch(
        self,
        decoder: ConceptDecoder,
        memory: ConceptMemory,
        batch_size: int = 8,
        use_combinations: bool = True,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Generate a batch of (image, prompt) pairs from existing concepts.
        Includes single concepts AND combinations for compositional learning.
        """
        concepts = memory.concepts
        if not concepts:
            return None, []

        prompts = []
        for _ in range(batch_size):
            if use_combinations and len(concepts) >= 2 and torch.rand(1).item() > 0.4:
                # Pick 2 concepts — compositional dreaming
                idxs = torch.randperm(len(concepts))[:2].tolist()
                prompt = f"{concepts[idxs[0]]} {concepts[idxs[1]]}"
            else:
                # Single concept
                idx = torch.randint(len(concepts), (1,)).item()
                prompt = concepts[idx]
            prompts.append(prompt)
            self._track_cooccurrence(prompt)

        # Generate images for all prompts
        decoder.eval()
        decoder_device = next(decoder.parameters()).device
        embs = []
        for prompt in prompts:
            emb = memory.query(prompt, sample=True)  # [1, 512]
            # Add small noise for variation
            noise = torch.randn_like(emb) * 0.05
            emb = F.normalize(emb + noise, dim=-1)
            embs.append(emb.view(1, CLIP_DIM))
        emb_batch = torch.cat(embs, dim=0).to(decoder_device)  # [B, 512]

        with torch.no_grad():
            generated = decoder(emb_batch)  # [B, 3, 128, 128]

        return generated.detach(), prompts

    def run(
        self,
        decoder: ConceptDecoder,
        memory: ConceptMemory,
        vgg,
        steps: int = 50,
        batch_size: int = 8,
        lr: float = 5e-5,          # much lower LR than normal training
        lambda_max: float = 0.02,  # tighter decay during replay
        progress_callback=None,
        apply_gravity: bool = True,
    ) -> ConceptDecoder:
        """
        Run the dream replay loop.
        LR is deliberately low — we are consolidating, not relearning.
        """
        self._stop_flag = False
        decoder = decoder.to(DEVICE)
        decoder.register_beta_hooks()

        # Replay optimizer — separate from main training optimizer
        optimizer = torch.optim.AdamW(decoder.parameters(), lr=lr, weight_decay=lambda_max)
        self._log(f"Dream replay: {steps} steps, lr={lr}, batch={batch_size}")
        self._log(f"Known concepts: {memory.concepts}")

        for step in range(steps):
            if self._stop_flag: break

            # 1. Generate a replay batch from memory
            replay_imgs, prompts = self.generate_replay_batch(
                decoder, memory, batch_size=batch_size
            )
            if replay_imgs is None:
                self._log("No concepts to replay yet.")
                break

            decoder.train()

            # 2. Re-generate with gradients (same embeddings, different forward pass)
            embs = []
            for prompt in prompts:
                emb = memory.query(prompt, sample=False)  # use centroid for training stability
                embs.append(emb.view(1, CLIP_DIM))
            emb_batch = torch.cat(embs, dim=0).to(DEVICE)

            regenerated = decoder(emb_batch)  # [B, 3, 128, 128]

            # 3. Loss: make regenerated match the replay target
            pixel_loss = F.l1_loss(regenerated, replay_imgs.to(DEVICE))

            # Perceptual loss on replay
            with torch.no_grad():
                real_feats = vgg(replay_imgs.to(DEVICE))
            gen_feats = vgg(regenerated)
            perc_loss = sum(F.l1_loss(g, r.detach())
                           for g, r in zip(gen_feats, real_feats)) / 3.0

            loss = pixel_loss + 0.05 * perc_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.5)
            optimizer.step()

            # 4. β update — during replay β should be LOW (known concepts)
            betas = decoder.compute_beta()
            decoder.update_beta(betas)
            gammas, beta_vals = decoder.get_gamma(self.alpha)

            # 5. Apply β-gated decay — crystallized layers get maximally shielded
            for group in optimizer.param_groups:
                avg_gamma = np.mean(list(gammas.values()))
                group['weight_decay'] = max(lambda_max * avg_gamma, 1e-7)

            if step % 10 == 0:
                avg_beta = np.mean(list(beta_vals.values()))
                avg_gamma = np.mean(list(gammas.values()))
                self._log(
                    f"Replay step {step+1}/{steps} | Loss: {loss.item():.4f} | "
                    f"β_avg={avg_beta:.3f} | Γ_avg={avg_gamma:.3f} | "
                    f"Prompts: {prompts[:3]}"
                )

            if progress_callback:
                progress_callback((step + 1) / steps)

        # 6. Apply semantic gravity — concepts that dreamed together drift together
        if apply_gravity and self.co_occurrence:
            memory.apply_semantic_gravity(self.co_occurrence, strength=0.015)
            self._log(f"Semantic gravity applied. Co-occurrences: {dict(list(self.co_occurrence.items())[:5])}")

        self._log("Dream replay complete.")
        return decoder


# ═════════════════════════════════════════════════════════════════════════════
# 5. SAVE / LOAD
# ═════════════════════════════════════════════════════════════════════════════

def save_brain(decoder: ConceptDecoder, memory: ConceptMemory, path: Path = SAVE_DIR,
               trainer: 'ClockfieldDreamTrainer' = None):
    path.mkdir(exist_ok=True)
    torch.save(decoder.state_dict(), path / 'decoder.pt')
    memory.save(path)
    if trainer is not None:
        torch.save(trainer.dynamics.state_dict(), path / 'dynamics.pt')
        trainer.graph.save(path)
    meta = {
        'concepts': memory.concepts,
        'image_counts': memory.image_counts,
        'saved_at': time.strftime("%Y-%m-%d %H:%M:%S"),
        'img_size': IMG_SIZE,
        'graph_nodes': len(trainer.graph.centroids) if trainer else 0,
    }
    with open(path / 'meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    return str(path)

def load_brain(path: Path = SAVE_DIR,
               trainer: 'ClockfieldDreamTrainer' = None):
    memory = ConceptMemory()
    decoder = ConceptDecoder()
    weights_path = path / 'decoder.pt'
    if weights_path.exists():
        try:
            decoder.load_state_dict(torch.load(weights_path, map_location='cpu'))
            print("[brain] Loaded decoder weights (exact match)")
        except RuntimeError as e:
            # Architecture changed — try partial load, else start fresh
            print(f"[brain] Architecture mismatch detected: {e}")
            try:
                saved = torch.load(weights_path, map_location='cpu')
                current = decoder.state_dict()
                compatible = {}
                skipped = []
                for k, v in saved.items():
                    if k in current and current[k].shape == v.shape:
                        compatible[k] = v
                    else:
                        skipped.append(k)
                if compatible:
                    decoder.load_state_dict(compatible, strict=False)
                    print(f"[brain] Partial load: {len(compatible)} params loaded, "
                          f"{len(skipped)} skipped (shape mismatch)")
                else:
                    print("[brain] No compatible weights found — starting fresh")
                    # Back up the old weights so they aren't lost
                    backup = path / 'decoder_old_arch.pt'
                    if not backup.exists():
                        import shutil
                        shutil.copy2(weights_path, backup)
                        print(f"[brain] Old weights backed up to {backup}")
            except Exception as e2:
                print(f"[brain] Could not load any weights: {e2} — starting fresh")
        memory.load(path)
        if trainer is not None:
            dyn_path = path / 'dynamics.pt'
            if dyn_path.exists():
                try:
                    trainer.dynamics.load_state_dict(
                        torch.load(dyn_path, map_location='cpu'))
                except RuntimeError:
                    print("[brain] Dynamics architecture changed — starting fresh")
            trainer.graph.load(path)
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
        mem_emb = memory.query(prompt, sample=(temperature > 0))  # [1, CLIP_DIM]
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
        'replay': DreamReplay(),
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
        gr = state['trainer'].graph
        dyn_b = float(state['trainer'].dynamics.beta_ema)
        dyn_g = float(np.exp(-3.0 * dyn_b))
        lines = [
            f"**Concepts:** {len(m.concepts)} | **Graph nodes:** {len(gr.centroids)}",
            f"**Known words:** {', '.join(m.concepts) if m.concepts else '—'}",
            f"**β_proj={b['proj']:.3f}  β_up4={b['up4']:.3f}  β_dyn={dyn_b:.3f}**",
            f"**Γ_proj={g['proj']:.3f}  Γ_up4={g['up4']:.3f}  Γ_dyn={dyn_g:.3f}** (0=shielded)",
            f"**Device:** {DEVICE}",
        ]
        return "\n".join(lines)

    def on_load():
        msg = ensure_clip()
        return msg, get_status()

    def on_save():
        path = save_brain(state['decoder'], state['memory'], trainer=state['trainer'])
        return f"✅ Saved to {path} | Concepts: {state['memory'].concepts} | Graph: {len(state['trainer'].graph.centroids)} nodes"

    def on_load_brain():
        decoder, memory = load_brain(trainer=state['trainer'])
        state['decoder'] = decoder.to(DEVICE)
        state['memory'] = memory
        return f"✅ Loaded brain. Concepts: {memory.concepts} | Graph: {len(state['trainer'].graph.centroids)} nodes", get_status()

    def on_reset():
        state['decoder'] = ConceptDecoder()
        state['memory'] = ConceptMemory()
        state['trainer'].graph = MemoryGraph()
        state['trainer'].dynamics = LatentDynamics()
        return "🧹 Brain reset.", get_status()

    def on_stop():
        state['trainer'].stop()
        state['replay'].stop()
        return "⏹ Stop requested."

    def on_replay(steps, batch_size, lr, apply_gravity, progress=gr.Progress()):
        ensure_clip()
        if not state['memory'].concepts:
            return "No concepts learned yet. Train first.", get_status()
        try:
            vgg = VGGPerceptual().to(DEVICE)
            vgg.eval()
            def prog_cb(p): progress(p, desc="Dreaming...")
            state['decoder'] = state['replay'].run(
                state['decoder'], state['memory'], vgg,
                steps=int(steps), batch_size=int(batch_size),
                lr=float(lr), apply_gravity=bool(apply_gravity),
                progress_callback=prog_cb,
            )
            log = "\n".join(state['replay'].log_lines[-30:])
            # Show co-occurrence map
            co = state['replay'].co_occurrence
            if co:
                top = sorted(co.items(), key=lambda x: -x[1])[:8]
                log += f"\n\nSemantic gravity map (top pairs):\n"
                log += "\n".join(f"  {k}: {v:.0f} co-dreams" for k,v in top)
            return log, get_status()
        except Exception as e:
            import traceback
            return f"Error: {e}\n{traceback.format_exc()}", get_status()


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
                    hf_ep    = gr.Slider(1, 1000, value=5,  step=1,   label="Epochs")
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
                    f_ep     = gr.Slider(1, 1000, value=5,  step=1,   label="Epochs")
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
                    u_ep     = gr.Slider(1, 1000, value=20, step=1,  label="Epochs (more = better for few images)")
                    u_bs     = gr.Slider(1, 32,  value=4,  step=1,  label="Batch Size")
                    u_lr     = gr.Number(value=2e-4,                label="Learning Rate")
                u_btn        = gr.Button("▶ TEACH THIS CONCEPT", variant="primary")
                u_log        = gr.Textbox(label="Log", lines=8, interactive=False)
                u_status     = gr.Markdown()

            # ── Memory Graph tab ──────────────────────────────────────────────
            with gr.Tab("🧠 Memory Graph"):
                gr.Markdown("""
**Unlabeled visual vocabulary.** The graph grows automatically as you train.

Each node is a visual cluster the model discovered — no labels required.
- **Stability** = Γ of that node (high = crystallized, resists drift)
- **Count** = how many embeddings strengthened this node
- **Neighbors** = nodes that frequently co-activate (semantic connections)
- Novel inputs far from all nodes birth new nodes automatically
                """)
                graph_refresh = gr.Button("↻ Refresh Graph")
                graph_out = gr.Textbox(label="Memory Graph State", lines=20, interactive=False)

                def show_graph():
                    gr = state['trainer'].graph
                    if not gr.centroids:
                        return "No nodes yet. Train first."
                    lines = [f"Total nodes: {len(gr.centroids)}\n"]
                    # Sort by count descending
                    order = sorted(range(len(gr.centroids)), key=lambda i: -gr.counts[i])
                    for rank, i in enumerate(order[:40]):
                        label    = gr.labels[i] or "?"
                        stab     = gr.stabilities[i]
                        count    = gr.counts[i]
                        age      = gr.ages[i]
                        nb_labels= [gr.labels[n] or f"#{n}" for n in list(gr.neighbors[i])[:4]]
                        bar      = "█" * int(stab * 10) + "░" * (10 - int(stab * 10))
                        lines.append(
                            f"#{rank+1:02d} [{bar}] Γ={stab:.2f} "
                            f"label={label:<20} count={count:>4} age={age:>4}  "
                            f"neighbors=[{', '.join(nb_labels)}]"
                        )
                    return "\n".join(lines)

                graph_refresh.click(fn=show_graph, outputs=[graph_out])

            # ── Dream Replay tab ──────────────────────────────────────────────
            with gr.Tab("🔁 Dream Replay"):
                gr.Markdown("""
**Self-training loop.** The model generates images from its own memory, then trains on them.

- Reinforces existing concepts without new data
- β is LOW during replay (known concepts) → Γ → 0 → maximum shielding
- **Semantic gravity**: concepts that dream together drift toward each other in embedding space
- Run after adding new concepts to consolidate the whole space
                """)
                with gr.Row():
                    rp_steps   = gr.Slider(10, 500, value=50,  step=10, label="Replay Steps")
                    rp_batch   = gr.Slider(2,  32,  value=8,   step=2,  label="Batch Size")
                    rp_lr      = gr.Number(value=5e-5, label="LR (keep low)")
                rp_gravity     = gr.Checkbox(value=True, label="Apply semantic gravity after replay")
                rp_btn         = gr.Button("🔁 RUN DREAM REPLAY", variant="primary")
                rp_log         = gr.Textbox(label="Replay Log", lines=14, interactive=False)
                rp_status      = gr.Markdown()

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
        rp_btn.click(
            fn=on_replay,
            inputs=[rp_steps, rp_batch, rp_lr, rp_gravity],
            outputs=[rp_log, rp_status],
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