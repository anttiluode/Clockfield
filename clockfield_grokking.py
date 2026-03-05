"""
Clockfield Grokking Experiment: β-Adaptive Weight Decay
========================================================
Antti Luode (PerceptionLab, Finland) | March 2026
Claude (Anthropic) — Collaborative formalization

HYPOTHESIS:
    The Clockfield metric predicts that weight decay should be
    ADAPTIVE, not uniform. Neurons that have crystallized into
    stable circuits (high β-roughness) should experience less
    decay — they're in a temporal well, their weights are correct.
    Neurons still memorizing (low β) should experience full decay.

    Uniform weight decay λ·||W||² is the ℏₙ → 0 degenerate limit
    of this mechanism — it applies the same erosion to all weights
    regardless of whether they've found structure.

EXPERIMENT:
    Three conditions, identical architecture and data:
    
    1. BASELINE:   AdamW with uniform weight_decay=0.1 (standard)
    2. CLOCKFIELD:  Adam (no WD) + per-parameter-group β-adaptive decay
    3. NO_DECAY:   Adam with weight_decay=0 (ablation control)
    
    All three use the same random seed, same data split, same
    transformer architecture. The ONLY difference is how weight
    decay is applied.

MEASUREMENT:
    - Epochs to grok (test acc > 95%)
    - β-gradient trajectory (structural lead-time)
    - Per-layer weight norms over training
    - The melt-crystallize phase transition timing

CLOCKFIELD MECHANISM:
    Every `update_every` epochs, compute β-roughness per parameter group.
    Map β to a per-group decay rate:
    
        decay_i = λ_max · (1 - β_normalized_i)
    
    High β (crystallized) → low decay → weights preserved.
    Low β (memorizing)   → high decay → weights eroded.
    
    This is the Clockfield: Γ_i = exp(-α·β_i) modulating the
    effective update rate. In the weight decay formulation:
    
        W_i ← W_i · (1 - lr · decay_i)
    
    When β_i is high, decay_i → 0, so W_i barely changes (frozen time).
    When β_i is low, decay_i → λ_max, so W_i gets full erosion (fast time).

Usage:
    python clockfield_grokking.py
    
    Outputs:
        clockfield_results/comparison.json
        clockfield_results/clockfield_vs_baseline.png
        clockfield_results/beta_trajectories.png
        clockfield_results/phase_analysis.png
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os
import copy
from datetime import datetime

# ============================================================================
# 1. THE BETA-GRADIENT PROBE (unchanged from original)
# ============================================================================

class BetaGradientAnalyzer:
    """Measures spectral roughness gradient between layers."""
    
    def compute_texture_beta(self, activation_tensor):
        acts = activation_tensor.detach().cpu().numpy()
        acts = (acts - acts.mean()) / (acts.std() + 1e-6)
        diffs = np.diff(acts, axis=1)
        return np.abs(diffs).mean()

    def measure_network_viscosity(self, model, dataloader, device='cuda'):
        model.eval()
        activations = {}
        hooks = []

        def get_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple): output = output[0]
                if name not in activations: activations[name] = []
                activations[name].append(output.detach())
            return hook

        for name, module in model.named_modules():
            if "output_head" in name or "transformer.layers.0.linear" in name:
                hooks.append(module.register_forward_hook(get_hook(name)))

        with torch.no_grad():
            for inputs, _ in dataloader:
                model(inputs.to(device))
                break

        for h in hooks: h.remove()

        layers = list(activations.keys())
        if len(layers) < 2: return 0.0
        
        b_shallow = self.compute_texture_beta(
            torch.cat(activations[layers[0]], dim=0))
        b_deep = self.compute_texture_beta(
            torch.cat(activations[layers[-1]], dim=0))
        
        return b_deep - b_shallow
    
    def measure_per_layer_beta(self, model, dataloader, device='cuda'):
        """
        Return per-layer roughness for adaptive decay computation.
        Returns dict: {layer_name: roughness_value}
        """
        model.eval()
        activations = {}
        hooks = []

        def get_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple): output = output[0]
                if name not in activations: activations[name] = []
                activations[name].append(output.detach())
            return hook

        # Hook ALL linear layers and the embedding
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                hooks.append(module.register_forward_hook(get_hook(name)))

        with torch.no_grad():
            for inputs, _ in dataloader:
                model(inputs.to(device))
                break

        for h in hooks: h.remove()

        result = {}
        for name, act_list in activations.items():
            acts = torch.cat(act_list, dim=0)
            if acts.dim() == 3:
                # [batch, seq, hidden] -> flatten seq into batch
                acts = acts.reshape(-1, acts.shape[-1])
            if acts.shape[1] < 2:
                result[name] = 0.0
                continue
            result[name] = float(self.compute_texture_beta(acts))
        
        return result


# ============================================================================
# 2. MODULAR ARITHMETIC DATASET (unchanged)
# ============================================================================

class ModularAdditionDataset(torch.utils.data.Dataset):
    def __init__(self, p=97, train=True):
        self.p = p
        self.data = [(a, b) for a in range(p) for b in range(p)]
        np.random.seed(42)
        np.random.shuffle(self.data)
        split = int(0.5 * len(self.data))
        self.data = self.data[:split] if train else self.data[split:]

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        a, b = self.data[idx]
        return torch.tensor([a, b]), (a + b) % self.p


# ============================================================================
# 3. TRANSFORMER ARCHITECTURE (unchanged)
# ============================================================================

class GrokkingTransformer(nn.Module):
    def __init__(self, vocab_size=97, hidden_size=256, num_layers=1, nhead=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size*4,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return self.output_head(x[:, -1, :])


# ============================================================================
# 4. CLOCKFIELD WEIGHT DECAY
# ============================================================================

class ClockfieldDecay:
    """
    Adaptive weight decay based on the β-Sieve probe.
    
    The Clockfield metric Γ_i = exp(-α·β_i) maps each parameter 
    group's crystallization state to a local time dilation factor.
    
    In the weight decay formulation:
        decay_i = λ_max · exp(-α · β_i)
    
    High β → low decay (frozen, crystallized)
    Low β  → high decay (searching, memorizing)
    
    This is NOT just a per-layer learning rate. The decay operates
    on the weight magnitudes (regularization), not on the gradient
    step size. It tells the optimizer: "leave the structured weights 
    alone, erode the unstructured ones."
    """
    
    def __init__(self, model, lambda_max=0.1, alpha=3.0, min_decay=0.001):
        """
        Args:
            model: The neural network
            lambda_max: Maximum weight decay (applied to lowest-β parameters)
            alpha: Clockfield coupling constant (controls how steeply β 
                   modulates decay; higher = more aggressive protection
                   of crystallized weights)
            min_decay: Floor on decay rate (prevents complete freezing)
        """
        self.model = model
        self.lambda_max = lambda_max
        self.alpha = alpha
        self.min_decay = min_decay
        
        # Map parameter names to their parent module names for β lookup
        self.param_to_module = {}
        for mod_name, module in model.named_modules():
            for param_name, _ in module.named_parameters(recurse=False):
                full_name = f"{mod_name}.{param_name}" if mod_name else param_name
                self.param_to_module[full_name] = mod_name
        
        # Current per-module decay rates
        self.decay_rates = {}
        
    def update_decay_rates(self, per_layer_beta):
        """
        Given per-layer β measurements, compute Clockfield decay rates.
        
        Args:
            per_layer_beta: dict {module_name: roughness_value}
        """
        if not per_layer_beta:
            return
            
        # Normalize β to [0, 1] range for stable exp() behavior
        beta_values = list(per_layer_beta.values())
        beta_min = min(beta_values)
        beta_max = max(beta_values)
        beta_range = beta_max - beta_min + 1e-8
        
        for mod_name, beta_raw in per_layer_beta.items():
            beta_norm = (beta_raw - beta_min) / beta_range
            
            # Clockfield: Γ = exp(-α·β), decay = λ_max · Γ
            gamma = np.exp(-self.alpha * beta_norm)
            decay = self.lambda_max * gamma
            decay = max(decay, self.min_decay)
            
            self.decay_rates[mod_name] = decay
    
    def apply_decay(self, lr):
        """
        Manually apply weight decay to all parameters based on 
        their module's Clockfield decay rate.
        
        Call this AFTER optimizer.step() each training step.
        
        The standard weight decay formula is:
            W ← W · (1 - lr · λ)
        
        We replace λ with the per-module Clockfield rate.
        """
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # Find the module this parameter belongs to
                mod_name = self.param_to_module.get(name, '')
                
                # Look up decay rate (try exact match, then parent modules)
                decay = self.lambda_max  # default: full decay
                
                # Try exact module match
                if mod_name in self.decay_rates:
                    decay = self.decay_rates[mod_name]
                else:
                    # Try matching by prefix (parameter might be in a child 
                    # of a hooked module)
                    for hooked_name, rate in self.decay_rates.items():
                        if mod_name.startswith(hooked_name) or hooked_name.startswith(mod_name):
                            decay = rate
                            break
                
                # Apply decay: W ← W · (1 - lr · decay)
                param.mul_(1.0 - lr * decay)
    
    def get_summary(self):
        """Return current decay rates for logging."""
        return dict(self.decay_rates)


# ============================================================================
# 5. TRAINING LOOP (parameterized by condition)
# ============================================================================

def train_one_condition(condition_name, p=97, max_epochs=2000, 
                         measure_every=10, seed=42, device='cuda',
                         lambda_max=0.1, clockfield_alpha=3.0,
                         clockfield_update_every=10):
    """
    Train a single condition and return results.
    
    Args:
        condition_name: 'baseline' | 'clockfield' | 'no_decay'
        max_epochs: Maximum training epochs
        measure_every: How often to evaluate (epochs)
        seed: Random seed for reproducibility
        lambda_max: Weight decay strength (baseline and clockfield max)
        clockfield_alpha: Clockfield coupling constant
        clockfield_update_every: How often to update β and decay rates
    """
    print(f"\n{'='*64}")
    print(f"  CONDITION: {condition_name.upper()}")
    print(f"{'='*64}")
    
    # Deterministic initialization
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    train_loader = torch.utils.data.DataLoader(
        ModularAdditionDataset(p, train=True), batch_size=512, shuffle=True,
        generator=torch.Generator().manual_seed(seed)
    )
    test_loader = torch.utils.data.DataLoader(
        ModularAdditionDataset(p, train=False), batch_size=512, shuffle=False
    )
    
    model = GrokkingTransformer(vocab_size=p, hidden_size=256).to(device)
    criterion = nn.CrossEntropyLoss()
    analyzer = BetaGradientAnalyzer()
    
    # Condition-specific optimizer setup
    lr = 1e-3
    
    if condition_name == 'baseline':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=lambda_max)
        clockfield = None
        
    elif condition_name == 'clockfield':
        # Adam WITHOUT weight decay — we apply it manually via Clockfield
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
        clockfield = ClockfieldDecay(model, lambda_max=lambda_max, 
                                      alpha=clockfield_alpha)
        
    elif condition_name == 'no_decay':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
        clockfield = None
    
    else:
        raise ValueError(f"Unknown condition: {condition_name}")
    
    results = {
        "condition": condition_name,
        "epochs": [],
        "train_acc": [],
        "test_acc": [],
        "beta_gradient": [],
        "decay_rates": [],           # per-epoch Clockfield decay rates
        "weight_norms": [],           # per-epoch weight norm by layer
        "grok_epoch": None,           # epoch where test > 95%
        "params": {
            "lr": lr,
            "lambda_max": lambda_max,
            "clockfield_alpha": clockfield_alpha if condition_name == 'clockfield' else None,
            "seed": seed,
            "p": p,
        }
    }
    
    grokked = False
    
    for epoch in range(max_epochs):
        model.train()
        correct, total = 0, 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            
            # Clockfield: apply adaptive decay AFTER optimizer step
            if clockfield is not None:
                clockfield.apply_decay(lr)
            
            correct += (output.argmax(dim=-1) == targets).sum().item()
            total += targets.size(0)
        
        train_acc = 100.0 * correct / total
        
        # Evaluation and β measurement
        if epoch % measure_every == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    preds = model(inputs.to(device)).argmax(dim=-1)
                    correct += (preds == targets.to(device)).sum().item()
                    total += targets.size(0)
            
            test_acc = 100.0 * correct / total
            beta_grad = analyzer.measure_network_viscosity(model, test_loader, device)
            
            # Update Clockfield decay rates
            if clockfield is not None and epoch % clockfield_update_every == 0:
                per_layer_beta = analyzer.measure_per_layer_beta(
                    model, test_loader, device)
                clockfield.update_decay_rates(per_layer_beta)
            
            # Record weight norms per named parameter group
            weight_norms = {}
            for name, param in model.named_parameters():
                weight_norms[name] = float(param.data.norm().item())
            
            # Record
            results["epochs"].append(int(epoch))
            results["train_acc"].append(float(train_acc))
            results["test_acc"].append(float(test_acc))
            results["beta_gradient"].append(float(beta_grad))
            results["weight_norms"].append(weight_norms)
            
            if clockfield is not None:
                results["decay_rates"].append(clockfield.get_summary())
            
            # Check for grokking
            if test_acc > 95.0 and not grokked:
                grokked = True
                results["grok_epoch"] = int(epoch)
                print(f"  >>> GROKKED at epoch {epoch}! <<<")
            
            # Print progress
            decay_info = ""
            if clockfield is not None and clockfield.decay_rates:
                rates = list(clockfield.decay_rates.values())
                decay_info = f" | Decay: {min(rates):.4f}-{max(rates):.4f}"
            
            print(f"  E{epoch:5d} | Tr:{train_acc:5.1f}% | Te:{test_acc:5.1f}% "
                  f"| β:{beta_grad:.4f}{decay_info}")
            
            if test_acc > 99.5:
                print(f"  Full generalization at epoch {epoch}")
                break
    
    if not grokked:
        print(f"  Did not grok within {max_epochs} epochs")
    
    return results


# ============================================================================
# 6. HEAD-TO-HEAD COMPARISON
# ============================================================================

def run_comparison(p=97, max_epochs=2000, measure_every=10, seed=42, 
                   device='cuda'):
    """
    Run all three conditions and produce comparison plots.
    """
    print("=" * 64)
    print("  CLOCKFIELD GROKKING EXPERIMENT")
    print("  β-Adaptive Weight Decay vs. Uniform Weight Decay")
    print("=" * 64)
    print(f"  Device: {device}")
    print(f"  Modular arithmetic: mod {p}")
    print(f"  Max epochs: {max_epochs}")
    print(f"  Seed: {seed}")
    print()
    
    os.makedirs("clockfield_results", exist_ok=True)
    
    all_results = {}
    
    # Run all three conditions with the same seed
    for condition in ['baseline', 'clockfield', 'no_decay']:
        results = train_one_condition(
            condition_name=condition,
            p=p,
            max_epochs=max_epochs,
            measure_every=measure_every,
            seed=seed,
            device=device,
            lambda_max=0.1,
            clockfield_alpha=3.0,
            clockfield_update_every=10,
        )
        all_results[condition] = results
    
    # Save raw results (strip weight_norms for JSON size)
    save_results = {}
    for cond, res in all_results.items():
        r = dict(res)
        # Simplify weight norms to just total norm
        r["total_weight_norm"] = [
            sum(wn.values()) for wn in r["weight_norms"]
        ]
        del r["weight_norms"]
        save_results[cond] = r
    
    with open("clockfield_results/comparison.json", "w") as f:
        json.dump(save_results, f, indent=2)
    
    # ── PLOT 1: Head-to-head grokking curves ──
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    fig.suptitle("Clockfield Grokking Experiment: Adaptive vs. Uniform Weight Decay",
                 fontsize=14, fontweight='bold')
    
    colors = {'baseline': '#2196F3', 'clockfield': '#FF5722', 'no_decay': '#9E9E9E'}
    labels = {'baseline': 'Baseline (uniform WD=0.1)', 
              'clockfield': 'Clockfield (β-adaptive WD)',
              'no_decay': 'No weight decay (ablation)'}
    
    # Top: Test accuracy
    for cond in ['baseline', 'clockfield', 'no_decay']:
        r = all_results[cond]
        ax1.plot(r['epochs'], r['test_acc'], color=colors[cond], 
                linewidth=2.5 if cond != 'no_decay' else 1.5,
                alpha=1.0 if cond != 'no_decay' else 0.5,
                label=labels[cond])
        if r['grok_epoch'] is not None:
            ax1.axvline(x=r['grok_epoch'], color=colors[cond], 
                       linestyle='--', alpha=0.5, linewidth=1)
            ax1.annotate(f"Grok: {r['grok_epoch']}", 
                        xy=(r['grok_epoch'], 95),
                        fontsize=8, color=colors[cond],
                        ha='center', va='bottom')
    
    ax1.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax1.set_ylim(-5, 105)
    ax1.legend(fontsize=10, loc='upper left')
    ax1.grid(True, alpha=0.2)
    ax1.set_title('Generalization (Test Accuracy)', fontsize=11)
    
    # Bottom: β-gradient trajectories
    for cond in ['baseline', 'clockfield', 'no_decay']:
        r = all_results[cond]
        ax2.plot(r['epochs'], r['beta_gradient'], color=colors[cond],
                linewidth=2.0, alpha=0.8, label=labels[cond])
    
    ax2.set_xlabel('Epochs', fontsize=11)
    ax2.set_ylabel('β-Gradient (Roughness)', fontsize=11)
    ax2.legend(fontsize=10, loc='upper left')
    ax2.grid(True, alpha=0.2)
    ax2.set_title('Internal Structure (β-Sieve Probe)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig("clockfield_results/clockfield_vs_baseline.png", dpi=200, 
                bbox_inches='tight')
    plt.close()
    
    # ── PLOT 2: Phase transition analysis ──
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Phase Transition Analysis", fontsize=13, fontweight='bold')
    
    for ax_idx, cond in enumerate(['baseline', 'clockfield', 'no_decay']):
        ax = axes[ax_idx]
        r = all_results[cond]
        
        # Dual axis: test acc + beta
        ax.plot(r['epochs'], r['test_acc'], color='green', linewidth=2, 
                label='Test Acc')
        ax.set_ylabel('Test Accuracy (%)', color='green')
        ax.set_ylim(-5, 105)
        ax.tick_params(axis='y', labelcolor='green')
        
        ax2 = ax.twinx()
        ax2.plot(r['epochs'], r['beta_gradient'], color='red', linewidth=1.5,
                label='β-gradient')
        ax2.set_ylabel('β-gradient', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax.set_xlabel('Epochs')
        ax.set_title(f'{cond.upper()}\n'
                    f'Grok: {"epoch " + str(r["grok_epoch"]) if r["grok_epoch"] else "never"}',
                    fontsize=10)
        ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("clockfield_results/phase_analysis.png", dpi=200, 
                bbox_inches='tight')
    plt.close()
    
    # ── PLOT 3: Clockfield decay rate evolution (if applicable) ──
    
    if all_results['clockfield']['decay_rates']:
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Collect per-module decay rates over time
        decay_history = all_results['clockfield']['decay_rates']
        epochs_with_decay = [all_results['clockfield']['epochs'][i] 
                            for i in range(len(decay_history))]
        
        # Get all module names
        all_modules = set()
        for dr in decay_history:
            if dr:
                all_modules.update(dr.keys())
        
        cmap = plt.cm.tab10
        for idx, mod_name in enumerate(sorted(all_modules)):
            rates = []
            for dr in decay_history:
                if dr and mod_name in dr:
                    rates.append(dr[mod_name])
                else:
                    rates.append(None)
            
            # Filter out Nones
            valid_epochs = [e for e, r in zip(epochs_with_decay, rates) if r is not None]
            valid_rates = [r for r in rates if r is not None]
            
            if valid_rates:
                # Shorten module name for legend
                short_name = mod_name.replace('transformer.layers.0.', 'T.')
                short_name = short_name.replace('self_attn.', 'attn.')
                ax.plot(valid_epochs, valid_rates, color=cmap(idx % 10),
                       linewidth=1.5, alpha=0.8, label=short_name[:30])
        
        ax.set_xlabel('Epochs', fontsize=11)
        ax.set_ylabel('Clockfield Decay Rate', fontsize=11)
        ax.set_title('Per-Module Adaptive Decay Rates (Clockfield Condition)',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=7, ncol=2, loc='upper right')
        ax.grid(True, alpha=0.2)
        ax.axhline(y=0.1, color='gray', linestyle='--', alpha=0.5, 
                   label='Uniform baseline (0.1)')
        
        plt.tight_layout()
        plt.savefig("clockfield_results/decay_evolution.png", dpi=200,
                    bbox_inches='tight')
        plt.close()
    
    # ── Summary ──
    
    print("\n" + "=" * 64)
    print("  RESULTS SUMMARY")
    print("=" * 64)
    
    for cond in ['baseline', 'clockfield', 'no_decay']:
        r = all_results[cond]
        grok_str = f"epoch {r['grok_epoch']}" if r['grok_epoch'] else "DID NOT GROK"
        final_test = r['test_acc'][-1] if r['test_acc'] else 0.0
        max_beta = max(r['beta_gradient']) if r['beta_gradient'] else 0.0
        print(f"\n  {cond.upper():15s} | Grok: {grok_str:15s} | "
              f"Final test: {final_test:5.1f}% | Max β: {max_beta:.4f}")
    
    # The key comparison
    baseline_grok = all_results['baseline']['grok_epoch']
    clock_grok = all_results['clockfield']['grok_epoch']
    
    if baseline_grok and clock_grok:
        speedup = baseline_grok / clock_grok
        print(f"\n  Clockfield speedup: {speedup:.2f}x "
              f"({baseline_grok} → {clock_grok} epochs)")
    elif clock_grok and not baseline_grok:
        print(f"\n  Clockfield grokked (epoch {clock_grok}), baseline did not!")
    elif baseline_grok and not clock_grok:
        print(f"\n  Baseline grokked (epoch {baseline_grok}), Clockfield did not.")
        print(f"  (Clockfield alpha may need tuning)")
    else:
        print(f"\n  Neither condition grokked within the epoch budget.")
    
    print(f"\n  Saved to clockfield_results/")
    print(f"  - comparison.json")
    print(f"  - clockfield_vs_baseline.png")
    print(f"  - phase_analysis.png")
    if all_results['clockfield']['decay_rates']:
        print(f"  - decay_evolution.png")
    
    return all_results


# ============================================================================
# 7. ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Main experiment
    results = run_comparison(
        p=97,
        max_epochs=2000,     # Your original ran to ~720; 2000 gives margin
        measure_every=10,    # Measure every 10 epochs for good resolution
        seed=42,
        device=device,
    )
