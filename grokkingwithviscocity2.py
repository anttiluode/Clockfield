import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Ensure it runs on all systems without a display
import json
import os
from datetime import datetime

# ============================================================================
# 1. THE BETA-GRADIENT PROBE (Roughness-based Viscosity)
# ============================================================================
class BetaGradientAnalyzer:
    def __init__(self):
        pass

    def compute_texture_beta(self, activation_tensor):
        """
        Measures 'Spectral Roughness' (Viscosity) of internal representations.
        """
        # acts shape: [Batch, Features]
        acts = activation_tensor.detach().cpu().numpy()
        
        # Normalize to ensure we measure structure, not just weight magnitude
        acts = (acts - acts.mean()) / (acts.std() + 1e-6)
        
        # Measure high-frequency jumps between adjacent neurons
        diffs = np.diff(acts, axis=1)
        roughness = np.abs(diffs).mean()
        
        return roughness

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

        # Hook layers to compare the structural depth
        for name, module in model.named_modules():
            if "output_head" in name or "transformer.layers.0.linear" in name:
                hooks.append(module.register_forward_hook(get_hook(name)))

        with torch.no_grad():
            for inputs, _ in dataloader:
                model(inputs.to(device))
                break  # Single batch is sufficient for texture probe

        for h in hooks: h.remove()

        layers = list(activations.keys())
        if len(layers) < 2: return 0.0
        
        b_shallow = self.compute_texture_beta(torch.cat(activations[layers[0]], dim=0))
        b_deep = self.compute_texture_beta(torch.cat(activations[layers[-1]], dim=0))
        
        # The Gradient: How much 'roughness' is the network generating as depth increases?
        return b_deep - b_shallow

# ============================================================================
# 2. MODULAR ARITHMETIC DATASET
# ============================================================================
class ModularAdditionDataset(torch.utils.data.Dataset):
    def __init__(self, p=97, train=True):
        self.p = p
        self.data = [(a, b) for a in range(p) for b in range(p)]
        np.random.seed(42)
        np.random.shuffle(self.data)
        split = int(0.5 * len(self.data)) # Harder split (50%) forces longer grokking plateau
        self.data = self.data[:split] if train else self.data[split:]

    def __len__(self): return len(self.data)

    def __getitem__(self, idx):
        a, b = self.data[idx]
        return torch.tensor([a, b]), (a + b) % self.p

# ============================================================================
# 3. TRANSFORMER ARCHITECTURE
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
# 4. THE LONG-PLATEAU EXPERIMENT
# ============================================================================
def run_long_experiment(p=97, max_epochs=10000, measure_every=20, device='cuda'):
    print(f"--- STARTING LONG-HAUL GROKKING EXPERIMENT (P={p}) ---")
    
    train_loader = torch.utils.data.DataLoader(ModularAdditionDataset(p, train=True), batch_size=512, shuffle=True)
    test_loader = torch.utils.data.DataLoader(ModularAdditionDataset(p, train=False), batch_size=512, shuffle=False)
    
    # HIGHER Hidden size + LOWER Weight Decay = Dramatically longer plateau
    model = GrokkingTransformer(vocab_size=p, hidden_size=256).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1) 
    criterion = nn.CrossEntropyLoss()
    analyzer = BetaGradientAnalyzer()

    results = {"epochs": [], "train_acc": [], "test_acc": [], "beta_gradient": []}

    for epoch in range(max_epochs):
        model.train()
        correct, total = 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(); loss = criterion(model(inputs), targets); loss.backward(); optimizer.step()
            correct += (model(inputs).argmax(dim=-1) == targets).sum().item(); total += targets.size(0)
        
        train_acc = 100 * correct / total

        # Evaluation
        if epoch % measure_every == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    preds = model(inputs.to(device)).argmax(dim=-1)
                    correct += (preds == targets.to(device)).sum().item(); total += targets.size(0)
            
            test_acc = 100 * correct / total
            beta_grad = analyzer.measure_network_viscosity(model, test_loader, device)
            
            # Cast for JSON
            results["epochs"].append(int(epoch))
            results["train_acc"].append(float(train_acc))
            results["test_acc"].append(float(test_acc))
            results["beta_gradient"].append(float(beta_grad))
            
            print(f"Epoch {epoch:5d} | Train: {train_acc:6.1f}% | Test: {test_acc:6.1f}% | β: {beta_grad:.4f}")

            if test_acc > 99.5:
                print(f"!!! FULL GROK AT EPOCH {epoch} !!!")
                break

    # Save results
    os.makedirs("grokking_results", exist_ok=True)
    with open("grokking_results/long_experiment.json", "w") as f:
        json.dump(results, f, indent=2)

    # SMOKING GUN OVERLAY PLOT
    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Accuracy (%)')
    ax1.plot(results["epochs"], results["train_acc"], color='blue', alpha=0.3, label='Train Acc')
    ax1.plot(results["epochs"], results["test_acc"], color='green', linewidth=3, label='Test Acc (Grokking)')
    ax1.set_ylim(-5, 105)

    ax2 = ax1.twinx()
    ax2.set_ylabel('β-Gradient (Viscosity)', color='red')
    ax2.plot(results["epochs"], results["beta_gradient"], color='red', linewidth=2, label='Internal Roughness (β)')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title(f"Structural Lead-Time: β-Gradient vs. Grokking Phase Transition")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc='upper left')

    plt.savefig("grokking_results/THE_SMOKING_GUN_LONG.png", dpi=300)
    print("✅ Experiment complete. Check grokking_results/THE_SMOKING_GUN_LONG.png")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    run_long_experiment(device=device)