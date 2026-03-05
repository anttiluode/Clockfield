import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') # Safe for all environments
import json
import os

# ============================================================================
# 1. BETA-GRADIENT ANALYZER (Roughness version)
# ============================================================================
class BetaGradientAnalyzer:
    def compute_texture_beta(self, activation_tensor):
        acts = activation_tensor.detach().cpu().numpy()
        acts = (acts - acts.mean()) / (acts.std() + 1e-6) # Normalize structure
        diffs = np.diff(acts, axis=1) # Local roughness
        return np.abs(diffs).mean()

    def measure_network_viscosity(self, model, dataloader, device='cuda'):
        model.eval()
        activations = {}
        def get_hook(name):
            def hook(module, input, output):
                if isinstance(output, tuple): output = output[0]
                if name not in activations: activations[name] = []
                activations[name].append(output.detach())
            return hook

        hooks = []
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
        
        b_shallow = self.compute_texture_beta(torch.cat(activations[layers[0]], dim=0))
        b_deep = self.compute_texture_beta(torch.cat(activations[layers[-1]], dim=0))
        return b_deep - b_shallow

# ============================================================================
# 2. SCRAMBLED DATASET (Pure Memorization, No Logic)
# ============================================================================
class ScrambledModularDataset(torch.utils.data.Dataset):
    def __init__(self, p=97, train=True):
        self.p = p
        all_pairs = [(a, b) for a in range(p) for b in range(p)]
        np.random.seed(42)
        
        # KEY DIFFERENCE: Every pair is mapped to a RANDOM result, breaking addition
        self.scrambled_map = {pair: np.random.randint(0, p) for pair in all_pairs}
        
        np.random.shuffle(all_pairs)
        split = int(0.5 * len(all_pairs)) # 50/50 split
        self.data = all_pairs[:split] if train else all_pairs[split:]

    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        pair = self.data[idx]
        return torch.tensor(pair), self.scrambled_map[pair]

# ============================================================================
# 3. TRANSFORMER ARCHITECTURE
# ============================================================================
class GrokkingTransformer(nn.Module):
    def __init__(self, vocab_size=97, hidden_size=256):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=4, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=1)
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.transformer(self.embed(x))
        return self.output_head(x[:, -1, :])

# ============================================================================
# 4. TRAINING LOOP
# ============================================================================
def run_control(p=97, epochs=1000, device='cuda'):
    print(f"--- STARTING CONTROL: SCRAMBLED LABELS ---")
    train_loader = torch.utils.data.DataLoader(ScrambledModularDataset(p, train=True), batch_size=512, shuffle=True)
    test_loader = torch.utils.data.DataLoader(ScrambledModularDataset(p, train=False), batch_size=512, shuffle=False)
    
    model = GrokkingTransformer(vocab_size=p).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    analyzer = BetaGradientAnalyzer()
    criterion = nn.CrossEntropyLoss()

    results = {"epochs": [], "train_acc": [], "test_acc": [], "beta_gradient": []}

    for epoch in range(epochs):
        model.train()
        correct, total = 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            correct += (outputs.argmax(dim=-1) == targets).sum().item()
            total += targets.size(0)
        
        train_acc = 100 * correct / total

        if epoch % 20 == 0:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    preds = model(inputs.to(device)).argmax(dim=-1)
                    correct += (preds == targets.to(device)).sum().item()
                    total += targets.size(0)
            
            test_acc = 100 * correct / total
            beta_grad = analyzer.measure_network_viscosity(model, test_loader, device)
            
            results["epochs"].append(int(epoch))
            results["train_acc"].append(float(train_acc))
            results["test_acc"].append(float(test_acc))
            results["beta_gradient"].append(float(beta_grad))
            
            print(f"Epoch {epoch:4d} | Train: {train_acc:6.1f}% (Memorizing) | β: {beta_grad:.4f}")

    # Save and Plot
    os.makedirs("grokking_results", exist_ok=True)
    with open("grokking_results/control_results.json", "w") as f:
        json.dump(results, f, indent=2)

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(results["epochs"], results["train_acc"], color='blue', linestyle='--', label='Train Acc (Memorization)')
    ax1.plot(results["epochs"], results["test_acc"], color='green', label='Test Acc (Random Noise)')
    ax1.set_ylabel("Accuracy %")
    ax1.set_ylim(-5, 105)

    ax2 = ax1.twinx()
    ax2.plot(results["epochs"], results["beta_gradient"], color='red', linewidth=2, label='β-Gradient (Viscosity)')
    ax2.set_ylabel("β-Gradient", color='red')
    
    plt.title("Control Experiment: Viscosity in the Absence of Logic")
    plt.savefig("grokking_results/CONTROL_PLOT.png")
    print("✅ Control complete. Check grokking_results/CONTROL_PLOT.png")

if __name__ == '__main__':
    run_control(device='cuda' if torch.cuda.is_available() else 'cpu')