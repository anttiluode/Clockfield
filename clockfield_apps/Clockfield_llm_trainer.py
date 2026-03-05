"""
Clockfield GPT-2 Medium (Conversational)
========================================
Full training script with Causal Attention Sharpness and Temporal Roughness.
Trains a 355M parameter multi-scale relativistic brain on the SamSum dataset.
Fits safely on a 12GB RTX 3060.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.pytorch_utils import Conv1D
from transformers import TrainerCallback
from datasets import load_dataset
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# 1. CLOCKFIELD GPT-2 ATTENTION (With Causal Sharpness Probe)
# ============================================================================

class ClockfieldGPT2Attention(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None, alpha=3.0):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        self.alpha = alpha
        self.register_buffer('beta_ema', torch.zeros(self.num_heads))

    def compute_head_beta(self, q, k):
        """
        Calculates Beta (crystallization) based on strictly causal ATTENTION SHARPNESS.
        """
        with torch.no_grad():
            # Calculate raw attention scores
            attn_weights = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.head_dim)
            
            # --- CAUSALITY PATCH ---
            # We must mask out the future before softmax, otherwise early tokens 
            # will look artificially flat/noisy because they are peering into the void.
            seq_len = q.size(-2)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device)).view(1, 1, seq_len, seq_len)
            attn_weights = attn_weights.masked_fill(causal_mask == 0, float('-inf'))
            # -----------------------

            # Convert to probabilities
            attn_probs = torch.nn.functional.softmax(attn_weights, dim=-1)
            
            # SHARPNESS: The maximum probability assigned to any single valid token.
            sharpness = attn_probs.max(dim=-1)[0] # Shape: [bsz, num_heads, seq_len]
            
            # Average the sharpness across the batch and sequence to get the head's Beta
            beta = sharpness.mean(dim=(0, 2)) # Shape: [num_heads]
            
        return beta

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[torch.Tensor] = None,
        past_key_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ):
        if layer_past is None and past_key_values is not None:
            layer_past = past_key_values

        bsz, q_len, _ = hidden_states.size()

        # 1. Update the Metric ONLY during training using Causal Attention Sharpness
        if self.training:
            with torch.no_grad():
                query, key, _ = self.c_attn(hidden_states).split(self.split_size, dim=2)
                q = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                k = key.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                current_beta = self.compute_head_beta(q, k)
                self.beta_ema = 0.9 * self.beta_ema + 0.1 * current_beta

        # 2. Compute Metric Time Dilation (Gamma)
        beta_min, beta_max = self.beta_ema.min(), self.beta_ema.max()
        beta_norm = (self.beta_ema - beta_min) / (beta_max - beta_min + 1e-8)
        gamma = torch.exp(-self.alpha * beta_norm)

        # 3. Calculate sequence lengths robustly
        kv_seq_len = q_len
        if layer_past is not None:
            if hasattr(layer_past, "get_seq_length"):
                kv_seq_len += layer_past.get_seq_length(self.layer_idx)
            else:
                curr = layer_past
                while isinstance(curr, (tuple, list)):
                    curr = curr[0]
                if hasattr(curr, "shape"):
                    kv_seq_len += curr.shape[-2]

        # 4. Generate Clockfield Proper-Time Mask
        min_window = min(16, kv_seq_len)
        window_sizes = (min_window + (kv_seq_len - min_window) * (1.0 - gamma)).view(1, self.num_heads, 1, 1)

        seq_ids_q = torch.arange(q_len, device=hidden_states.device)
        seq_ids_k = torch.arange(kv_seq_len, device=hidden_states.device)
        if q_len == 1 and kv_seq_len > 1:
            seq_ids_q = torch.tensor([kv_seq_len - 1], device=hidden_states.device)

        distance = seq_ids_q.view(1, 1, q_len, 1) - seq_ids_k.view(1, 1, 1, kv_seq_len)
        clockfield_mask_bool = (distance >= 0) & (distance <= window_sizes)

        mask_value = torch.finfo(hidden_states.dtype).min
        clockfield_additive = torch.zeros((1, self.num_heads, q_len, kv_seq_len), device=hidden_states.device, dtype=hidden_states.dtype)
        clockfield_additive = clockfield_additive.masked_fill(~clockfield_mask_bool, mask_value)

        # 5. Inject Metric into HF mask
        if attention_mask is not None:
            combined_mask = attention_mask + clockfield_additive
        else:
            combined_mask = clockfield_additive

        # 6. Pass back to HF's optimized GPT2 Attention
        return super().forward(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=combined_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs
        )

def replace_gpt2_attention_with_clockfield(module, config, alpha=3.0):
    replaced_count = 0
    for name, child in module.named_children():
        if isinstance(child, GPT2Attention):
            is_cross = getattr(child, "is_cross_attention", False)
            layer_idx = getattr(child, "layer_idx", None)
            clockfield_attn = ClockfieldGPT2Attention(config, is_cross_attention=is_cross, layer_idx=layer_idx, alpha=alpha)
            clockfield_attn.load_state_dict(child.state_dict(), strict=False)
            setattr(module, name, clockfield_attn)
            replaced_count += 1
        else:
            replaced_count += replace_gpt2_attention_with_clockfield(child, config, alpha)
    return replaced_count

# ============================================================================
# 2. CLOCKFIELD ADAPTIVE WEIGHT DECAY CALLBACK (With Temporal Roughness)
# ============================================================================

class ClockfieldAdaptiveDecayCallback(TrainerCallback):
    def __init__(self, lambda_max=0.1, alpha=3.0, update_every_steps=10):
        self.lambda_max = lambda_max
        self.alpha = alpha
        self.update_every_steps = update_every_steps
        self.activations = {}
        self.hooks = []
        self.step_count = 0

    def _get_hook(self, name):
        def hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.activations[name] = output.detach()
        return hook

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        print("\n[Clockfield] Registering Temporal β-probes on GPT-2 dense layers...")
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, Conv1D)) and ('c_fc' in name or 'c_proj' in name or 'c_attn' in name):
                self.hooks.append(module.register_forward_hook(self._get_hook(name)))

    def on_step_end(self, args, state, control, model=None, optimizer=None, **kwargs):
        self.step_count += 1
        
        if self.step_count % self.update_every_steps == 0 and optimizer:
            self._update_optimizer_decay(optimizer)
            self.activations.clear()

        # QM-KG Metric Logging
        if self.step_count % 10 == 0 and model is not None:
            all_betas = []
            for mod in model.modules():
                if hasattr(mod, 'beta_ema'):
                    all_betas.append(mod.beta_ema)
            
            if all_betas:
                stacked_betas = torch.cat(all_betas)
                beta_var = torch.var(stacked_betas)
                gamma_mean = torch.mean(torch.exp(-self.alpha * stacked_betas))
                print(f"[QM-KG Metric] Step {self.step_count} | β-variance: {beta_var.item():.4f} | Γ-mean: {gamma_mean.item():.4f}")

    def _update_optimizer_decay(self, optimizer):
        if not self.activations: return
        layer_betas = {}
        
        for name, acts in self.activations.items():
            # Temporal Roughness Patch: Measure how violently thoughts jump across sequence words (dim=1)
            if acts.dim() == 3:
                acts_norm = (acts - acts.mean(dim=1, keepdim=True)) / (acts.std(dim=1, keepdim=True) + 1e-6)
                diffs = torch.abs(torch.diff(acts_norm, dim=1)) # dim=1 is Sequence Length
                layer_betas[name] = diffs.mean().item()
            else:
                layer_betas[name] = torch.abs(torch.diff(acts, dim=0)).mean().item()

        if not layer_betas: return
        beta_values = list(layer_betas.values())
        beta_min, beta_max = min(beta_values), max(beta_values)
        beta_range = (beta_max - beta_min) + 1e-8

        for group in optimizer.param_groups:
            module_name = group.get('clockfield_module_name')
            if module_name and module_name in layer_betas:
                beta = layer_betas[module_name]
                beta_norm = (beta - beta_min) / beta_range
                gamma = np.exp(-self.alpha * beta_norm)
                # Ensure the metric acts as a shield
                group['weight_decay'] = max(self.lambda_max * gamma, 1e-5)

    def on_train_end(self, args, state, control, **kwargs):
        for h in self.hooks: h.remove()

# ============================================================================
# 3. SETUP & TRAINING EXECUTION (12GB VRAM Safe)
# ============================================================================

def create_clockfield_optimizer_groups(model, default_lr=1e-4):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        module_name = name.rsplit('.', 1)[0]
        group = {
            'params': [param],
            'lr': default_lr,
            'weight_decay': 0.0, 
            'clockfield_module_name': module_name if not any(nd in name for nd in no_decay) else None
        }
        optimizer_grouped_parameters.append(group)
    return optimizer_grouped_parameters

def main():
    # 1. BIGGER MODEL
    model_name = "gpt2-medium" 
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("\nInjecting Clockfield Attention...")
    count = replace_gpt2_attention_with_clockfield(model, model.config, alpha=3.0)
    print(f"Replaced {count} attention layers.")

    # 2. DISCUSSION DATASET
    print("Loading SamSum Conversational dataset...")
    dataset = load_dataset("knkarthick/samsum", split="train") 
    
    def tokenize_function(examples):
        return tokenizer(examples["dialogue"], truncation=True, max_length=256)
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("Configuring Clockfield Adaptive Optimizer...")
    optimizer_groups = create_clockfield_optimizer_groups(model, default_lr=1e-4)
    optimizer = torch.optim.AdamW(optimizer_groups)
    
    callbacks = [ClockfieldAdaptiveDecayCallback(lambda_max=0.1, alpha=3.0, update_every_steps=10)]

    # 3. SAFE TRAINING PARAMS
    training_args = TrainingArguments(
        output_dir="./gpt2_medium_clockfield_chat",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        logging_steps=50,
        save_steps=1000,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets,
        data_collator=data_collator,
        callbacks=callbacks,
        optimizers=(optimizer, None),
    )

    print("\n=== STARTING CLOCKFIELD GPT-2 MEDIUM CHAT TRAINING ===")
    trainer.train()

if __name__ == "__main__":
    main()