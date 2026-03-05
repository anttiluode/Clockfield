"""
Clockfield Live v5: The Immortal Agent
======================================
Includes Document Ingestion, Persistent Saving, and Left-Side Truncation fixes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
import tkinter as tk
from tkinter import scrolledtext, filedialog
import threading
import glob
import os
import json
import time
from datetime import datetime
from collections import deque

# ============================================================================
# 1. CLOCKFIELD ATTENTION
# ============================================================================

class ClockfieldGPT2AttentionLive(GPT2Attention):
    def __init__(self, config, is_cross_attention=False, layer_idx=None, alpha=3.0):
        super().__init__(config, is_cross_attention=is_cross_attention, layer_idx=layer_idx)
        self.alpha = alpha
        self.register_buffer('beta_ema', torch.full((self.num_heads,), 0.35))
        self.last_token_sharpness = None
        self._update_beta = False

    def compute_head_beta_and_token_sharpness(self, q, k):
        with torch.no_grad():
            attn_weights = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.head_dim)
            seq_len = q.size(-2)
            causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=q.device)).view(1, 1, seq_len, seq_len)
            attn_weights = attn_weights.masked_fill(causal_mask == 0, float('-inf'))
            attn_probs = F.softmax(attn_weights, dim=-1)

            sharpness = attn_probs.max(dim=-1)[0]
            head_beta = sharpness.mean(dim=(0, 2))
            received = attn_probs.max(dim=-2)[0]
            token_sharp = received.max(dim=1)[0].mean(dim=0)

        return head_beta, token_sharp

    def forward(
        self, hidden_states, layer_past=None, past_key_values=None,
        attention_mask=None, head_mask=None, encoder_hidden_states=None,
        encoder_attention_mask=None, use_cache=False, output_attentions=False, **kwargs
    ):
        if layer_past is None and past_key_values is not None:
            layer_past = past_key_values

        bsz, q_len, _ = hidden_states.size()

        if q_len >= 8:
            with torch.no_grad():
                query, key, _ = self.c_attn(hidden_states).split(self.split_size, dim=2)
                q = query.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                k = key.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                current_beta, token_sharp = self.compute_head_beta_and_token_sharpness(q, k)
                self.last_token_sharpness = token_sharp

                if self._update_beta:
                    self.beta_ema = 0.9 * self.beta_ema + 0.1 * current_beta

        beta_min, beta_max = self.beta_ema.min(), self.beta_ema.max()
        beta_norm = (self.beta_ema - beta_min) / (beta_max - beta_min + 1e-8)
        gamma = torch.exp(-self.alpha * beta_norm)

        kv_seq_len = q_len
        if layer_past is not None:
            if hasattr(layer_past, "get_seq_length"): kv_seq_len += layer_past.get_seq_length(self.layer_idx)
            else:
                curr = layer_past
                while isinstance(curr, (tuple, list)): curr = curr[0]
                if hasattr(curr, "shape"): kv_seq_len += curr.shape[-2]

        min_window = min(16, kv_seq_len)
        window_sizes = (min_window + (kv_seq_len - min_window) * (1.0 - gamma)).view(1, self.num_heads, 1, 1)

        seq_ids_q = torch.arange(q_len, device=hidden_states.device)
        seq_ids_k = torch.arange(kv_seq_len, device=hidden_states.device)
        if q_len == 1 and kv_seq_len > 1: seq_ids_q = torch.tensor([kv_seq_len - 1], device=hidden_states.device)

        distance = seq_ids_q.view(1, 1, q_len, 1) - seq_ids_k.view(1, 1, 1, kv_seq_len)
        clockfield_mask_bool = (distance >= 0) & (distance <= window_sizes)

        # ATTENTION SINK: Never forget the prompt formatting
        if kv_seq_len >= 4:
            clockfield_mask_bool[:, :, :, :4] = True

        mask_value = torch.tensor(-10000.0, device=hidden_states.device, dtype=hidden_states.dtype)
        clockfield_additive = torch.zeros((1, self.num_heads, q_len, kv_seq_len), device=hidden_states.device, dtype=hidden_states.dtype)
        clockfield_additive = clockfield_additive.masked_fill(~clockfield_mask_bool, mask_value)

        if attention_mask is not None: combined_mask = attention_mask + clockfield_additive
        else: combined_mask = clockfield_additive

        return super().forward(
            hidden_states=hidden_states, layer_past=layer_past, attention_mask=combined_mask,
            head_mask=head_mask, encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask, use_cache=use_cache,
            output_attentions=output_attentions, **kwargs
        )

def replace_attention_live(module, config, alpha=3.0):
    count = 0
    for name, child in module.named_children():
        if isinstance(child, GPT2Attention):
            is_cross = getattr(child, "is_cross_attention", False)
            layer_idx = getattr(child, "layer_idx", None)
            live_attn = ClockfieldGPT2AttentionLive(config, is_cross_attention=is_cross, layer_idx=layer_idx, alpha=alpha)
            live_attn.load_state_dict(child.state_dict(), strict=False)
            setattr(module, name, live_attn)
            count += 1
        else: count += replace_attention_live(child, config, alpha)
    return count

def set_beta_update_gate(model, enabled: bool):
    for mod in model.modules():
        if hasattr(mod, '_update_beta'): mod._update_beta = enabled

# ============================================================================
# 2. HIPPOCAMPUS (With Save/Load)
# ============================================================================

class ClockfieldMemory:
    def __init__(self, max_memories=50, sharpness_threshold=0.35, decay_rate=0.998):
        self.memories = deque(maxlen=max_memories)
        self.sharpness_threshold = sharpness_threshold
        self.decay_rate = decay_rate

    def consolidate(self, text, token_ids, sharpness_scores, tokenizer):
        if sharpness_scores is None or len(sharpness_scores) == 0: return
        sharp_mask = sharpness_scores > self.sharpness_threshold
        if not sharp_mask.any(): return
        
        sharp_indices = torch.where(sharp_mask)[0].cpu().tolist()
        spans =[]
        current_span = [sharp_indices[0]]
        for idx in sharp_indices[1:]:
            if idx == current_span[-1] + 1: current_span.append(idx)
            else:
                if len(current_span) >= 2: spans.append(current_span)
                current_span = [idx]
        if len(current_span) >= 2: spans.append(current_span)

        for span in spans:
            start, end = max(0, span[0] - 3), min(len(token_ids), span[-1] + 4)
            memory_text = tokenizer.decode(token_ids[start:end], skip_special_tokens=True).strip()
            if len(memory_text) > 5:
                self.memories.append({
                    'text': memory_text,
                    'sharpness': float(sharpness_scores[span[0]:span[-1]+1].mean()),
                    'strength': 1.0,
                })

    def decay(self):
        surviving = deque(maxlen=self.memories.maxlen)
        for mem in self.memories:
            mem['strength'] *= self.decay_rate
            if mem['strength'] > 0.1: surviving.append(mem)
        self.memories = surviving

    def get_context_prefix(self):
        if not self.memories: return ""
        sorted_mems = sorted(self.memories, key=lambda m: m['strength'], reverse=True)[:6]
        return "Context from earlier:\n" + "\n".join([f"- {m['text']}" for m in sorted_mems]) + "\n\n"

    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(list(self.memories), f, indent=2)

    def load(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.memories = deque(json.load(f), maxlen=self.memories.maxlen)

# ============================================================================
# 3. ONLINE LEARNING ENGINE
# ============================================================================

class ClockfieldLearningEngine:
    def __init__(self, model, tokenizer, device, learning_rate=1e-6, lambda_max=0.01, alpha=3.0):
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.alpha, self.lambda_max = alpha, lambda_max
        self.optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)

    def learn_from_exchange(self, conversation_text):
        set_beta_update_gate(self.model, True)
        self.model.train()

        tokens = self.tokenizer(conversation_text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
        if tokens.input_ids.shape[1] < 8:
            set_beta_update_gate(self.model, False)
            return None

        outputs = self.model(**tokens, labels=tokens.input_ids)
        loss = outputs.loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        # Adaptive Decay
        with torch.no_grad():
            betas =[mod.beta_ema for mod in self.model.modules() if hasattr(mod, 'beta_ema')]
            if betas:
                all_beta = torch.cat(betas)
                beta_norm = ((all_beta.mean() - all_beta.min()) / (all_beta.max() - all_beta.min() + 1e-8)).item()
                decay = self.lambda_max * np.exp(-self.alpha * beta_norm)
                for name, param in self.model.named_parameters():
                    if param.requires_grad and 'bias' not in name and 'LayerNorm' not in name:
                        param.mul_(1.0 - self.optimizer.param_groups[0]['lr'] * decay)

        set_beta_update_gate(self.model, False)
        self.model.eval()
        return loss.item()

# ============================================================================
# 4. LOAD AND METRICS
# ============================================================================

def get_clockfield_metrics(model, alpha=3.0):
    betas =[mod.beta_ema for mod in model.modules() if hasattr(mod, 'beta_ema')]
    if not betas: return {'beta_var': 0, 'gamma_mean': 0}
    stacked = torch.cat(betas)
    return {'beta_var': torch.var(stacked).item(), 'gamma_mean': torch.mean(torch.exp(-alpha * stacked)).item()}

def load_model():
    print("Loading gpt2-medium base...")
    model_name = "gpt2-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # CRITICAL: Force tokenizer to truncate the OLD history, not your new message
    tokenizer.truncation_side = 'left'  
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    replace_attention_live(model, model.config, alpha=3.0)

    save_dir = "./clockfield_saved_brain"
    checkpoints = sorted(glob.glob("./gpt2_medium_clockfield_chat/checkpoint-*"), key=os.path.getctime)
    
    if os.path.exists(save_dir):
        print(f"Loading previously SAVED live brain from {save_dir}...")
        model.load_state_dict(torch.load(os.path.join(save_dir, "weights.pt"), map_location="cpu"), strict=False)
    elif checkpoints:
        latest = checkpoints[-1]
        print(f"Loading initial trained weights from {latest}...")
        sf_path = os.path.join(latest, "model.safetensors")
        if os.path.exists(sf_path):
            from safetensors.torch import load_file
            model.load_state_dict(load_file(sf_path), strict=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return model, tokenizer, device

# ============================================================================
# 5. GUI & SYSTEM LOOP
# ============================================================================

class ClockfieldLiveApp:
    def __init__(self, root, model, tokenizer, device):
        self.root = root
        self.root.title("Clockfield Live v5 - Immortal Agent")
        self.root.geometry("800x750")
        self.root.configure(bg='#0d0f14')

        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.memory = ClockfieldMemory()
        if os.path.exists("./clockfield_saved_brain/hippocampus.json"):
            self.memory.load("./clockfield_saved_brain/hippocampus.json")

        self.learner = ClockfieldLearningEngine(model, tokenizer, device)
        self.conversation_history = ""
        self.learning_enabled = False # Start OFF by default for safety

        self._build_ui()

    def _build_ui(self):
        # Top Bar
        tk.Label(self.root, text="CLOCKFIELD LIVE v5", font=('Consolas', 14, 'bold'), fg='#00c8ff', bg='#0d0f14').pack(pady=(8, 0))
        
        self.metrics_frame = tk.Frame(self.root, bg='#1a1e29')
        self.metrics_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.metric_labels = {}
        for name in ['β-var', 'Γ-mean', 'Memories']:
            frame = tk.Frame(self.metrics_frame, bg='#1a1e29')
            frame.pack(side=tk.LEFT, padx=10, pady=5)
            tk.Label(frame, text=name, font=('Consolas', 8), fg='#4a5568', bg='#1a1e29').pack()
            lbl = tk.Label(frame, text="—", font=('Consolas', 10, 'bold'), fg='#00c8ff', bg='#1a1e29')
            lbl.pack()
            self.metric_labels[name] = lbl

        # Buttons
        self.learn_btn = tk.Button(self.metrics_frame, text="🧠 OFF", font=('Consolas', 8, 'bold'), fg='#ff6b35', bg='#1a1e29', command=self._toggle_learning)
        self.learn_btn.pack(side=tk.RIGHT, padx=5)

        self.read_btn = tk.Button(self.metrics_frame, text="📄 READ DOC", font=('Consolas', 8, 'bold'), fg='#e8eaf0', bg='#1a1e29', command=self.ingest_document)
        self.read_btn.pack(side=tk.RIGHT, padx=5)

        self.save_btn = tk.Button(self.metrics_frame, text="💾 SAVE BRAIN", font=('Consolas', 8, 'bold'), fg='#e8eaf0', bg='#1a1e29', command=self.save_brain)
        self.save_btn.pack(side=tk.RIGHT, padx=5)

        # Chat area
        self.chat = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state='disabled', font=('Consolas', 10), bg='#12151c', fg='#e8eaf0', padx=10, pady=8)
        self.chat.pack(padx=10, pady=5, fill=tk.BOTH, expand=True)
        self.chat.tag_configure('user', foreground='#00c8ff')
        self.chat.tag_configure('ai', foreground='#34d399')
        self.chat.tag_configure('sys', foreground='#ffb86c')

        # Input area
        inp_frame = tk.Frame(self.root, bg='#0d0f14')
        inp_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.entry = tk.Entry(inp_frame, font=('Consolas', 11), bg='#1a1e29', fg='#e8eaf0', insertbackground='#00c8ff')
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 8), ipady=6)
        self.entry.bind("<Return>", self.send_message)
        self.send_btn = tk.Button(inp_frame, text="Send", command=self.send_message, bg='#00c8ff', font=('Consolas', 10, 'bold'))
        self.send_btn.pack(side=tk.RIGHT)

        self._update_metrics()
        self._append('sys', "System Online. Type a message or click READ DOC to ingest text files.")

    def _append(self, tag, text):
        self.chat.config(state='normal')
        self.chat.insert(tk.END, text + '\n', tag)
        self.chat.yview(tk.END)
        self.chat.config(state='disabled')

    def _toggle_learning(self):
        self.learning_enabled = not self.learning_enabled
        self.learn_btn.config(text="🧠 ON" if self.learning_enabled else "🧠 OFF", fg='#34d399' if self.learning_enabled else '#ff6b35')

    def _update_metrics(self):
        m = get_clockfield_metrics(self.model)
        self.metric_labels['β-var'].config(text=f"{m['beta_var']:.4f}")
        self.metric_labels['Γ-mean'].config(text=f"{m['gamma_mean']:.3f}")
        self.metric_labels['Memories'].config(text=f"{len(self.memory.memories)}")
        self.root.after(2000, self._update_metrics)

    # --- NEW FEATURE: SAVE BRAIN ---
    def save_brain(self):
        self._append('sys', "Saving neural weights and Hippocampus to disk...")
        save_dir = "./clockfield_saved_brain"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(save_dir, "weights.pt"))
        self.memory.save(os.path.join(save_dir, "hippocampus.json"))
        self._append('sys', "Save complete! This brain state will be loaded on next startup.")

    # --- NEW FEATURE: READ DOCUMENT ---
    def ingest_document(self):
        filepath = filedialog.askopenfilename(filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if not filepath: return
        threading.Thread(target=self._process_document_thread, args=(filepath,), daemon=True).start()

    def _process_document_thread(self, filepath):
        self.root.after(0, self._append, 'sys', f"Reading {os.path.basename(filepath)}...")
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                words = f.read().split()

            chunk_size, overlap = 150, 30
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i+chunk_size])
                if len(chunk) < 10: continue

                prompt = f"Document excerpt: {chunk}"
                
                # 1. Extract memory by running standard forward pass
                tokens = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(self.device)
                self.model.eval()
                with torch.no_grad():
                    self.model(**tokens)
                
                # Consolidate high-sharpness concepts
                sharp = self._get_token_sharpness()
                if sharp is not None:
                    self.memory.consolidate(prompt, tokens.input_ids[0], sharp, self.tokenizer)

                # 2. Learn weights if 🧠 is ON
                if self.learning_enabled:
                    self.learner.learn_from_exchange(prompt)

            self.root.after(0, self._append, 'sys', f"Finished reading! Extracted memories and adjusted spacetime metric.")
        except Exception as e:
            self.root.after(0, self._append, 'sys', f"Failed to read doc: {e}")

    # --- CHAT LOGIC ---
    def send_message(self, event=None):
        user_input = self.entry.get().strip()
        if not user_input: return
        self.entry.delete(0, tk.END)
        self._append('user', f"You: {user_input}")
        threading.Thread(target=self._process_chat, args=(user_input,), daemon=True).start()

    def _process_chat(self, user_input):
        import re
        self.conversation_history += f"User: {user_input}\nClockfield:"
        # TRUNCATION FIX: Safely limit to 1500 chars so the tokenizer doesn't drop new text
        if len(self.conversation_history) > 1500:
            self.conversation_history = self.conversation_history[-1500:]

        full_context = self.memory.get_context_prefix() + self.conversation_history

        inputs = self.tokenizer(full_context, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=40, temperature=0.6, top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id, repetition_penalty=1.15,
                use_cache=False  # <--- MUST HAVE THIS!
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        response = response.split('\n')[0].replace("User:", "").replace("Clockfield:", "").strip()
        
        self.conversation_history += f" {response}\n"
        self.root.after(0, self._append, 'ai', f"Clockfield: {response}")

        sharp = self._get_token_sharpness()
        if sharp is not None:
            ex_tokens = self.tokenizer(response, return_tensors="pt").input_ids[0]
            self.memory.consolidate(response, ex_tokens[:len(sharp)], sharp[:len(ex_tokens)], self.tokenizer)
            self.memory.decay()

        if self.learning_enabled and len(response) > 2 and not re.search(r'([^\w\s.,!?\'"])\1{3,}', response):
            self.learner.learn_from_exchange(self.conversation_history[-500:])

    def _get_token_sharpness(self):
        s_list =[m.last_token_sharpness for m in self.model.modules() if hasattr(m, 'last_token_sharpness') and m.last_token_sharpness is not None]
        if not s_list: return None
        return torch.stack([s[:min(x.shape[0] for x in s_list)] for s in s_list]).max(dim=0)[0]


if __name__ == "__main__":
    model, tokenizer, device = load_model()
    root = tk.Tk()
    app = ClockfieldLiveApp(root, model, tokenizer, device)
    root.mainloop()