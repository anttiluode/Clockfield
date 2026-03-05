Clockfield Applied: Spacetime Engineering in LLMs and Vision

This directory takes the theoretical Clockfield proper-time metric —

    dτ_i = exp(−α ⋅ β_i) dt

—and engineers it directly into the forward passes of HuggingFace
Transformers and Latent Diffusion models.

Instead of just using the Clockfield as a diagnostic probe, these
applications use it as an active physics engine. The models dynamically
curve their own internal spacetime, resizing causal attention windows,
gating live learning, and selectively freezing their own generated
hallucinations.

Requirements & Installation

Run this single pip command to install the dependencies for all
applications in this folder:

pip install torch transformers datasets numpy opencv-python diffusers
accelerate pillow gradio safetensors

1. The Immortal Agent (clockfield_chat_interface.py)

What it is: A live small conversational AI (GPT‑2 Medium) that you can chat
with, feed documents to, and leave in a continuous state of online
learning (🧠 ON or OFF on upper left of gui) without triggering catastrophic forgetting.

How it relates to the Clockfield:

Standard LLMs left in continuous learning mode will poison their own
weights (mode collapse) because uniform weight decay erodes their
foundational logic.

We apply the Clockfield Adaptive Decay: If a circuit has sharp,
crystallized attention (β → 1), local time slows down (Γ → 0) and its
weight decay drops to near‑zero. This mathematically protects its core
logic.

We apply Dynamic Lightcones: The causal attention mask is no longer a
static triangle. “Confused” heads (low β) have their context windows
shrunk so they focus locally, while “confident” heads (high β) open
their windows to see the entire conversation history.

Key Features:

Hippocampus Memory Continuously measures the β‑sharpness of the
conversation. When it detects highly structured concepts (names, places,
facts), it permanently commits them to a .json memory drive.

Document Ingestion Click READ DOC to feed it raw text. It reads the
document, extracts the highest‑β memories, and (if learning is enabled)
seamlessly adjusts its weights.

Brain Saving Click SAVE BRAIN to dump the live modified neural weights
and memories to disk, allowing the agent to persist across reboots.

2. Relativistic Vision (clockfield_janus.py)

What it is: A real‑time webcam‑to‑image cinematic hallucination engine
powered by SDXL‑Turbo but governed by 2D proper‑time mechanics.

How it relates to the Clockfield:

Instead of text, β is calculated using the Spatial Laplacian of the
generated image (measuring visual crystallization or sharpness).

The engine projects a 2D spacetime metric over the canvas (visible in
the UI as a live heatmap).

Where the model hallucinates a perfect sharp structure (for example a
golden statue), β spikes, time freezes (Γ → 0), and those pixels lock
onto the screen, immune to further diffusion noise.

Where there is empty space or sudden webcam motion (such as you waving
your hand), β drops, time speeds up (Γ → 1), and the model rapidly
ingests the new visual data to dream up something new.

You effectively act as a “Time Lord” — your physical motion shatters the
visual crystals, speeding up time and forcing the AI to redraw reality
around you.

3. The Spacetime LLM Trainer (Clockfield_llm_trainer.py)

What it is: The foundational training script used to bake the Clockfield
physics into a raw GPT‑2 Medium model using the SAMSum conversational
dataset.

The “Black Hole” Fix (Temporal Roughness vs Feature Roughness):

Initial attempts to apply the Clockfield to Transformers resulted in a
“temporal black hole”. Measuring β across the feature dimension failed
because LayerNorm artificially flattens features, tricking the metric
into thinking uniform noise was a perfect crystal. This froze the
network’s proper time to zero.

This trainer introduces the mathematical fixes:

Causal Attention Sharpness β is measured by the entropy (peakiness) of
the QKᵀ attention distribution after the causal mask is applied.

Temporal Roughness For dense layers, β is measured across dim=1
(sequence length), measuring how violently the network’s thoughts jump
from word‑to‑word in time rather than across abstract features.

Quick Start

To train the base conversational brain:

python Clockfield_llm_trainer.py

To boot up the Immortal Agent (requires trained checkpoint):

python clockfield_chat_interface.py

To launch the Relativistic Vision Engine (requires a webcam):

python clockfield_janus.py
