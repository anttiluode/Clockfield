import gradio as gr
import numpy as np
import cv2
import torch
from diffusers import AutoPipelineForImage2Image
from PIL import Image
import time

class ClockfieldVisionEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"🚀 Initializing Clockfield Vision Engine (Relativistic Spacetime)...")
        
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=self.dtype, variant="fp16"
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        
        self.last_dream_np = None
        self.prompt = "a cinematic portrait of pharao"
        self.active = False
        self.last_txt, self.embeds, self.pooled = "", None, None
        
        # Clockfield Physics Constants
        self.alpha = 4.0          # Coupling constant: How aggressively time freezes
        self.beta_memory = None   # EMA of the spatial crystallization

    def get_embeds(self):
        if self.prompt == self.last_txt and self.embeds is not None:
            return self.embeds, self.pooled
        with torch.no_grad():
            pe, _, ppe, _ = self.pipe.encode_prompt(
                prompt=self.prompt, device=self.device, 
                num_images_per_prompt=1, do_classifier_free_guidance=False
            )
            self.last_txt, self.embeds, self.pooled = self.prompt, pe, ppe
            return pe, ppe

    def process_frame(self, frame_bgr, zoom_val=1.00):
        raw_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        raw_res = cv2.resize(raw_rgb, (512, 512)).astype(np.float32) / 255.0
        
        if self.last_dream_np is None:
            self.last_dream_np = raw_res * 255.0
            self.beta_memory = np.zeros((512, 512), dtype=np.float32)
            return self.last_dream_np.astype(np.uint8), np.zeros((512, 512, 3), dtype=np.uint8), 1.0

        dream_float = self.last_dream_np.astype(np.float32) / 255.0

        # =================================================================
        # 1. THE CLOCKFIELD METRIC (Spatial β-Sieve)
        # Measure crystallization (sharpness/structure) of the hallucination
        # =================================================================
        gray_dream = cv2.cvtColor(dream_float, cv2.COLOR_RGB2GRAY)
        
        # Laplacian measures local spatial frequency (structure)
        laplacian = cv2.Laplacian(gray_dream, cv2.CV_32F)
        current_beta = np.abs(laplacian)
        
        # Create "Temporal Wells" by blurring the beta field slightly
        current_beta = cv2.GaussianBlur(current_beta, (31, 31), 0)
        
        # EMA update of the metric (like the Transformer heads)
        self.beta_memory = 0.8 * self.beta_memory + 0.2 * current_beta

        # Normalize Beta to [0, 1]
        b_min, b_max = self.beta_memory.min(), self.beta_memory.max()
        beta_norm = (self.beta_memory - b_min) / (b_max - b_min + 1e-8)

        # CALCULATE PROPER-TIME (Γ)
        # High Beta -> Gamma approaches 0 (Time Freezes, weights/pixels protected)
        # Low Beta -> Gamma approaches 1 (Time Fast, rapid erosion/redrawing)
        gamma_2d = np.exp(-self.alpha * beta_norm)
        gamma_3d = np.expand_dims(gamma_2d, axis=-1)

        # =================================================================
        # 2. RELATIVISTIC SPATIAL BLEND
        # =================================================================
        h, w = dream_float.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), 0, zoom_val)
        zoomed_dream = cv2.warpAffine(dream_float, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        # The Magic: Instead of a flat alpha blend, the canvas uses Spacetime!
        # Where time is fast (Gamma=1), it ingests the webcam.
        # Where time is frozen (Gamma=0), it permanently loops its own perfect dream.
        blended_float = (raw_res * gamma_3d) + (zoomed_dream * (1.0 - gamma_3d))
        blended_uint8 = np.clip(blended_float * 255.0, 0, 255).astype(np.uint8)

        # =================================================================
        # 3. GLOBAL TIME MODULATION (Adaptive Decay Analog)
        # =================================================================
        gamma_mean = np.mean(gamma_2d)
        # SDXL-Turbo Strength: Lower strength = less destruction = frozen time
        dynamic_strength = np.clip(0.35 + (gamma_mean * 0.45), 0.35, 0.85)
        
        pe, ppe = self.get_embeds()
        
        with torch.no_grad():
            result_pil = self.pipe(
                prompt_embeds=pe, pooled_prompt_embeds=ppe, 
                image=Image.fromarray(blended_uint8), strength=dynamic_strength, 
                num_inference_steps=4, guidance_scale=0.0, output_type="pil"
            ).images[0]

        self.last_dream_np = np.array(result_pil)

        # Generate Heatmap for UI Visualization (Inferno: Black=Frozen, Yellow=Fast Time)
        gamma_heatmap = (gamma_2d * 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(gamma_heatmap, cv2.COLORMAP_INFERNO)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        return self.last_dream_np, heatmap_rgb, gamma_mean

# --- Loop ---
engine = ClockfieldVisionEngine()

def cinematic_loop(cam_id):
    cap = cv2.VideoCapture(int(cam_id), cv2.CAP_DSHOW)
    engine.active = True
    while engine.active:
        ret, frame = cap.read()
        if not ret: break
        dream, heatmap, gamma_mean = engine.process_frame(frame)
        
        status = f"Γ-Mean (Global Time Speed): {gamma_mean:.3f} | Prompt: {engine.prompt}"
        yield dream, heatmap, status
    cap.release()

# --- Full Screen CSS ---
css = """
#dream-container { height: 75vh !important; }
#metric-container { height: 75vh !important; }
#prompt-box { font-size: 20px !important; }
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=css) as demo:
    gr.Markdown("# 🕰️ Clockfield Vision: Relativistic AI Generation")
    gr.Markdown("*When the hallucination crystallizes, time slows down. Black = Frozen Time. Yellow = Fast Time.*")
    
    with gr.Row():
        with gr.Column(elem_id="dream-container", scale=1):
            v_dream = gr.Image(label="Conscious Dream", interactive=False)
        with gr.Column(elem_id="metric-container", scale=1):
            v_metric = gr.Image(label="Γ (Proper-Time) Metric", interactive=False)
    
    with gr.Row():
        prompt = gr.Textbox(
            label="Semantic Target", 
            placeholder="Change the dream here...", 
            value=engine.prompt,
            elem_id="prompt-box",
            scale=4
        )
        btn_start = gr.Button("🔴 ACTIVATE CLOCKFIELD", variant="primary", scale=1)
        btn_stop = gr.Button("⬛ STOP", scale=1)

    with gr.Accordion("Technical Settings", open=False):
        cam = gr.Dropdown([0, 1], value=0, label="Camera Source")
        status = gr.Label(value="Standby")

    # Interactive Updates
    prompt.change(lambda x: setattr(engine, 'prompt', x), prompt)
    btn_start.click(cinematic_loop, [cam],[v_dream, v_metric, status])
    btn_stop.click(lambda: setattr(engine, 'active', False))

demo.queue().launch()