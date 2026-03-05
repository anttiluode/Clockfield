import gradio as gr
import numpy as np
import cv2
import torch
import time
from diffusers import AutoPipelineForInpainting
from PIL import Image

class PureClockfieldEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"🚀 Initializing PURE Clockfield Engine (Isolated Spacetime)...")
        
        # Inpainting Pipeline to access the True Latent Mask
        self.pipe = AutoPipelineForInpainting.from_pretrained(
            "stabilityai/sdxl-turbo", torch_dtype=self.dtype, variant="fp16"
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)
        
        self.last_dream_np = None
        self.prompt = "a cinematic, hyper-detailed portrait of a golden marble statue of a greek god, nebula background, masterpiece"
        self.active = False
        self.last_txt, self.embeds, self.pooled = "", None, None
        
        # Spacetime Controls
        self.alpha = 8.0          # Viscosity: How hard the space freezes
        self.zoom_val = 1.02      # Erosion: How fast we travel into the dream
        self.beta_memory = None   

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

    def process_frame(self):
        # 0. The Genesis Pulse: Kickstart the universe with pure noise if it's empty
        if self.last_dream_np is None:
            self.last_dream_np = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
            self.beta_memory = np.zeros((512, 512), dtype=np.float32)

        dream_float = self.last_dream_np.astype(np.float32) / 255.0

        # =================================================================
        # 1. THE β-SIEVE (Measuring Internal Crystallization)
        # =================================================================
        gray_dream = cv2.cvtColor(dream_float, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray_dream, cv2.CV_32F)
        current_beta = np.abs(laplacian)
        current_beta = cv2.GaussianBlur(current_beta, (31, 31), 0)
        
        self.beta_memory = 0.8 * self.beta_memory + 0.2 * current_beta
        b_min, b_max = self.beta_memory.min(), self.beta_memory.max()
        beta_norm = (self.beta_memory - b_min) / (b_max - b_min + 1e-8)

        # =================================================================
        # 2. PROPER-TIME (Γ) -> THE ISOLATED LATENT MASK
        # =================================================================
        gamma_2d = np.exp(-self.alpha * beta_norm)
        
        # White (255) = Fast Time (Hallucinate new details)
        # Black (0)   = Frozen Time (Lock current structure in VRAM)
        latent_mask_uint8 = (gamma_2d * 255.0).astype(np.uint8)
        latent_mask_pil = Image.fromarray(latent_mask_uint8)

        # =================================================================
        # 3. SPATIAL EROSION (Traveling into the Dream)
        # =================================================================
        h, w = dream_float.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), 0, self.zoom_val)
        # We stretch the dream. The frozen crystals stretch and eventually break,
        # forcing them into the "Fast Time" zones to be re-hallucinated.
        zoomed_dream = cv2.warpAffine(dream_float, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        zoomed_uint8 = np.clip(zoomed_dream * 255.0, 0, 255).astype(np.uint8)

        # =================================================================
        # 4. SPATIALLY-VARIANT DIFFUSION
        # =================================================================
        pe, ppe = self.get_embeds()
        
        with torch.no_grad():
            result_pil = self.pipe(
                prompt_embeds=pe, pooled_prompt_embeds=ppe, 
                image=Image.fromarray(zoomed_uint8), 
                mask_image=latent_mask_pil,
                strength=0.85, # High chaos for the exposed latents
                num_inference_steps=4, guidance_scale=0.0, output_type="pil"
            ).images[0]

        self.last_dream_np = np.array(result_pil)

        # Heatmap Generation
        heatmap_colored = cv2.applyColorMap(latent_mask_uint8, cv2.COLORMAP_INFERNO)
        heatmap_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        return self.last_dream_np, heatmap_rgb

# --- Loop ---
engine = PureClockfieldEngine()

def cinematic_loop():
    engine.active = True
    while engine.active:
        dream, heatmap = engine.process_frame()
        status = f"Status: Dreaming | Viscosity: {engine.alpha} | Erosion: {engine.zoom_val}"
        yield dream, heatmap, status
        # A tiny sleep to prevent UI locking, though GPU will bottleneck anyway
        time.sleep(0.01)

def reset_dream():
    engine.last_dream_np = None
    engine.beta_memory = None
    return "Dream reset. Awaiting new Genesis pulse."

# --- UI (Immersive Full Screen CSS) ---
css = """
body, html { margin: 0; padding: 0; background-color: #050505; color: #fff; }
.gradio-container { max-width: 100% !important; padding: 10px !important; }
#dream-container img { height: 75vh !important; width: 100%; object-fit: contain; background: #000; border-radius: 8px;}
#metric-container img { height: 75vh !important; width: 100%; object-fit: contain; background: #000; border-radius: 8px;}
#controls { background: #111; padding: 15px; border-radius: 8px; border: 1px solid #333; }
#prompt-box textarea { font-size: 18px !important; font-weight: bold; }
"""

with gr.Blocks(theme=gr.themes.Monochrome(), css=css) as demo:
    with gr.Row():
        with gr.Column(elem_id="dream-container", scale=1):
            v_dream = gr.Image(label="The Conscious Dream", interactive=False, show_label=False)
        with gr.Column(elem_id="metric-container", scale=1):
            v_metric = gr.Image(label="Γ (Proper-Time) Mask", interactive=False, show_label=False)
    
    with gr.Column(elem_id="controls"):
        with gr.Row():
            prompt = gr.Textbox(
                label="The Seed Thought", 
                value=engine.prompt, elem_id="prompt-box", scale=4
            )
            btn_start = gr.Button("🔴 INITIATE DREAM", variant="primary", scale=1)
            btn_stop = gr.Button("⬛ WAKE UP", scale=1)
            btn_reset = gr.Button("🌀 RESET UNIVERSE", scale=1)

        with gr.Row():
            viscosity_slider = gr.Slider(
                minimum=1.0, maximum=15.0, value=engine.alpha, step=0.5,
                label="Spacetime Viscosity (α)",
                info="Higher = Blacker mask. The hallucination freezes rapidly and acts like solid stone."
            )
            zoom_slider = gr.Slider(
                minimum=1.0, maximum=1.1, value=engine.zoom_val, step=0.005,
                label="Erosion Rate (Zoom)",
                info="1.0 = Perfect Stillness. > 1.0 = Travel into the dream, melting structures as they stretch."
            )
        
        status = gr.Label(value="Void. Waiting for initiation.")

    # Event Wiring
    prompt.change(lambda x: setattr(engine, 'prompt', x), prompt)
    viscosity_slider.change(lambda x: setattr(engine, 'alpha', x), viscosity_slider)
    zoom_slider.change(lambda x: setattr(engine, 'zoom_val', x), zoom_slider)
    
    btn_start.click(cinematic_loop, None, [v_dream, v_metric, status])
    btn_stop.click(lambda: setattr(engine, 'active', False))
    btn_reset.click(reset_dream, None, status)

demo.queue().launch()