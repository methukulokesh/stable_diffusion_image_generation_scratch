import gradio as gr
import numpy as np
from PIL import Image
import traceback
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import CLIPTokenizer
from Stable_Diffusion import clip, encoder, decoder, diffusion
from pipeline import generate


# This function no longer opens the file, but directly processes the PIL Image object
def load_input_image(pil_image, device='cpu'):
    """
    Preprocess a PIL Image object to a tensor on the specified device.
    """
    if pil_image is None:
        return None
    
    image = pil_image.convert("RGB")
    image = image.resize((512, 512))
    image = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(device)
    print("Loaded and preprocessed the input image.")
    return tensor


class StableDiffusionEngine:
    def __init__(self, device):
        self.device = device
        self.models = None
        self.tokenizer = None

        self.repo_id = "hoshikrana/stable_diffusion_image_generation_v1"
        self.clip_filename = "model_safetensors_files/clip_model_state_dict.safetensors"
        self.encoder_filename = "model_safetensors_files/encoder_model_state_dict.safetensors"
        self.decoder_filename = "model_safetensors_files/decoder_model_state_dict.safetensors"
        self.diffusion_filename = "model_safetensors_files/diffusion_model_state_dict_merged.safetensors"

    def download_and_load(self, filename):
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            repo_type="model",
        )
        print(f"Downloaded {filename} to local path: {local_path}")
        weights = load_file(local_path, device=self.device)
        return weights

    def load_models(self):
        print("Downloading and loading models from Hugging Face Hub...")
        
        # Check for existence of clip, encoder, etc. before moving forward
        try:
            clip_model = clip.CLIP().to(self.device)
            encoder_model = encoder.VAE_Encoder().to(self.device)
            decoder_model = decoder.VAE_Decoder().to(self.device)
            diffusion_model = diffusion.Diffusion().to(self.device)

            clip_weights = self.download_and_load(self.clip_filename)
            encoder_weights = self.download_and_load(self.encoder_filename)
            decoder_weights = self.download_and_load(self.decoder_filename)
            diffusion_weights = self.download_and_load(self.diffusion_filename)

            clip_model.load_state_dict(clip_weights)
            encoder_model.load_state_dict(encoder_weights)
            decoder_model.load_state_dict(decoder_weights)
            diffusion_model.load_state_dict(diffusion_weights)

            self.tokenizer = CLIPTokenizer.from_pretrained(
                "hoshikrana/stable_diffusion_image_generation_v1",
                subfolder="tokenizer"
            )

            print("Models successfully loaded.")

            clip_model.eval()
            encoder_model.eval()
            decoder_model.eval()
            diffusion_model.eval()

            self.models = {
                'clip': clip_model,
                'encoder': encoder_model,
                'decoder': decoder_model,
                'diffusion': diffusion_model,
                'tokenizer': self.tokenizer
            }
            return True

        except Exception as e:
            print(f"Error downloading or loading models: {e}")
            print(traceback.format_exc())
            self.models = None
            self.tokenizer = None
            return False

    def preprocess_input_image(self, input_image):
        if input_image is not None:
            if isinstance(input_image, torch.Tensor):
                return input_image.detach().clone().to(self.device)
            elif isinstance(input_image, np.ndarray):
                return torch.from_numpy(input_image).to(self.device)
            else:
                raise ValueError("input_image must be a numpy array or torch tensor")
        return None

    def generate_image(
        self,
        prompt,
        uncond_prompt='',
        input_image=None,
        strength=0.75,
        do_cfg=True,
        cfg_scale=7.5,
        sampler_name='ddpm',
        n_inference_steps=50,
        seed=None
    ):
        if self.models is None or self.tokenizer is None:
            raise RuntimeError("Models and tokenizer not loaded. Call load_models() first.")
        
        # `input_image` is already a tensor, so no more preprocessing here
        input_tensor = input_image

        output_array = generate(
            prompt=prompt,
            uncond_prompt=uncond_prompt,
            input_image=input_tensor.squeeze() if input_tensor is not None else None,
            strength=strength,
            do_cfg=do_cfg,
            cfg_scale=cfg_scale,
            sampler_name=sampler_name,
            n_inference_steps=n_inference_steps,
            models=self.models,
            seed=seed,
            device=self.device,
            tokenizer=self.tokenizer,
        )

        output_image = Image.fromarray(output_array)
        print("Image generation complete.")
        return output_image


# Initialize engine and load models
device = "cuda" if torch.cuda.is_available() else "cpu"
engine = StableDiffusionEngine(device=device)
engine.load_models()


def generate_image(
        prompt,
        neg_prompt="blurry, low-res",
        strength=0.8,
        steps=20,
        input_image_file=None, # This is now a PIL Image object
):
    try:
        input_image = None
        if input_image_file is not None:
            # Pass the PIL Image directly to the modified function
            input_image = load_input_image(input_image_file, device='cpu')
        
        print("Generating image please wait.....")
        generated_image = engine.generate_image(
            prompt=prompt,
            uncond_prompt=neg_prompt,
            input_image=input_image,
            strength=strength,
            do_cfg=True,
            cfg_scale=7.5,
            sampler_name="ddpm",
            n_inference_steps=steps,
            seed=42,
        )

        # The engine's `generate_image` already returns a PIL Image
        return generated_image, ""

    except Exception as e:
        return None, f"Error: {e}\n\nTraceback:\n{traceback.format_exc()}"


def set_loading():
    return "Image generating, please wait..."


with gr.Blocks() as demo:
    prompt = gr.Textbox(label="Prompt", lines=2)
    neg_prompt = gr.Textbox(label="Negative Prompt", value="blurry, low-res", lines=1)
    strength = gr.Slider(label="Strength", minimum=0.1, maximum=1.0, step=0.01, value=0.8)
    steps = gr.Slider(label="Inference Steps", minimum=10, maximum=100, step=1, value=20)
    # The Gradio component type="pil" returns a PIL Image object
    input_image = gr.Image(label="Input Image (optional)", type="pil")

    output_image = gr.Image(label="Generated Image")
    status = gr.Textbox(label="Status", interactive=False, value="")
    generate_button = gr.Button("Generate Image")

    generate_button.click(set_loading, [], status)
    generate_button.click(generate_image, [prompt, neg_prompt, strength, steps, input_image], [output_image, status])

demo.launch()
