from flask import Flask, request, jsonify, send_file
import torch
from diffusers import FluxPipeline
from PIL import Image
from io import BytesIO

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Initialize the pipeline with optimizations
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.float16,
    use_safetensors=True
)

pipe.load_lora_weights("nissan_kicks_nt.safetensors", dtype=torch.float16, device="cuda")
pipe = pipe.to("cuda")

prompt="Una Nissan Kicks en un mundo de caramelo"

with torch.inference_mode(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
    # Image generation with model
    image = pipe(
        prompt,
        width=576,
        height=1024,
        num_inference_steps=20,
        guidance_scale=3.5
    ).images[0]
    
    # Convert the image to bytes
    img_byte_arr = BytesIO()
    image.save('new.png')