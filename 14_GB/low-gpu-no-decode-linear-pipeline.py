from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel
import torch
import gc


def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()


def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024


flush()

ckpt_id = "black-forest-labs/FLUX.1-dev"
ckpt_4bit_id = "sayakpaul/flux.1-dev-nf4-pkg"


text_encoder_2_4bit = T5EncoderModel.from_pretrained(
    ckpt_4bit_id,
    subfolder="text_encoder_2",
)

transformer_4bit = FluxTransformer2DModel.from_pretrained(ckpt_4bit_id, subfolder="transformer")
pipeline = FluxPipeline.from_pretrained(
    ckpt_id,
    text_encoder_2=text_encoder_2_4bit,
    transformer=transformer_4bit,
    torch_dtype=torch.float16
)
pipeline.enable_model_cpu_offload()

flush()

pipeline.load_lora_weights("nissan_kicks_nt.safetensors", dtype=torch.float16)
pipeline.enable_model_cpu_offload()

pipeline = pipeline.to("cpu")
flush()


prompt = "Una Nissan Kicks ambientada en una ciudad de dulces"
print("Running denoising.")
height, width = 1024, 576
images = pipeline(
    prompt,
    num_inference_steps=20,
    guidance_scale=5.5,
    height=height,
    width=width,
    output_type="pil",
).images
images[0].save("output.png")