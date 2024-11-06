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

def encode_prompt(prompt: str):
    with torch.no_grad():
        print("Encoding prompt.")
        prompt_embeds, pooled_prompt_embeds, _ = pipeline_encoder.encode_prompt(
            prompt=prompt, prompt_2=None, max_sequence_length=256
        )
    return prompt_embeds, pooled_prompt_embeds

flush()

ckpt_id = "black-forest-labs/FLUX.1-dev"
ckpt_4bit_id = "sayakpaul/flux.1-dev-nf4-pkg"

text_encoder_2_4bit = T5EncoderModel.from_pretrained(
    ckpt_4bit_id,
    subfolder="text_encoder_2",
)

pipeline_encoder = FluxPipeline.from_pretrained(
    ckpt_id,
    text_encoder_2=text_encoder_2_4bit,
    transformer=None,
    vae=None,
    torch_dtype=torch.float16,
)
pipeline_encoder.enable_model_cpu_offload()

transformer_4bit = FluxTransformer2DModel.from_pretrained(ckpt_4bit_id, subfolder="transformer")
pipeline_inference = FluxPipeline.from_pretrained(
    ckpt_id,
    text_encoder=None,
    text_encoder_2=None,
    tokenizer=None,
    tokenizer_2=None,
    transformer=transformer_4bit,
    torch_dtype=torch.float16,
)
pipeline_inference.load_lora_weights("nissan_kicks_nt.safetensors", dtype=torch.float16)
pipeline_inference.enable_model_cpu_offload()
pipeline_inference = pipeline_inference.to("cpu") 
flush()

prompt = "Una Nissan Kicks ambientada en una ciudad de dulces"
prompt_embeds, pooled_prompt_embeds = encode_prompt(prompt)

prompt_embeds = prompt_embeds.to("cpu")
pooled_prompt_embeds = pooled_prompt_embeds.to("cpu")

print("Running denoising.")
height, width = 1024, 576
image = pipeline_inference(
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    num_inference_steps=20,
    guidance_scale=5.5,
    height=height,
    width=width,
    output_type="pil",
).images
image[0].save("output.png")