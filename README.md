# Low GPU Flux-dev scripts

The purpose of this is to have a collection of different scripts for efficient and low-gpu flux-dev.1 inference model.

This repository is divided based on its gpu usage. The results considerates for all the cases when denoising:

- Inference Steps = 20
- Resolution = 1024 x 1024
- Guidance Scale = 5.5

Inference time could vary if those conditions are changed.

## Requirements

- NVIDIA GPU
- 16 GB RAM
- Windows/Ubuntu

## Testing scripts

By using the ***requirements.txt*** file, it's guaranteed that all the scripts here will be able to run properly.


- Ubuntu
    ```bash
    python3 -m venv venv
    source venv\bin\activate
    pip install -r requirements.txt
    ```

- Windows
    ```bash
    python -m venv venv
    venv\Scripts\activate.bat
    pip install -r requirements.txt
    ```


## 8 GB GPU (Approx)

The scripts listed here has been using between 8 and 8.5 GB of VRAM.

- ### low-gpu-decode
    
    It's basically loads a pipeline for text encoding, then deletes it and loads a pipeline for inference using only the encoded prompt.

## 14 GB GPU (Approx)

The scripts listed here has been using between 14.3 and 14.6 GB of VRAM.

- ### low-gpu-no-decode-linear-pipeline

    The main difference against the low-gpu-decode approach (8 GB) is that it makes no encoding of the prompt, allowing for faster denoising process but diminishing the efficiency of gpu usage.

- ### low-gpu-no-decode-multipipeline

    Against its twin (low-gpu-no-decode-linear-pipeline) this script loads 2 pipeline, one for inference and one for decoding. It allows to load both pipelines once and then just making inference based on the new prompts.

## 38 GB GPU (Approx)

The scripts listed here has been using 38 or more GB of VRAM.

- ### normal-gpu-inference

    This is a common and not optimized inference pipeling on the gpu-usage, however the inference time is quite optimal.

## Sources

The scripts listed here contains slightly variations of scripts found in different repositories.

- https://gist.github.com/ariG23498/948c263116886b3aafae95e69ac3336a
- https://gist.github.com/sayakpaul/23862a2e7f5ab73dfdcc513751289bea

Also, the diffusers repository branch for this scripts is taken from the quantization branch.

- https://github.com/huggingface/diffusers/tree/add-quantization-ci
