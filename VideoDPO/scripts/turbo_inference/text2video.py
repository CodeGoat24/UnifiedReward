# Adapted from https://github.com/luosiallen/latent-consistency-model
from __future__ import annotations

import argparse
import os
import random
import time
from omegaconf import OmegaConf

import numpy as np
import json

try:
    import intel_extension_for_pytorch as ipex
except:
    pass

from lvdm.models.turbo_utils.lora import collapse_lora, monkeypatch_remove_lora
from lvdm.models.turbo_utils.lora_handler import LoraHandler
from utils.common_utils import load_model_checkpoint
from utils.common_utils import instantiate_from_config
from lvdm.models.turbo_utils.t2v_turbo_scheduler import T2VTurboScheduler
from lvdm.models.turbo_utils.t2v_turbo_pipeline import T2VTurboVC2Pipeline

import torch
import torchvision

from concurrent.futures import ThreadPoolExecutor
import uuid

DESCRIPTION = """# T2V-Turbo 🚀
We provide T2V-Turbo (VC2) distilled from [VideoCrafter2](https://ailab-cvc.github.io/videocrafter2/) with the reward feedback from [HPSv2.1](https://github.com/tgxs002/HPSv2/tree/master) and [InternVid2 Stage 2 Model](https://huggingface.co/OpenGVLab/InternVideo2-Stage2_1B-224p-f4).

You can download the the models from [here](https://huggingface.co/jiachenli-ucsb/T2V-Turbo-VC2). Check out our [Project page](https://t2v-turbo.github.io) 😄
"""
if torch.cuda.is_available():
    DESCRIPTION += "\n<p>Running on CUDA 😀</p>"
elif hasattr(torch, "xpu") and torch.xpu.is_available():
    DESCRIPTION += "\n<p>Running on XPU 🤓</p>"
else:
    DESCRIPTION += "\n<p>Running on CPU 🥶 This demo does not work on CPU.</p>"

MAX_SEED = np.iinfo(np.int32).max
CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv("CACHE_EXAMPLES") == "1"
USE_TORCH_COMPILE = os.getenv("USE_TORCH_COMPILE") == "1"


"""
Operation System Options:
    If you are using MacOS, please set the following (device="mps") ;
    If you are using Linux & Windows with Nvidia GPU, please set the device="cuda";
    If you are using Linux & Windows with Intel Arc GPU, please set the device="xpu";
"""
# device = "mps"    # MacOS
# device = "xpu"    # Intel Arc GPU
device = "cuda"  # Linux & Windows


"""
   DTYPE Options:
      To reduce GPU memory you can set "DTYPE=torch.float16",
      but image quality might be compromised
"""
DTYPE = (
    torch.float16
)  # torch.float16 works as well, but pictures seem to be a bit worse


def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed


def save_video(vid_tensor, output_path, fps=16):

    # Convert the video tensor from [C, T, H, W] to [T, C, H, W]
    video = vid_tensor.detach().cpu()
    video = torch.clamp(video.float(), -1.0, 1.0)
    video = video.permute(1, 0, 2, 3)  # t,c,h,w
    video = (video + 1.0) / 2.0
    video = (video * 255).to(torch.uint8).permute(0, 2, 3, 1)

    # Save the video using torchvision.io.write_video
    torchvision.io.write_video(
        output_path, video, fps=fps, video_codec="h264", options={"crf": "10"}
    )
    return output_path

def save_videos(video_array, output_paths, fps=16):
    paths = []
    with ThreadPoolExecutor() as executor:
        paths = list(
            executor.map(
                save_video,
                video_array,
                output_paths,
                [fps] * len(video_array),
            )
        )
    return paths



# this function comes from VC2 scripts/evaluation/funcs.py
def load_prompts(prompt_file):
    with open(prompt_file, 'r') as file:
        dataset = json.load(file)

    prompt_list = []
    for item in dataset:
        prompt_list.append(item['prompt'])

    return prompt_list

def generate_videos_from_prompts(prompts_file, output_dir, unet_dir, base_model_dir):

    os.makedirs(output_dir, exist_ok=True)
    seed = 123
    guidance_scale = 7.5
    num_inference_steps = 4
    num_frames = 16
    fps = 8
    randomize_seed = True
    param_dtype = "torch.float16"
    
    config = OmegaConf.load("configs/inference_t2v_512_v2.0.yaml")
    model_config = config.pop("model", OmegaConf.create())
    pretrained_t2v = instantiate_from_config(model_config)
    pretrained_t2v = load_model_checkpoint(pretrained_t2v, base_model_dir)

    unet_config = model_config["params"]["unet_config"]
    unet_config["params"]["time_cond_proj_dim"] = 256
    unet = instantiate_from_config(unet_config)

    unet.load_state_dict(
        pretrained_t2v.model.diffusion_model.state_dict(), strict=False
    )

    use_unet_lora = True
    lora_manager = LoraHandler(
        version="cloneofsimo",
        use_unet_lora=use_unet_lora,
        save_for_webui=True,
        unet_replace_modules=["UNetModel"],
    )
    lora_manager.add_lora_to_model(
        use_unet_lora,
        unet,
        lora_manager.unet_replace_modules,
        lora_path=unet_dir,
        dropout=0.1,
        r=64,
    )
    unet.eval()
    collapse_lora(unet, lora_manager.unet_replace_modules)
    monkeypatch_remove_lora(unet)

    pretrained_t2v.model.diffusion_model = unet
    scheduler = T2VTurboScheduler(
        linear_start=model_config["params"]["linear_start"],
        linear_end=model_config["params"]["linear_end"],
    )
    
    pipeline = T2VTurboVC2Pipeline(pretrained_t2v, scheduler, model_config)
    pipeline.to(device)
    pipeline.to(
        # torch_device="cuda", # test cuda usage
        torch_dtype=torch.float16 if param_dtype == "torch.float16" else torch.float32,
    )
    
    prompts = load_prompts(prompts_file)
    N_VIDEOS_PER_PROMPT=10

    data_list = []

    from tqdm import tqdm
    for idx, prompt in tqdm(enumerate(prompts),total=len(prompts)):
        seed = randomize_seed_fn(seed, randomize_seed)
        torch.manual_seed(seed)
        output_dir_vid = os.path.join(output_dir, 'videos')
        output_paths = [os.path.join(output_dir_vid,f"{idx:06}-{sample_idx}.mp4") for sample_idx in range(N_VIDEOS_PER_PROMPT)]
        if all([os.path.exists(path) for path in output_paths]):
            continue
            
        results_array = []
        for i in range(N_VIDEOS_PER_PROMPT // 2):
            result = pipeline(
                prompt=prompt,
                frames=num_frames,
                fps=fps,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                num_videos_per_prompt=2,
            )

            results_array.extend([result[i] for i in range(result.shape[0])])

        save_videos(results_array, output_paths , fps)

        data_list.append(
            {
                'idx': f"{idx:06}",
                'caption': prompt,
                'videos': output_paths
            }
        )
        with open(os.path.join(output_dir, 'data.json'), 'w') as output_file:
            json.dump(data_list, output_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch video generation from prompts.")
    parser.add_argument(
        "--prompts_file",
        type=str,
        required=True,
        help="Path to the file containing prompts.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save generated videos.",
    )
    parser.add_argument(
        "--unet_dir",
        type=str,
        required=True,
        help="Directory of the UNet model.",
    )
    parser.add_argument(
        "--base_model_dir",
        type=str,
        required=True,
        help="Directory of the VideoCrafter2 checkpoint.",
    )

    args = parser.parse_args()
    generate_videos_from_prompts(args.prompts_file, args.output_dir, args.unet_dir, args.base_model_dir)