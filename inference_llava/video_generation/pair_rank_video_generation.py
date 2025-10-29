from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
import tqdm
import sys
import warnings
import os
from datasets import load_dataset, load_from_disk

pretrained = "CodeGoat24/UnifiedReward-7b"

def _load_video(video_path, num_video_frames, loader_fps, fps=None, frame_count=None):
        from torchvision import transforms

        from llava.mm_utils import opencv_extract_frames

        try:
            pil_imgs, frames_loaded = opencv_extract_frames(video_path, num_video_frames, loader_fps, fps, frame_count)
        except Exception as e:
            video_loading_succeed = False
            print(f"bad data path {video_path}")
            print(f"[DEBUG] Error processing {video_path}: {e}")
            empty_num_video_frames = int(random.uniform(2, num_video_frames))
            pil_imgs = [Image.new("RGB", (448, 448), (0, 0, 0))] * empty_num_video_frames
            frames_loaded = 0

        return pil_imgs, frames_loaded

warnings.filterwarnings("ignore")

model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()


prompt = ""

video_path_1 = "/path/to/video1"
video_path_2 = "/path/to/video2"


num_video_frames = 8
loader_fps = 0.0
fps = None
frame_count = None

images, frames_loaded = _load_video(
    video_path_1, num_video_frames, loader_fps, fps=fps, frame_count=frame_count
)
images_, frames_loaded = _load_video(
    video_path_2, num_video_frames, loader_fps, fps=fps, frame_count=frame_count
)

images.extend(images_)
image_sizes = []
for img in images:
    img.resize((512, 512))
    image_sizes.append(img.size)

image_tensor = image_processor.preprocess(images, return_tensors="pt")["pixel_values"].cuda().bfloat16()

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

question = '<image>\n'*len(images) + f'Suppose you are an expert in judging and evaluating the quality of AI-generated videos. You are given a text caption and the frames of two generated videos based on that caption. Your task is to evaluate and compare two videos based on two key criteria:\n1. Alignment with the Caption: Assess how well each video aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Video Quality: Examine the visual quality of each video, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nCompare both videos using the above criteria and select the one that better aligns with the caption while exhibiting superior visual quality.\nProvide a clear conclusion such as \"Video 1 is better than Video 2.\", \"Video 2 is better than Video 1.\" and \"Both videos are equally good.\"\nYour task is provided as follows:\nText Caption: [{prompt}]'


conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

with torch.cuda.amp.autocast():
    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)

print(text_outputs[0])


