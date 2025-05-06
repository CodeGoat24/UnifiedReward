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

pretrained = "CodeGoat24/UnifiedReward-Think-7b"

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


def get_results(video_path_1, video_path_2, prompt):
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
    
    question = '<image>\n'*len(images) + f"Given a caption and two videos generated based on this caption, please analyze in detail the two provided videos. Evaluate them on various dimensions such as semantic consistency (how closely the video content aligns with the caption), temporal coherence (smoothness and logical flow of motion across frames), authenticity (realism and attention to detail), and any other factors you deem relevant. For each evaluation dimension, provide a score between 1-10 for both videos (e.g., Video 1: 8/10, Video 2: 6/10) and provide a concise rationale for the score. Calculate the total score for each video by summing all dimension scores. Use a chain-of-thought process to detail your reasoning steps, and enclose all your detailed reasoning within <think> and </think> tags. Then, in the <answer> tag, output exactly one of the following strings: 'Video 1 is better' or 'Video 2 is better' based on the total scores. No additional text is allowed in the <answer> section.\n\nExample output format:\n<think>\n1. Semantic consistency: Video 1 (9/10) - ...; Video 2 (7/10) - ...\n2. Temporal coherence: Video 1 (8/10) - ...; Video 2 (6/10) - ...\n3. Authenticity: Video 1 (7/10) - ...; Video 2 (5/10) - ...\n[Additional dimensions if any]: Video 2 (8/10) - ...; Video 1 (6/10) - ...\nTotal score:\nVideo 1: 9+8+7+6=30\nVideo 2: 7+6+5+8=26\n</think>\n<answer>Video 1 is better</answer>\n**Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given videos.**\n\nYour task is provided as follows:\nText Caption: [{prompt}]" 


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
        output = text_outputs[0]

    return output

import json
dataset = load_dataset("KwaiVGI/VideoGen-RewardBench")['eval']

warnings.filterwarnings("ignore")

model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()

correct = 0
correct_tie = 0
num_tie = 0
num_all = 0
for i in tqdm.trange(len(dataset)):

    data = dataset[i]
    answer = data['Overall']
    if answer == 'same':
        num_tie += 1
    prompt = data['prompt']
    output = get_results(data['path_A'], data['path_B'], prompt)

    if output in 'Video 1 is better':
        pred = 'A'
    elif output in 'Video 2 is better':
        pred = 'B'
    else:
        pred = 'same'

    if pred == answer:
        correct += 1
        if answer == 'same':
            correct_tie += 1
    
    num_all += 1

print(correct)
accuracy = correct / num_all
print(f"Acc.: {correct} / {num_all} = {accuracy}")

accuracy_no_tie = (correct - correct_tie) / (num_all - num_tie)
print(f"Acc.: {correct} - {correct_tie} / {num_all} - {num_tie} = {accuracy_no_tie}")