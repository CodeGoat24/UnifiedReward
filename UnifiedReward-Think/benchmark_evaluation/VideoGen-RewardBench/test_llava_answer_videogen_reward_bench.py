from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

from PIL import Image

import copy
import torch
import tqdm

import warnings

from datasets import load_dataset
import random

pretrained = "CodeGoat24/UnifiedReward-Think-7b"

def _load_video(video_path, num_video_frames, loader_fps, fps=None, frame_count=None):


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
    images = [img.resize((512, 512)) for img in images]
    image_sizes = [img.size for img in images]

    image_tensor = image_processor.preprocess(images, return_tensors="pt")["pixel_values"].cuda().bfloat16()

    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models
    
    question = (
        "<image>\n" * len(images) +
        "Imagine you are an expert tasked with evaluating AI-generated videos. You are provided with a text caption and two videos generated based on that caption. Your job is to assess and compare these videos based on the following two main factors:\n\n"
        "1. Caption Alignment: Evaluate how closely each video matches the description provided in the caption. Pay attention to the accuracy of objects depicted, their relationships, and any attributes described in the caption.\n\n"
        "2. Overall Video Quality: Look at the overall visual appeal of each video, considering clarity, the level of detail, color accuracy, and how aesthetically pleasing the video is.\n\n"
        "Using these factors, compare the two videos and determine which one better reflects the caption and exhibits better visual quality.\n\n"
        "Give your final judgment, such as 'Video 1 is better,' 'Video 2 is better,' or 'Both videos are equally good.'\n\n"
        "Your task is as follows:\n"
        f"Text Caption: [{prompt}]\n"
    )

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
