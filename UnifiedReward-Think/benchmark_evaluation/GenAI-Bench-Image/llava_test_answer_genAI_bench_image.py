from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
import itertools
from PIL import Image
import requests
import copy
import torch
import warnings
import os
from datasets import load_dataset, load_from_disk
import tqdm
import json

pretrained = "CodeGoat24/UnifiedReward-Think-7b"
dataset = load_dataset("TIGER-Lab/GenAI-Bench", 'video_generation')['test']
warnings.filterwarnings("ignore")

model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()


correct = 0
correct_tie = 0
num_all = 0
num_all_tie = 0
for i in tqdm.trange(len(dataset)):
    data = dataset[i]

    data['left_image'] = data['left_image'].resize((512, 512))
    data['right_image'] = data['right_image'].resize((512, 512))

    image_tensor = process_images([data['left_image'], data['right_image']], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

   
    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

    prompt = data['prompt']

    question = f'<image>\n <image>\nYou are given a text caption and two generated images based on that caption. Your task is to evaluate and compare these images based on two key criteria:\n1. Alignment with the Caption: Assess how well each image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of each image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nCompare both images using the above criteria and select the one that better aligns with the caption while exhibiting superior visual quality.\nProvide a clear conclusion such as \"Image 1 is better.\", \"Image 2 is better.\" and \"Both images are equally good.\"\nYour task is provided as follows:\nText Caption: [{prompt}]'
    
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [data['left_image'].size, data['right_image'].size]

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

    if 'left' in data['vote_type']:
        answer = 'Image 1 is better than Image 2'
    elif 'right' in data['vote_type']:
        answer = 'Image 2 is better than Image 1'
    else:
        answer = 'Both images are equally good'
        num_all_tie += 1


    num_all += 1
    if answer in output:
        correct += 1
        if answer == 'Both images are equally good':
            correct_tie += 1

accuracy = correct / num_all
print(f"Acc.: {correct} / {num_all} = {accuracy}")

accuracy_no_tie = (correct - correct_tie) / (num_all - num_all_tie)
print(f"Acc. w/o tie: {correct} - {correct_tie}/ {num_all} - {num_all_tie} = {accuracy_no_tie}")
