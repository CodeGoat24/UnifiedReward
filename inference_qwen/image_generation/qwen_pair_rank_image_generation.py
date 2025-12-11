from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_from_disk
from PIL import Image
import torch
import tqdm
import os
import random
import json


model_path = 'CodeGoat24/UnifiedReward-qwen-7b'
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map={"": 'cuda:0'}
)
processor = AutoProcessor.from_pretrained(model_path)

image_path_1 = '/path/to/image1'
image_path_2 = '/path/to/image2'

prompt = "" # prompt of given images

image1 = Image.open(image_path_1)
image2 = Image.open(image_path_2)


messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image1},
            {"type": "image", "image": image2},
            {
                "type": "text",
                "text": f"You are given a text caption and two generated images based on that caption. Your task is to evaluate and compare these images based on two key criteria:\n1. Alignment with the Caption: Assess how well each image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of each image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nCompare both images using the above criteria and select the one that better aligns with the caption while exhibiting superior visual quality.\nProvide a clear conclusion such as \"Image 1 is better.\", \"Image 2 is better.\" and \"Both images are equally good.\"\nYour task is provided as follows:\nText Caption: [{prompt}]"
            },
        ],
    }
]


text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(model.device)


with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=512)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(output_text)
