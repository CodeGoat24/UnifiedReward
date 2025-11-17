from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from datasets import load_from_disk
from PIL import Image
import torch
import tqdm
import os
import random
import json


model_path = 'CodeGoat24/UnifiedReward-Think-qwen-7b'
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map={"": 'cuda:0'}
)
processor = AutoProcessor.from_pretrained(model_path)


dataset = load_dataset("TIGER-Lab/GenAI-Bench", 'image_generation')['test']

correct = 0
correct_tie = 0
num_all = 0
num_all_tie = 0

for i in tqdm.trange(len(dataset)):
    data = dataset[i]

    if 'both' in data['vote_type'] or 'tie' in data['vote_type']:
        num_all_tie += 1
        num_all += 1
        continue

    if random.choices([True, False])[0]:
        left_image = data['right_image'].resize((512, 512))
        right_image = data['left_image'].resize((512, 512))
        if 'left' in data['vote_type']:
            data['vote_type'] = 'right'
        elif 'right' in data['vote_type']:
            data['vote_type'] = 'left'
    else:
        left_image = data['left_image'].resize((512, 512))
        right_image = data['right_image'].resize((512, 512))

    prompt = data['prompt']


    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": left_image},
                {"type": "image", "image": right_image},
                {
                    "type": "text",
                    "text": f"Given a caption and two images generated based on this caption, please analyze in detail the two provided images. Evaluate them on various dimensions such as semantic consistency (how closely the image content aligns with the caption), aesthetics (composition, color usage, artistic expression), authenticity (realism and attention to detail), and any other factors you deem relevant. For each evaluation dimension, provide a score between 1-10 for both images (e.g., Image 1: 8/10, Image 2: 6/10) and provide a concise rationale for the score. Calculate the total score for each image by summing all dimension scores. Use a chain-of-thought process to detail your reasoning steps, and enclose all your detailed reasoning within <think> and </think> tags. Then, in the <answer> tag, output exactly one of the following strings: \'Image 1 is better\' or \'Image 2 is better\' based on the total scores. No additional text is allowed in the <answer> section.\n\nExample output format:\n<think>\n1. Semantic consistency: Image 1 (9/10) - ...; Image 2 (7/10) - ...\n2. Aesthetics: Image 2 (8/10) - ...; Image 1 (8/10) - ...\n3. Authenticity: Image 1 (8/10) - ...; Image 2 (5/10) - ...\n[Additional dimensions if any]: Image 2 (8/10) - ...; Image 1 (6/10) - ...\nTotal score:\nImage 1: 9+8+8+6=31\nImage 2: 7+8+5+8=28\n</think>\n<answer>Image 1 is better</answer>\n**Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given images.**\n\nYour task is provided as follows:\nText Caption: [{prompt}]"
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

 
    if 'left' in data['vote_type']:
        answer = 'Image 1 is better'
    elif 'right' in data['vote_type']:
        answer = 'Image 2 is better'
    else:
        answer = 'Both images are equally good'
        num_all_tie += 1

    num_all += 1
    if answer in output_text:
        correct += 1
        if data['vote_type'] == 'tie':
            correct_tie += 1


print(f"Acc.: {correct} / {num_all} = {correct / num_all:.4f}")
print(f"Acc. w/o tie: ({correct} - {correct_tie}) / ({num_all} - {num_all_tie}) = {(correct - correct_tie) / (num_all - num_all_tie):.4f}")
