from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


import torch
import tqdm

import random



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
