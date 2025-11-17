import torch
import os
import json
import random
from PIL import Image
from tqdm import trange
import warnings
from datasets import load_dataset

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

model_path = 'CodeGoat24/UnifiedReward-Think-qwen-7b'
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map={"": 'cuda:0'}
)
processor = AutoProcessor.from_pretrained(model_path)

def get_results(video_path_1, video_path_2, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path_1},
                {"type": "video", "video": video_path_2},
                {"type": "text", "text": f"Given a caption and two videos generated based on this caption, please analyze in detail the two provided videos. Evaluate them on various dimensions such as semantic consistency (how closely the video content aligns with the caption), temporal coherence (smoothness and logical flow of motion across frames), authenticity (realism and attention to detail), and any other factors you deem relevant. For each evaluation dimension, provide a score between 1-10 for both videos (e.g., Video 1: 8/10, Video 2: 6/10) and provide a concise rationale for the score. Calculate the total score for each video by summing all dimension scores. Use a chain-of-thought process to detail your reasoning steps, and enclose all your detailed reasoning within <think> and </think> tags. Then, in the <answer> tag, output exactly one of the following strings: 'Video 1 is better' or 'Video 2 is better' based on the total scores. No additional text is allowed in the <answer> section.\n\nExample output format:\n<think>\n1. Semantic consistency: Video 1 (9/10) - ...; Video 2 (7/10) - ...\n2. Temporal coherence: Video 1 (8/10) - ...; Video 2 (6/10) - ...\n3. Authenticity: Video 1 (7/10) - ...; Video 2 (5/10) - ...\n[Additional dimensions if any]: Video 2 (8/10) - ...; Video 1 (6/10) - ...\nTotal score:\nVideo 1: 9+8+7+6=30\nVideo 2: 7+6+5+8=26\n</think>\n<answer>Video 1 is better</answer>\n**Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given videos.**\n\nYour task is provided as follows:\nText Caption: [{prompt}]"}
            ]
        }
    ]


    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt", **video_kwargs).to(model.device)


    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

    return output_text


dataset = load_dataset("TIGER-Lab/GenAI-Bench", 'video_generation')['test']


correct = 0
correct_tie = 0
num_tie = 0
num_all = 0

for i in trange(len(dataset)):
    data = dataset[i]
    answer = 'A' if 'left' in data['vote_type'] else 'B'
    if 'both' in data['vote_type'] or 'tie' in data['vote_type']:
        answer = 'same'
        num_tie += 1
        num_all += 1
        continue
    prompt = data['prompt']

    if random.choices([True, False])[0]:
        left_video = data['right_video']
        right_video = data['left_video']
        if answer == 'A':
            answer = 'B'
        elif answer == 'B':
            answer = 'A'
    else:
        left_video = data['left_video']
        right_video = data['right_video']

    output = get_results(left_video, right_video, prompt)

    if 'Video 1 is better' in output:
        pred = 'A'
    elif 'Video 2 is better' in output:
        pred = 'B'
    else:
        pred = 'same'

    if pred == answer:
        correct += 1
        if answer == 'same':
            correct_tie += 1

    num_all += 1


accuracy = correct / num_all
accuracy_no_tie = (correct - correct_tie) / (num_all - num_tie)
print(f"Acc.: {correct} / {num_all} = {accuracy:.4f}")
print(f"Acc. w/o tie: ({correct} - {correct_tie}) / ({num_all} - {num_tie}) = {accuracy_no_tie:.4f}")
