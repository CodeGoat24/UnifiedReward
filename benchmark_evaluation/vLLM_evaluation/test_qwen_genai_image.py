import random
import tqdm
from datasets import load_dataset
from io import BytesIO
import base64
from PIL import Image

from vllm_qwen.vllm_request import evaluate_batch

def _encode_image(img: Image.Image):
    img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


dataset = load_dataset("TIGER-Lab/GenAI-Bench", "image_generation")["test"]

input_data = []
for i in tqdm.trange(len(dataset)):
    data = dataset[i]

    if 'both' in data['vote_type'] or 'tie' in data['vote_type']:
        continue
    
    if 'left' in data['vote_type']:
        answer = 'Image 1 is better'
    elif 'right' in data['vote_type']:
        answer = 'Image 2 is better'
    else:
        continue
    
    if random.choices([True, False])[0]:
        left_image = data['right_image']
        right_image = data['left_image']
        if 'left' in data['vote_type']:
            data['vote_type'] = 'right'
        elif 'right' in data['vote_type']:
            data['vote_type'] = 'left'
    else:
        left_image = data['left_image']
        right_image = data['right_image']

    prompt = data['prompt']


    problem = f"You are given a text caption and two generated images based on that caption. Your task is to evaluate and compare these images based on two key criteria:\n1. Alignment with the Caption: Assess how well each image aligns with the provided caption. Consider the accuracy of depicted objects, their relationships, and attributes as described in the caption.\n2. Overall Image Quality: Examine the visual quality of each image, including clarity, detail preservation, color accuracy, and overall aesthetic appeal.\nCompare both images using the above criteria and select the one that better aligns with the caption while exhibiting superior visual quality.\nProvide a clear conclusion such as \"Image 1 is better.\", \"Image 2 is better.\" and \"Both images are equally good.\"\nYour task is provided as follows:\nText Caption: [{prompt}]"

    left_image = _encode_image(left_image)
    right_image = _encode_image(right_image)

    input_data.append({
        'problem': problem,
        'images': [left_image, right_image],
        'answer': answer
    })

output = evaluate_batch(input_data, "http://localhost:8080")

correct = 0
for item in output:
    if item['answer'] in item['model_output']:
        correct +=1

accuracy = correct / len(input_data)
print(f"Acc.: {correct} / {len(input_data)} = {accuracy}")