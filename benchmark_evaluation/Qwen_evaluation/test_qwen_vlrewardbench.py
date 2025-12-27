
import random
import torch
import tqdm

import warnings

from datasets import load_dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

model_path = "CodeGoat24/UnifiedReward-qwen-7b"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map={"": 'cuda:0'}
)
processor = AutoProcessor.from_pretrained(model_path)

dataset = load_dataset("MMInstruction/VL-RewardBench")['test']

group_correct = {'general': 0, 'hallucination': 0, 'reasoning': 0}
group_total = {'general': 0, 'hallucination': 0, 'reasoning': 0}
group_mapping = {
    "vlfeedback": "general",
    "povid": "hallucination",
    "reasoning_tasks": "reasoning",
    "rlhf-v": "hallucination",
    "rlaif-v": "hallucination",
    "wildvision-battle": "general"
}

correct = 0
random.seed(0)


for i in tqdm.trange(len(dataset)):
    data = dataset[i]
    image = data['image']

    if data["human_ranking"][0] == 0:
        if random.random() < 0.5:
            R1, R2, answer = data["response"][0], data["response"][1], "Answer 1 is better"
        else:
            R1, R2, answer = data["response"][1], data["response"][0], "Answer 2 is better"
    else:
        if random.random() < 0.5:
            R1, R2, answer = data["response"][0], data["response"][1], "Answer 2 is better"
        else:
            R1, R2, answer = data["response"][1], data["response"][0], "Answer 1 is better"

    Query = data["query"]


    prompt_text = (
        "You are given an image and a question related to it. Your job is to evaluate the two responses based on these five factors:\n\n"
        "1. Accuracy of Object Descriptions: Review how accurately the objects are described in the responses, ensuring they match those in the ground truth. Be mindful of irrelevant or incorrect objects being mentioned.\n\n"
        "2. Relationship Between Objects: Check if the response properly describes how the objects relate to each other, reflecting their actual positions or interactions, as seen in the image.\n\n"
        "3. Description of Attributes: Assess how well the response captures the attributes (e.g., size, color, shape) of the objects in the image, in line with the ground truth.\n\n"
        "4. Helpfulness: Consider whether the response offers useful information that enhances the understanding of the image. Does it add context or provide extra insights? Also, evaluate whether it follows the instructions given in the prompt.\n\n"
        "5. Ethical Concerns: Review the response to ensure it avoids sensitive, harmful, or inappropriate content. The response should be fair, respect privacy, and be free of bias or offensive material.\n\n"
        "After evaluating both answers, determine which one is better based on these factors and clearly state your decision, such as 'Answer 1 is better' or 'Answer 2 is better.'\n\n"
        f"Question: {Query}\n"
        f"Answer 1: {R1}\n"
        f"Answer 2: {R2}\n"
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    chat_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[chat_input],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
        padding=True
    ).to("cuda")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output = processor.batch_decode(generated_trimmed, skip_special_tokens=True)[0]

    id_value = data["id"]
    split_index = min((id_value.find('_'), id_value.find('-')), key=lambda x: x if x != -1 else float('inf'))
    id_prefix = id_value[:split_index] if split_index != -1 else id_value
    dtype = {
        "RLAIF": "rlaif-v",
        "RLHF": "rlhf-v",
        "mathverse": "reasoning_tasks",
        "mmmu": "reasoning_tasks",
        "wildvision": "wildvision-battle"
    }.get(id_prefix, "vlfeedback")

    group = group_mapping[dtype]
    group_total[group] += 1

    if answer in output:
        correct += 1
        group_correct[group] += 1

accuracy = correct / len(dataset)
print(f"Acc.: {correct} / {len(dataset)} = {accuracy}")

task_list =  ['reasoning', 'hallucination', 'general']

macro_average = sum(group_correct[k]/group_total[k] for k in task_list) / 3

reasoning_acc = group_correct['reasoning']/group_total['reasoning']
hallucination_acc = group_correct['hallucination']/group_total['hallucination']
general_acc = group_correct['general']/group_total['general']

print(f'reasoning: {reasoning_acc}')
print(f'hallucination: {hallucination_acc}')
print(f'general: {general_acc}')
print(f'macro: {macro_average}')
