from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates


import copy
import torch
import tqdm
import warnings

from datasets import load_dataset
import random
warnings.filterwarnings("ignore")

pretrained = "CodeGoat24/UnifiedReward-7b-v1.5"
model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args
model.eval()



dataset = load_dataset("MMInstruction/VL-RewardBench")['test']

group_correct = {}
group_total = {}
group_mapping = {
        "vlfeedback": "general",
        "povid": "hallucination",
        "reasoning_tasks": "reasoning",
        "rlhf-v": "hallucination",
        "rlaif-v": "hallucination",
        "wildvision-battle": "general"
    }
random.seed(0)
group_total['general'] = 0
group_total['hallucination'] = 0
group_total['reasoning'] = 0
group_correct['general'] = 0
group_correct['hallucination'] = 0
group_correct['reasoning'] = 0
correct = 0
for i in tqdm.trange(len(dataset)):
    data = dataset[i]

    image = data['image']
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

    if data['human_ranking'][0] == 0:
        if random.choices([True, False])[0]:
            R1 = data['response'][0]
            R2 = data['response'][1]
            answer = 'Answer 1 is better'
        else:
            R1 = data['response'][1]
            R2 = data['response'][0]
            answer = 'Answer 2 is better'
    else:
        if random.choices([True, False])[0]:
            R1 = data['response'][0]
            R2 = data['response'][1]
            answer = 'Answer 2 is better'
        else:
            R1 = data['response'][1]
            R2 = data['response'][0]
            answer = 'Answer 1 is better'

    Query = data['query']
    question = (
        "<image>\nYou are given an image and a question related to it. Your job is to evaluate the two responses based on these five factors:\n\n"
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
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
    image_sizes = [image.size]

    cont = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=False,
        temperature=0,
        max_new_tokens=4096,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
    output = text_outputs[0]

    id_value = data['id']
    split_index = min((id_value.find('_'), id_value.find('-')), key=lambda x: x if x != -1 else float('inf'))
    if split_index != -1:
        id_prefix = id_value[:split_index]
    else:
        id_prefix = id_value
    
    if id_prefix=="RLAIF":
        dtype = "rlaif-v"
    elif id_prefix=="RLHF":
        dtype = "rlhf-v"
    elif id_prefix=="mathverse" or id_prefix=="mmmu":
        dtype = "reasoning_tasks"
    elif id_prefix=="wildvision":
        dtype = "wildvision-battle"
    else:
        dtype = "vlfeedback"
    
    group_total[group_mapping[dtype]] += 1

    if answer in output:
        correct += 1
        group_correct[group_mapping[dtype]] += 1
        
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
