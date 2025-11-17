from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
import tqdm
import warnings
import os
from datasets import load_dataset

warnings.filterwarnings("ignore")
pretrained = "CodeGoat24/UnifiedReward-7b"

model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()

data_path = '/path/to/data.json'
# The data should adhere to the following structure:
# [
#     {
#       "prompt": "",
#       "images": ["/path/to/frame1", "/path/to/frame2", ...],
#       "responses": ["response1", "response2"],
#       "answer": "Response 1 is better than Response 2." or "Response 2 is better than Response 1.",
#     },
#     ...
# ]
image_folder = '/path/to/images'

import json
with open(data_path, 'r') as file:
    dataset = json.load(file)

correct = 0
for i in tqdm.trange(len(dataset)):
    data = dataset[i]

    image_list = []
    image_sizes = []
    for img in data['images']:
        image = Image.open(os.path.join(image_folder, img)).resize((512, 512))
        image_list.append(image)
        image_sizes.append(image.size)

    image_tensor = image_processor.preprocess(image_list, return_tensors="pt")["pixel_values"].cuda().bfloat16()

    conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

    Query = data['prompt']
    R1 = data['responses'][0]
    R2 = data['responses'][1]

    question = '<image>\n'*8+f'Please analyze the following video and question, then determine which of the two provided answers is better\nQuestion: {Query}\nAnswer 1: {R1}\nAnswer 2: {R2}\nPlease evaluate both answers based on the following criteria:\n1. Accuracy: How well does the answer align with the visual information in the video?\n2. Completeness: Does the answer fully address all aspects of the question?\n3. Clarity: Is the answer easy to understand and well-articulated?\n4. Relevance: Does the answer directly relate to the question and the video?\nAfter your evaluation, please:\n1. Explain your reasoning for each criterion.\n2. Provide an overall judgment on which answer is better (Answer 1 or Answer 2). For example: \"Overall Judgment: Answer 1 is better than Answer 2.\"\nYour response should be structured and detailed, demonstrating your understanding of both the visual and textual elements of the task. You have to choose the better of the two answers and can\'t say that both answers are equally good.'


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
            max_new_tokens=4096,
        )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)

    if 'Overall Judgment' in text_outputs[0]:
        output = text_outputs[0].split('Overall Judgment')[-1].strip().split('.')[0].strip()
    else:
        output = text_outputs[0]

    answer = data['answer']

    if output == answer or answer in output:
        correct += 1
        

accuracy = correct / len(dataset)
print(f"Acc.: {correct} / {len(dataset)} = {accuracy}")
