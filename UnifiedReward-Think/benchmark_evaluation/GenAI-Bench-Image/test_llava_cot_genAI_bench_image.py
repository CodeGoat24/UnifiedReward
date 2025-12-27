from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates



import copy
import torch
import warnings

from datasets import load_dataset
import tqdm


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

    question = f"<image>\n <image>\nGiven a caption and two images generated based on this caption, please analyze in detail the two provided images. Evaluate them on various dimensions such as semantic consistency (how closely the image content aligns with the caption), aesthetics (composition, color usage, artistic expression), authenticity (realism and attention to detail), and any other factors you deem relevant. For each evaluation dimension, provide a score between 1-10 for both images (e.g., Image 1: 8/10, Image 2: 6/10) and provide a concise rationale for the score. Calculate the total score for each image by summing all dimension scores. Use a chain-of-thought process to detail your reasoning steps, and enclose all your detailed reasoning within <think> and </think> tags. Then, in the <answer> tag, output exactly one of the following strings: \'Image 1 is better\' or \'Image 2 is better\' based on the total scores. No additional text is allowed in the <answer> section.\n\nExample output format:\n<think>\n1. Semantic consistency: Image 1 (9/10) - ...; Image 2 (7/10) - ...\n2. Aesthetics: Image 2 (8/10) - ...; Image 1 (8/10) - ...\n3. Authenticity: Image 1 (8/10) - ...; Image 2 (5/10) - ...\n[Additional dimensions if any]: Image 2 (8/10) - ...; Image 1 (6/10) - ...\nTotal score:\nImage 1: 9+8+8+6=31\nImage 2: 7+8+5+8=28\n</think>\n<answer>Image 1 is better</answer>\n**Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given images.**\n\nYour task is provided as follows:\nText Caption: [{prompt}]"

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
