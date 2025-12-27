# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

from PIL import Image

import copy
import torch

import warnings



warnings.filterwarnings("ignore")
pretrained = "CodeGoat24/UnifiedReward-Think-7b"

model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()

image_path = '/path/to/image'

Query = '' # Question

R1 = '' # Response1
R2 = '' # Response2

image = Image.open(image_path)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "qwen_1_5"

question = ("<image>\nGiven a question and a reference image, please analyze in detail the two provided answers (Answer 1 and Answer 2). " \
            "Evaluate them based on the following three core dimensions:\n" \
            "1. Semantic accuracy: How well the answer reflects the visual content of the image\n" \
            "2. Correctness: Whether the answer is logically and factually correct\n" \
            "3. Clarity: Whether the answer is clearly and fluently expressed\n" \
            "You may also consider additional dimensions if you find them relevant (e.g., reasoning ability, attention to detail, multimodal grounding, etc.). " \
            "For each dimension, provide a score from 1 to 10 for both answers, and briefly explain your reasoning. " \
            "Then, compute the total score for each answer by explicitly adding the scores for all dimensions and showing the full calculation. " \
            "Enclose your full reasoning within <think> and </think> tags. " \
            "Then, in the <answer> tag, output exactly one of the following: 'Answer 1 is better' or 'Answer 2 is better'. No other text is allowed in the <answer> section.\n\n" \
            "Example format:\n" \
            "<think>\n" \
            "1. Semantic accuracy: Answer 1 (9/10) - ...; Answer 2 (7/10) - ...\n" \
            "2. Correctness: Answer 1 (8/10) - ...; Answer 2 (7/10) - ...\n" \
            "3. Clarity: Answer 1 (9/10) - ...; Answer 2 (8/10) - ...\n" \
            "[Additional dimensions if any]: Answer 1 (6/10) - ...; Answer 2 (7/10) - ...\n" \
            "Total score:\nAnswer 1: 9+8+9+6=32\nAnswer 2: 7+7+8+7=29\n" \
            "</think>\n" \
            "<answer>Answer 1 is better</answer>\n\n" \
            "**Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given answers.**\n\n"
            f"Your task is provided as follows:\nQuestion: [{Query}]\nAnswer 1: [{R1}]\nAnswer 2: [{R2}]")
            
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

print(text_outputs[0])
