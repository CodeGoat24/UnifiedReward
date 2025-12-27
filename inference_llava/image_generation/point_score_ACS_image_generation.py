from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

from PIL import Image

import copy
import torch


import warnings






warnings.filterwarnings("ignore")

pretrained = "CodeGoat24/UnifiedReward-7b-v1.5"

model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()

image_path = '/path/to/image'

prompt ='' #prompt of the given image


image = Image.open(image_path).convert('RGB')

image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

question = ("<image>\n"
            "You are presented with a generated image and its associated text caption. Your task is to analyze the image across multiple dimensions in relation to the caption. Specifically:\n\n"
            "1. Evaluate each word in the caption based on how well it is visually represented in the image. Assign a numerical score to each word using the format:\n"
            "   Word-wise Scores: [[\"word1\", score1], [\"word2\", score2], ..., [\"wordN\", scoreN], [\"[No_mistakes]\", scoreM]]\n"
            "   - A higher score indicates that the word is less well represented in the image.\n"
            "   - The special token [No_mistakes] represents whether all elements in the caption were correctly depicted. A high score suggests no mistakes; a low score suggests missing or incorrect elements.\n\n"
            "2. Provide overall assessments for the image along the following axes (each rated from 1 to 5):\n"
            "- Alignment Score: How well the image matches the caption in terms of content.\n"
            "- Coherence Score: How logically consistent the image is (absence of visual glitches, object distortions, etc.).\n"
            "- Style Score: How aesthetically appealing the image looks, regardless of caption accuracy.\n\n"
            "Output your evaluation using the format below:\n\n"
            "---\n\n"
            "Word-wise Scores: [[\"word1\", score1], ..., [\"[No_mistakes]\", scoreM]]\n\n"
            "Alignment Score (1-5): X\n"
            "Coherence Score (1-5): Y\n"
            "Style Score (1-5): Z\n\n"
            f"Your task is provided as follows:\nText Caption: [{prompt}]")

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
    max_new_tokens=4096
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)

print(text_outputs[0])
