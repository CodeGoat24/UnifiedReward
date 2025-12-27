

import torch

from PIL import Image
import warnings

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

model_path = "CodeGoat24/UnifiedReward-qwen-7b"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map={"": 'cuda:0'}
)
processor = AutoProcessor.from_pretrained(model_path)


image_path = '/path/to/image'

Query = '' # Question

R1 = '' # Response1
R2 = '' # Response2

image = Image.open(image_path)


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
print(output)
