

import torch

from PIL import Image
import warnings

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

model_path = "CodeGoat24/UnifiedReward-Think-qwen-7b"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map={"": 'cuda:0'}
)
processor = AutoProcessor.from_pretrained(model_path)


image_path = '/path/to/image'

Query = '' # Question

R1 = '' # Response1
R2 = '' # Response2

image = Image.open(image_path)


prompt_text = ("Given a question and a reference image, please analyze in detail the two provided answers (Answer 1 and Answer 2). " \
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
