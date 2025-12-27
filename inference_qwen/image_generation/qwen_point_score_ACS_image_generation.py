from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from PIL import Image
import torch






model_path = 'CodeGoat24/UnifiedReward-qwen-7b'
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map={"": 'cuda:0'}
)
processor = AutoProcessor.from_pretrained(model_path)

image_path = '/path/to/image'

prompt ='' #prompt of the given image

image = Image.open(image_path)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": (
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
            },
        ],
    }
]


text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(model.device)


with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)[0]

print(output_text)
