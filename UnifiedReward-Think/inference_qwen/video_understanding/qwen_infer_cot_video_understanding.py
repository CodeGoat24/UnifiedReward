





import warnings

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

model_path = 'CodeGoat24/UnifiedReward-Think-qwen-7b'
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map={"": 'cuda:0'}
)

processor = AutoProcessor.from_pretrained(model_path)

Query = "" # Question
R1 = "" # Response1
R2 = ""# Response2

video_path = '/path/to/video'

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": "Given a question and a reference video, please evaluate the two provided answers (Answer 1 and Answer 2). " \
                "Judge them based on the following key dimensions:\n" \
                "1. Semantic accuracy: Does the answer align with the visual and temporal content in the video?\n" \
                "2. Correctness: Is the answer factually and logically correct?\n" \
                "3. Clarity: Is the answer expressed fluently, clearly, and coherently?\n" \
                "You are encouraged to consider any additional dimensions if relevant (e.g., temporal reasoning, causal understanding, visual detail, emotional perception, etc.). " \
                "For each dimension, assign a score from 1 to 10 for both answers and explain briefly. " \
                "Then, compute and explicitly show the total score as an addition of all dimension scores. " \
                "Wrap your full reasoning in <think> tags. In the <answer> tag, output exactly one of the following: 'Answer 1 is better' or 'Answer 2 is better'. No additional commentary is allowed in the <answer> section.\n\n" \
                "Example format:\n" \
                "<think>\n" \
                "1. Semantic accuracy: Answer 1 (8/10) - ...; Answer 2 (9/10) - ...\n" \
                "2. Correctness: Answer 1 (7/10) - ...; Answer 2 (6/10) - ...\n" \
                "3. Clarity: Answer 1 (9/10) - ...; Answer 2 (8/10) - ...\n" \
                "[Additional dimensions if any]: Answer 1 (7/10) - ...; Answer 2 (6/10) - ...\n" \
                "Total score:\nAnswer 1: 8+7+9+7=31\nAnswer 2: 9+6+8+6=29\n" \
                "</think>\n" \
                "<answer>Answer 1 is better</answer>\n\n" \
                "**Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given answers.**\n\n" \
                f"Your task is provided as follows:\nQuestion: {Query}\nAnswer 1: {R1}\nAnswer 2: {R2}\n"}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt", **video_kwargs).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

print(output_text)
