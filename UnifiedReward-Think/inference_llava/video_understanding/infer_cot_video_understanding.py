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
pretrained = "CodeGoat24/UnifiedReward-Think-7b"

def _load_video(video_path, num_video_frames, loader_fps, fps=None, frame_count=None):
        from torchvision import transforms

        from llava.mm_utils import opencv_extract_frames

        try:
            pil_imgs, frames_loaded = opencv_extract_frames(video_path, num_video_frames, loader_fps, fps, frame_count)
        except Exception as e:
            video_loading_succeed = False
            print(f"bad data path {video_path}")
            print(f"[DEBUG] Error processing {video_path}: {e}")
            empty_num_video_frames = int(random.uniform(2, num_video_frames))
            pil_imgs = [Image.new("RGB", (448, 448), (0, 0, 0))] * empty_num_video_frames
            frames_loaded = 0

        return pil_imgs, frames_loaded

model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()

Query = "" # Question
R1 = "" # Response1
R2 = ""# Response2

video_path = '/path/to/video'

num_video_frames = 8
loader_fps = 0.0
fps = None
frame_count = None

images, frames_loaded = _load_video(
    video_path, num_video_frames, loader_fps, fps=fps, frame_count=frame_count
)
image_sizes = []
for img in images:
    img.resize((512, 512))
    image_sizes.append(img.size)

image_tensor = image_processor.preprocess(images, return_tensors="pt")["pixel_values"].cuda().bfloat16()

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models



question = ('<image>\n'*len(images) + "Given a question and a reference video, please evaluate the two provided answers (Answer 1 and Answer 2). " \
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
                "Total score:\Answer 1: 8+7+9+7=31\Answer 2: 9+6+8+6=29\n" \
                "</think>\n" \
                "<answer>Answer 1 is better</answer>\n\n" \
                "**Note: In the example above, scores and the final answer are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of two given answers.**\n\n" \
                f"Your task is provided as follows:\nQuestion: {Query}\nAnswer 1: {R1}\nAnswer 2: {R2}\n"
    )

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

print(text_outputs[0])
        
