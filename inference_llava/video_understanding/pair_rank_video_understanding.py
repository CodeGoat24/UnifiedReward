from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

from PIL import Image

import copy
import torch

import warnings


import random

warnings.filterwarnings("ignore")
pretrained = "CodeGoat24/UnifiedReward-7b"

def _load_video(video_path, num_video_frames, loader_fps, fps=None, frame_count=None):


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
images = [img.resize((512, 512)) for img in images]
image_sizes = [img.size for img in images]

image_tensor = image_processor.preprocess(images, return_tensors="pt")["pixel_values"].cuda().bfloat16()

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models



question = '<image>\n'*len(images) + f'Please analyze the following video and question, then determine which of the two provided answers is better\nQuestion: {Query}\nAnswer 1: {R1}\nAnswer 2: {R2}\nPlease evaluate both answers based on the following criteria:\n1. Accuracy: How well does the answer align with the visual information in the video?\n2. Completeness: Does the answer fully address all aspects of the question?\n3. Clarity: Is the answer easy to understand and well-articulated?\n4. Relevance: Does the answer directly relate to the question and the video?\nAfter your evaluation, please:\n1. Explain your reasoning for each criterion.\n2. Provide an overall judgment on which answer is better (Answer 1 or Answer 2). For example: \"Overall Judgment: Answer 1 is better than Answer 2.\"\nYour response should be structured and detailed, demonstrating your understanding of both the visual and textual elements of the task. You have to choose the better of the two answers and can\'t say that both answers are equally good.'


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
