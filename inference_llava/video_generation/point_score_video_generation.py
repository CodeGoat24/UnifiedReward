from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

from PIL import Image

import copy
import torch


import warnings


import random

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

warnings.filterwarnings("ignore")

model_name = "llava_qwen"
device = "cuda"
device_map = "auto"

tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()


prompt = ""

video_path = "/path/to/video"

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

question = '<image>\n'*len(images) + f'Suppose you are an expert in judging and evaluating the quality of AI-generated videos, please watch the frames of a given video and see the text prompt for generating the video.\nThen give scores from 5 different dimensions:\n(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color\n(2) temporal consistency, the consistency of objects or humans in video\n(3) dynamic degree, the degree of dynamic changes\n(4) text-to-video alignment, the alignment between the text prompt and the video content\n(5) factual consistency, the consistency of the video content with the common-sense and factual knowledge\n\nFor each dimension, output a number from [1,2,3,4], \nin which \'1\' means \'Bad\', \'2\' means \'Average\', \'3\' means \'Good\', \n\'4\' means \'Real\' or \'Perfect\' (the video is like a real video)\nFinally, based on above 5 dimensions, assign a score from 1 to 10 after \'Final Score:\'\nHere is an output example:\nvisual quality: 4\ntemporal consistency: 4\ndynamic degree: 3\ntext-to-video alignment: 1\nfactual consistency: 2\nFinal Score: 6\n\n**Note: In the example above, scores are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of the given video.**\nYour task is provided as follows: Text Prompt: [{prompt}]'


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
        max_new_tokens=4096
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)

print(text_outputs[0])
