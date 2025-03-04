from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
import tqdm
import sys
import warnings
import os
from datasets import load_dataset
import random
from random import sample

warnings.filterwarnings("ignore")

def _load_video(video_path, num_video_frames, loader_fps, fps=None, frame_count=None):
        from torchvision import transforms

        from llava.mm_utils import opencv_extract_frames

        image_size = 512

        try:
            pil_imgs, frames_loaded = opencv_extract_frames(video_path, num_video_frames, loader_fps, fps, frame_count)
        except Exception as e:
            video_loading_succeed = False
            print(f"bad data path {video_path}")
            print(f"[DEBUG] Error processing {video_path}: {e}")
            # video_outputs = torch.zeros(3, 8, image_size, image_size, dtype=torch.uint8)
            empty_num_video_frames = int(random.uniform(2, num_video_frames))
            # pil_imgs = [torch.zeros(3, image_size, image_size, dtype=torch.float32)] * empty_num_video_frames
            pil_imgs = [Image.new("RGB", (448, 448), (0, 0, 0))] * empty_num_video_frames
            frames_loaded = 0

        return pil_imgs, frames_loaded

pretrained = "CodeGoat24/LLaVA-Video-7B-Qwen2-UnifiedReward-DPO"

save_path = "./msvd_results.json"


model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()

import json
with open('./test_qa.json', 'r') as file:
    dataset = json.load(file)

# random.seed(0)
# if len(dataset) > 5000:
#     print(len(dataset))
#     dataset = sample(dataset, 5000)

file_path = './youtube_mapping.txt'

# Initialize an empty dictionary to store the results
mapping_dict = {}

# Read the file and process each line
with open(file_path, 'r') as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) == 2:  # Ensure there are exactly two parts per line
            value, key = parts  # Assign the first part as value and the second part as key
            key = key.lstrip('vid')
            mapping_dict[key] = value.lstrip('-')

if os.path.exists(save_path):
    with open(save_path, 'r') as file:
        data_list = json.load(file)
else:
    data_list = []
for i in tqdm.trange(len(dataset)):
    if i < len(data_list):
        continue

    data = dataset[i]

    video_folder = "./videos"
    mapping_file = mapping_dict[str(data['video_id'])]
    file_name = f"{mapping_file}.avi"
    video_path = os.path.join(video_folder, file_name)

    num_video_frames = 32
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

    Query = data['question']

    question = '<image>\n'*len(images) + Query


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
    output = text_outputs[0]


    data_list.append({
        "id": data['id'],
        "video": file_name,
        "question": data['question'],
        "answer": data['answer'],
        "pred": output
    })
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)