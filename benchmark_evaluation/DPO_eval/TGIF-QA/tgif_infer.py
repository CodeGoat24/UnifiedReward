from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

from PIL import Image

import copy
import torch
import tqdm

import warnings
import os

import random


warnings.filterwarnings("ignore")

def _load_video(video_path, num_video_frames, loader_fps, fps=None, frame_count=None):


        from llava.mm_utils import opencv_extract_frames

        image_size = 512

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

import json
with open('./test_q.json', 'r') as file:
    query = json.load(file)

with open('./test_a.json', 'r') as file:
    answer = json.load(file)

dataset = []

for q, a in zip(query, answer):
    q['answer'] = a['answer']
    dataset.append(q)

pretrained = "CodeGoat24/LLaVA-Video-7B-Qwen2-UnifiedReward-DPO"

save_path = "./tgif_results.json"

model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()

# random.seed(0)
# if len(dataset) > 5000:
#     dataset = sample(dataset, 5000)

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
    file_name = f"{data['video_name']}.mp4"
    video_path = os.path.join(video_folder, file_name)

    num_video_frames = 32
    loader_fps = 0.0
    fps = None
    frame_count = None

    images, frames_loaded = _load_video(
        video_path, num_video_frames, loader_fps, fps=fps, frame_count=frame_count
    )
    image_sizes = []
    images = [img.resize((512, 512)) for img in images]
    for img in images:
        image_sizes.append(img.size)

    image_tensor = image_processor.preprocess(images, return_tensors="pt")["pixel_values"].cuda().bfloat16()

    conv_template = "qwen_1_5" 

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
        "id": data['question_id'],
        "video": file_name,
        "question": data['question'],
        "answer": data['answer'],
        "pred": output
    })
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)
