from datasets import load_from_disk
from PIL import Image
import torch
import tqdm
import os
import random
import json
from io import BytesIO
import base64
from vllm_request import evaluate_batch

import cv2

def read_video_frames(video_path, num_frames=8, save_dir="./frames", prefix="video"):
    os.makedirs(save_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return []

    selected_frames = [int(i * total_frames / num_frames) for i in range(num_frames)]

    frame_paths = []
    current_frame = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in selected_frames:
            frame_name = f"{prefix}_frame_{frame_idx:03d}.jpg"
            frame_path = os.path.join(save_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            frame_idx += 1

        current_frame += 1
        if frame_idx >= num_frames:
            break

    cap.release()
    return frame_paths

input_data = []

prompt = ""

video_path = ''

images.extend(read_video_frames(video_path, num_frames=16, save_dir="./frames", prefix="video"))

problem = (
    "You are presented with a generated video and its associated text caption. "
    "Your task is to analyze the video across multiple dimensions in relation to the caption. Specifically:\n"
    "Provide overall assessments for the video along the following axes (each rated from 1 to 5):\n"
    "- Alignment Score: How well the video matches the caption in terms of content.\n"
    "- Physics Score: How well the gravity, movements, collisions, and interactions make physical sense.\n"
    "- Style Score: How visually appealing the video looks, regardless of caption accuracy.\n\n"
    "Output your evaluation using the format below:\n\n"
    "Alignment Score (1-5): X\n"
    "Physics Score (1-5): Y\n"
    "Style Score (1-5): Z\n\n"
    "Your task is provided as follows:\n"
    f"Text Caption: [{prompt}]"
)

input_data.append({
    'problem': problem,
    'images': images
})

output = evaluate_batch(input_data, "http://localhost:8080", image_root=None)

print(output[0]['model_output'])

'''Example output:

Alignment Score (1-5): 2.4036
Physics Score (1-5): 3.0987
Style Score (1-5): 3.3889
'''
