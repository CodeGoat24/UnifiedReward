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

def read_video_frames(video_path, num_frames=8):

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        cap.release()
        return []

    selected_frames = [int(i * total_frames / num_frames) for i in range(num_frames)]

    frames = []
    current_frame = 0
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame in selected_frames:
            frames.append(frame)
            frame_idx += 1

        current_frame += 1
        if frame_idx >= num_frames:
            break

    cap.release()
    return frames



input_data = []
images = []

prompt = ""

video_path_1 = ''
video_path_2 = ''

images.extend(read_video_frames(video_path_1, num_frames=8))

images.extend(read_video_frames(video_path_2, num_frames=8))

problem = (
    "You are presented with two generated videos (Video 1 and Video 2) along with a shared text caption. "
    "Your task is to comparatively evaluate the two videos across three specific dimensions:\n\n"
    "- Alignment Score: How well each video matches the caption in terms of content.\n"
    "- Coherence Score: How logically consistent and visually coherent each video is (absence of visual glitches, distorted objects, etc.).\n"
    "- Style Score: How aesthetically appealing each video is, regardless of caption accuracy.\n\n"
    "For each dimension, you must assign a relative score to Video 1 and Video 2, such that:\n"
    "- Each score is a float between 0 and 1 (inclusive).\n"
    "- The scores for Video 1 and Video 2 must sum to exactly 1.0 for each dimension.\n"
    "- The higher the score, the better that video is in the corresponding dimension *relative to the other*.\n\n"
    "This format emphasizes comparative quality rather than absolute evaluation.\n\n"
    "Please provide your evaluation in the format below:\n\n"
    "Alignment Score:\n"
    " Video 1: X\n"
    " Video 2: Y\n\n"
    "Coherence Score:\n"
    " Video 1: X\n"
    " Video 2: Y\n\n"
    "Style Score:\n"
    " Video 1: X\n"
    " Video 2: Y\n\n"
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

Alignment Score:
 Video 1: 0.4736
 Video 2: 0.5264

Coherence Score:
 Video 1: 0.5981
 Video 2: 0.4019

Style Score:
 Video 1: 0.3484
 Video 2: 0.6516

'''