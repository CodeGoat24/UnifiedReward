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

def _encode_image(image):
    if isinstance(image, str):
        with Image.open(image) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")
    else:
        img = image.convert("RGB")
        buffered = BytesIO()
        img.save(buffered, format="JPEG", quality=95)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

input_data = []

prompt = ''

image_path1 = ''
image_path2 = ''


problem = (
        "You are presented with two generated images (Image 1 and Image 2) along with a shared text caption. "
        "Your task is to comparatively evaluate the two images across three specific dimensions:\n\n"
        "- Alignment Score: How well each image matches the caption in terms of content.\n"
        "- Coherence Score: How logically consistent and visually coherent each image is (absence of visual glitches, distorted objects, etc.).\n"
        "- Style Score: How aesthetically appealing each image is, regardless of caption accuracy.\n\n"
        "For each dimension, you must assign a relative score to Image 1 and Image 2, such that:\n"
        "- Each score is a float between 0 and 1 (inclusive).\n"
        "- The scores for Image 1 and Image 2 must sum to exactly 1.0 for each dimension.\n"
        "- The higher the score, the better that image is in the corresponding dimension *relative to the other*.\n\n"
        "This format emphasizes comparative quality rather than absolute evaluation.\n\n"
        "Please provide your evaluation in the format below:\n\n"
        "Alignment Score:\n"
        " Image 1: X\n"
        " Image 2: Y\n\n"
        "Coherence Score:\n"
        " Image 1: X\n"
        " Image 2: Y\n\n"
        "Style Score:\n"
        " Image 1: X\n"
        " Image 2: Y\n\n"
        "Your task is provided as follows:\n"
        f"Text Caption: [{prompt}]"
    )
images = [
            _encode_image(image_path1),
            _encode_image(image_path2),
        ]

input_data.append({
    'problem': problem,
    'images': images
})

output = evaluate_batch(input_data, "http://localhost:8080", image_root=None)

print(output[0]['model_output'])

''' Example output:

Alignment Score:
 Image 1: 0.435699999332428
 Image 2: 0.564300000667572

Coherence Score:
 Image 1: 0.4180000126361847
 Image 2: 0.5820000171661377

Style Score:
 Image 1: 0.40790000557899475
 Image 2: 0.5921000242233276
'''