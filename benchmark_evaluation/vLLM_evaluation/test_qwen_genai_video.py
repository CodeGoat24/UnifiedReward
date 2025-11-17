import requests
import tempfile
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import decord
from datasets import load_dataset
import tqdm

from vllm_request import evaluate_batch


def encode_img(img: Image.Image):
    buf = BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def download_video(url: str):
    r = requests.get(url, timeout=20)
    if r.status_code != 200:
        raise ValueError(f"Failed: {url}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(r.content)
    tmp.close()
    return tmp.name


def extract_frames(video_path, num_frames=8):
    vr = decord.VideoReader(video_path, ctx=decord.cpu())
    total = len(vr)

    if total == 0:
        raise ValueError(f"No frames in video: {video_path}")

    idx = np.linspace(0, total - 1, num_frames).astype(np.int32)
    frames = vr.get_batch(idx).asnumpy()

    return [Image.fromarray(f) for f in frames]


dataset = load_dataset("TIGER-Lab/GenAI-Bench", "video_generation")["test"]

input_data = []

for i in tqdm.trange(len(dataset)):
    data = dataset[i]

    # Skip ties
    if "both" in data["vote_type"] or "tie" in data["vote_type"]:
        continue

    answer = (
        "Video 1 is better" if "left" in data["vote_type"] else "Video 2 is better"
    )

    prompt = data["prompt"]

    problem = (     
            "Imagine you are an expert tasked with evaluating AI-generated videos. You are provided with a text caption and two videos generated based on that caption. Your job is to assess and compare these videos based on the following two main factors:\n\n"
            "1. Caption Alignment: Evaluate how closely each video matches the description provided in the caption. Pay attention to the accuracy of objects depicted, their relationships, and any attributes described in the caption.\n\n"
            "2. Overall Video Quality: Look at the overall visual appeal of each video, considering clarity, the level of detail, color accuracy, and how aesthetically pleasing the video is.\n\n"
            "Using these factors, compare the two videos and determine which one better reflects the caption and exhibits better visual quality.\n\n"
            "Give your final judgment, such as 'Video 1 is better,' 'Video 2 is better,' or 'Both videos are equally good.'\n\n"
            "Your task is as follows:\n"
            f"Text Caption: [{prompt}]\n"
            )
   
    v1 = download_video(data["left_video"])
    v2 = download_video(data["right_video"])

    frames_v1 = extract_frames(v1, num_frames=8)
    frames_v2 = extract_frames(v2, num_frames=8)

    encoded_frames = [encode_img(f) for f in (frames_v1 + frames_v2)]

    input_data.append({
        "problem": problem,
        "images": encoded_frames,
        "answer": answer,
    })


output = evaluate_batch(input_data, "http://localhost:8080")

correct = 0
for item in output:
    if item["answer"] in item["model_output"]:
        correct += 1
    else:
        print(item["model_output"])
        print(item["answer"])

acc = correct / len(input_data)
print(f"Acc.: {correct}/{len(input_data)} = {acc:.4f}")
