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


def extract_frames(video_path, num_frames=8):
    try:
        vr = decord.VideoReader(video_path, ctx=decord.cpu())
    except Exception as e:
        raise RuntimeError(f"Decord failed to read: {video_path}, error: {e}")

    total = len(vr)
    if total == 0:
        raise ValueError(f"No frames in video: {video_path}")

    idx = np.linspace(0, total - 1, num_frames).astype(np.int32)
    frames = vr.get_batch(idx).asnumpy()
    return [Image.fromarray(f) for f in frames]



dataset = load_dataset("KwaiVGI/VideoGen-RewardBench")["eval"]


# wget https://huggingface.co/datasets/KlingTeam/VideoGen-RewardBench/resolve/main/videos.zip
VIDEO_ROOT = "./" 


input_data = []

for i in tqdm.trange(len(dataset)):
    data = dataset[i]

    label = data["Overall"].strip()
    if label == "same":
        continue
    elif label == "A":
        answer = "Video 1 is better"
    elif label == "B":
        answer = "Video 2 is better"
    else:
        continue

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

    path_A = VIDEO_ROOT + data["path_A"]  # usually './videos/xxxx.mp4'
    path_B = VIDEO_ROOT + data["path_B"]

    try:
        frames_v1 = extract_frames(path_A, num_frames=8)
        frames_v2 = extract_frames(path_B, num_frames=8)
    except Exception as e:
        print("Error extracting frames:", e)
        continue

    encoded_frames = [encode_img(f) for f in (frames_v1 + frames_v2)]

    input_data.append({
        "problem": problem,
        "images": encoded_frames,
        "answer": answer,
    })


outputs = []
batch_size = 32

for i in tqdm.trange(0, len(input_data), batch_size):
    batch = input_data[i:i+batch_size]
    out = evaluate_batch(batch, "http://localhost:8080")
    outputs.extend(out)


correct = 0
for item in outputs:
    if item["answer"] in item["model_output"]:
        correct += 1
    else:
        print("Wrong output:", item["model_output"])
        print("Correct:", item["answer"])

acc = correct / len(outputs)
print(f"Acc.: {correct}/{len(outputs)} = {acc:.4f}")
