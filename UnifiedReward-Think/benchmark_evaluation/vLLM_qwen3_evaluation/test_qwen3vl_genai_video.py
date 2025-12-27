import requests
import tempfile
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import decord
from datasets import load_dataset
import tqdm

from vllm_qwen.vllm_request import evaluate_batch


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

    # UnifiedReward-Think-qwen3vl Models Input Template for Video Generation
    problem = f'''You are an objective and precise evaluator for video quality comparison. I will provide you with a text caption and a sequence of consecutive frames extracted from two generated videos based on this caption. The first half of the frames belong to Video 1, and the second half of the frames belong to Video 2. You must analyze these two videos carefully and determine which video is better.

        Instructions (MUST follow strictly):
        1. All reasoning, analysis, explanations, and scores MUST be written strictly inside <think> and </think> tags.
        2. The <think> block must start immediately with the first evaluation dimension. Do NOT include any introduction, notes, or explanations before the first numbered dimension.
        3. After </think>, output the final judgment strictly inside <answer> and </answer> tags, containing only one of:
        - Video 1 is better
        - Video 2 is better
        4. Do NOT output anything outside <think> and <answer>. No extra explanations, notes, or prefaces.

        Evaluation procedure:

        1. The caption for the generated videos is: 「{prompt}」. The provided frames represent two candidate videos:
        - First half: Video 1
        - Second half: Video 2

        2. You must evaluate the two videos across these core dimensions:
        - Semantic consistency (how closely the video content aligns with the caption)
        - Temporal coherence (smoothness and logical flow of motion across frames)
        - Authenticity (realism and attention to detail)

        3. You may also add up to two additional evaluation dimensions if they are clearly relevant (e.g., camera stability, lighting consistency, creativity). If no extra dimensions are relevant, keep only the three core dimensions.

        4. For each evaluation dimension:
        - Provide a score between 1–10 for both Video 1 and Video 2.
        - Provide a short rationale for each score (2–5 short sentences).
        - Each dimension must follow exactly this 3-line block format with numbering, line breaks, and indentation:
            N. Dimension name: 
                Video 1 (x/10) - rationale; 
                Video 2 (y/10) - rationale

        5. After evaluating all dimensions, calculate the total score for each video and show the calculation explicitly, following this exact format:
            Total score:
            Video 1: x+x+x(+...)=total
            Video 2: y+y+y(+...)=total

        6. All reasoning, analysis, scoring, and totals must be written strictly inside <think> and </think> tags. Nothing related to reasoning or scores may appear outside <think>.

        Required output format (follow this exactly, including line breaks and indentation):

        <think>
        1. Semantic consistency: 
            Video 1 (9/10) - ...; 
            Video 2 (7/10) - ...
        2. Temporal coherence: 
            Video 1 (8/10) - ...; 
            Video 2 (6/10) - ...
        3. Authenticity: 
            Video 1 (7/10) - ...; 
            Video 2 (5/10) - ...
        [Additional dimension if any]: 
            Video 1 (8/10) - ...; 
            Video 2 (6/10) - ...
        [Additional dimension if any]: 
            Video 1 (7/10) - ...; 
            Video 2 (7/10) - ...
        Total score:
        Video 1: 9+8+7+8+7=39
        Video 2: 7+6+5+6+7=31
        </think>
        <answer>Video 1 is better</answer>

        Note: The example above is only to illustrate the exact format (numbering, line breaks, indentation, and style). Your actual evaluation must follow this format exactly, but be based on the given caption and the two provided videos (frames divided into two halves).
        '''
   
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
