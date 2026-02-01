from io import BytesIO
import base64
from PIL import Image
import cv2
import numpy as np

from vllm_qwen.vllm_request import evaluate_batch


def _encode_image(img: Image.Image):
    img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_frames(video_path, num_frames=8):
    """Uniformly sample N frames from a video and return PIL Images."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        raise ValueError(f"Video has no frames: {video_path}")

    indices = np.linspace(0, frame_count - 1, num_frames).astype(int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Failed reading frame {idx} from {video_path}")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame)
        frames.append(pil_img)

    cap.release()
    return frames



video_path_1 = "/path/to/video1.mp4"
video_path_2 = "/path/to/video2.mp4"

prompt = ""  # your caption


problem = f"""## Identity
You are a top-tier AI Video Evaluation Expert. Perform a hierarchical, multi-dimensional comparative analysis of Video 1 and Video 2 based on the provided Prompt.

## Evaluation Framework

### 1. Mandatory Categories
For each, independently define **3-5 specific sub-dimensions** based on the videos' actual content:
- **A. Semantic Alignment & Accuracy**: Accuracy of subjects, attributes, spatial relationships, and environment as defined by the prompt.
- **B. Video Quality & Dynamic Realism**: Technical fidelity, temporal stability (no flickering/warping), subject identity persistence, and physical plausibility of motion.
- **C. Narrative, Aesthetics & Cinematography**: Composition, color harmony, camera movement quality (smoothness/intent), and narrative flow.
*Note: If the prompt involves unique traits, you are encouraged to add a personalized Category D.*

### 2. Core Rules
- **Dynamic Selection**: Do NOT simply copy a fixed list. Choose sub-dimensions that most effectively differentiate the two videos.
- **Sum-of-10 Scoring**: For every sub-dimension, the total score (Video 1 + Video 2) MUST strictly equal 10 points (e.g., 6+4, 5+5).
- **Evidence-Based Reasoning**: Provide professional, critical analysis pointing to specific visual/temporal evidence.

## Input Data
**Prompt:** [{prompt}]

**Content to be Evaluated:**
[Video 1] 
[Video 2] 

## Output Format
Return a single, valid JSON object in English.

```json
{{
  "prompt": "[Original Prompt]",
  "categories": [
    {{
      "name": "[Category Name]",
      "dims": [
        {{
          "name": "[Custom Sub-dimension]",
          "reason_1": "[Specific evidence]",
          "reason_2": "[Specific evidence]",
          "score_1": 0-10,
          "score_2": 0-10
        }}
      ],
      "cat_reason": "[Category-level analysis]",
      "cat_winner": "Video 1/2"
    }}
  ],
  "reason": "[Overall analysis]",
  "winner": "Video 1/2"
}}
"""


frames_v1 = extract_frames(video_path_1, num_frames=8)
frames_v2 = extract_frames(video_path_2, num_frames=8)

encoded_frames = [_encode_image(f) for f in frames_v1 + frames_v2]


input_data = [{
    'problem': problem,
    'images': encoded_frames
}]

output = evaluate_batch(input_data, "http://localhost:8080")[0]['model_output']

output_path = "pair_rank_video_output.json"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(output)

print(output)
