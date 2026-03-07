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



video_path = "/path/to/video.mp4"

Query = "" # Question
R1 = "" # Response1
R2 = "" # Response2


problem = f"You are an objective and precise evaluator for video understanding. I will provide you with a question, a reference video, and two candidate answers. You must analyze carefully and determine which answer is better.\n\n        Instructions (MUST follow strictly):\n        1. All reasoning, analysis, explanations, and scores MUST be written strictly inside <think> and </think> tags.\n        2. The <think> block must start immediately with the first evaluation dimension. Do NOT include any introduction, notes, or explanations before the first numbered dimension.\n        3. After </think>, output the final judgment strictly inside <answer> and </answer> tags, containing only one of:\n        - Answer 1 is better\n        - Answer 2 is better\n        4. Do NOT output anything outside <think> and <answer>. No extra explanations, notes, or prefaces.\n\n        Evaluation procedure:\n\n        1. The task is to evaluate two answers given a reference video and the following question: \u300c{Query}\u300d. The provided answers are:\n        - Answer 1: {R1}\n        - Answer 2: {R2}\n\n        2. You must evaluate the two answers across these core dimensions:\n        - Semantic accuracy (does the answer align with the visual and temporal content in the video?)\n        - Correctness (is the answer factually and logically correct?)\n        - Clarity (is the answer expressed fluently, clearly, and coherently?)\n\n        3. You may also add up to two additional evaluation dimensions if they are clearly relevant (e.g., temporal reasoning, causal understanding, visual detail, emotional perception). If no extra dimensions are relevant, keep only the three core dimensions.\n\n        4. For each evaluation dimension:\n        - Provide a score between 1\u201310 for both Answer 1 and Answer 2.\n        - Provide a short rationale for each score (2\u20135 short sentences).\n        - Each dimension must follow exactly this 3-line block format with numbering, line breaks, and indentation:\n            N. Dimension name: \n                Answer 1 (x/10) - rationale; \n                Answer 2 (y/10) - rationale\n\n        5. After evaluating all dimensions, calculate the total score for each answer and show the calculation explicitly, following this exact format:\n            Total score:\n            Answer 1: x+x+x(+...)=total\n            Answer 2: y+y+y(+...)=total\n\n        6. All reasoning, analysis, scoring, and totals must be written strictly inside <think> and </think> tags. Nothing related to reasoning or scores may appear outside <think>.\n\n        Required output format (follow this exactly, including line breaks and indentation):\n\n        <think>\n        1. Semantic accuracy: \n            Answer 1 (9/10) - ...; \n            Answer 2 (7/10) - ...\n        2. Correctness: \n            Answer 1 (8/10) - ...; \n            Answer 2 (6/10) - ...\n        3. Clarity: \n            Answer 1 (9/10) - ...; \n            Answer 2 (8/10) - ...\n        [Additional dimension if any]: \n            Answer 1 (7/10) - ...; \n            Answer 2 (6/10) - ...\n        [Additional dimension if any]: \n            Answer 1 (8/10) - ...; \n            Answer 2 (7/10) - ...\n        Total score:\n        Answer 1: 9+8+9+7+8=41\n        Answer 2: 7+6+8+6+7=34\n        </think>\n        <answer>Answer 1 is better</answer>\n\n        Note: The example above is only to illustrate the exact format (numbering, line breaks, indentation, and style). Your actual evaluation must follow this format exactly, but be based on the given question, reference video, and candidate answers.\n        "


frames = extract_frames(video_path, num_frames=8)

encoded_frames = [_encode_image(f) for f in frames]


input_data = [{
    'problem': problem,
    'images': encoded_frames
}]

output = evaluate_batch(input_data, "http://localhost:8080")[0]['model_output']
print(output)
