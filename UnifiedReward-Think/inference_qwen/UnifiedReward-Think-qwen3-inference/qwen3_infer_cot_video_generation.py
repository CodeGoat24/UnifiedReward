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


frames_v1 = extract_frames(video_path_1, num_frames=8)
frames_v2 = extract_frames(video_path_2, num_frames=8)

encoded_frames = [_encode_image(f) for f in frames_v1 + frames_v2]


input_data = [{
    'problem': problem,
    'images': encoded_frames
}]

output = evaluate_batch(input_data, "http://localhost:8080")[0]['model_output']
print(output)
