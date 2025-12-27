
from io import BytesIO
import base64
from PIL import Image

from vllm_qwen.vllm_request import evaluate_batch

def _encode_image(img: Image.Image):
    img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


input_data = []

image_path = '/path/to/image'

Query = '' # Question

R1 = '' # Response1
R2 = '' # Response2

problem = f"You are an objective and precise evaluator for image-based question answering. I will provide you with a question, a reference image, and two candidate answers. You must analyze the two answers carefully and determine which one is better.\n\n        Instructions (MUST follow strictly):\n        1. All reasoning, analysis, explanations, and scores MUST be written strictly inside <think> and </think> tags. \n        2. The <think> block must start immediately with the first evaluation dimension. Do NOT include any introduction, notes, or explanations before the first numbered dimension.\n        3. After </think>, output the final judgment strictly inside <answer> and </answer> tags, containing only one of:\n        - Answer 1 is better\n        - Answer 2 is better\n        4. Do NOT output anything outside <think> and <answer>. No extra explanations, notes, or prefaces.\n\n        Evaluation procedure:\n        1. The question is: \u300c{Query}\u300d. The reference image is provided. The two candidate answers are:\n        Answer 1: \u300c{R1}\u300d\nAnswer 2: \u300c{R2}\u300d\n\n2. You must evaluate the two answers across these core dimensions:\n        - Semantic accuracy (how well the answer reflects the visual content of the image)\n        - Correctness (whether the answer is logically and factually correct)\n        - Clarity (whether the answer is clearly and fluently expressed)\n\n        3. You are also encouraged to add up to two additional evaluation dimensions if they are relevant (e.g., reasoning ability, attention to detail, multimodal grounding). If no extra dimensions are relevant, just keep the three core dimensions.\n\n        4. For each evaluation dimension:\n        - Provide a score between 1\u201310 for both Answer 1 and Answer 2\n        - Provide a short rationale for each score (2\u20135 short sentences)\n        - Each dimension must follow exactly this 3-line block format with numbering, line breaks, and indentation:\n            N. Dimension name: \n                Answer 1 (x/10) - rationale; \n                Answer 2 (y/10) - rationale\n\n        5. After evaluating all dimensions, calculate the total score for each answer and show the calculation explicitly, following this exact format:\n            Total score:\n            Answer 1: x+x+x=total\n            Answer 2: y+y+y=total\n\n        Required output format:\n\n        <think>\n        1. Semantic accuracy: \n            Answer 1 (9/10) - ...; \n            Answer 2 (7/10) - ...\n        2. Correctness: \n            Answer 1 (8/10) - ...; \n            Answer 2 (7/10) - ...\n        3. Clarity: \n            Answer 1 (9/10) - ...; \n            Answer 2 (8/10) - ...\n        [Additional dimension if any]: \n            Answer 1 (6/10) - ...; \n            Answer 2 (7/10) - ...\n        [Additional dimension if any]: \n            Answer 1 (9/10) - ...; \n            Answer 2 (7/10) - ...\n        Total score:\n        Answer 1: 9+8+9+6+9=41\n        Answer 2: 7+7+8+7+7=36\n        </think>\n        <answer>Answer 1 is better</answer>\n\n        Note: The example above is only to illustrate the exact format (numbering, line breaks, indentation, and style). Your actual evaluation must follow this format exactly, but be based on the given question, reference image, and candidate answers.\n        "


image = _encode_image(Image.open(image_path))


input_data = [{
    'problem': problem,
    'images': [image]
}]

output = evaluate_batch(input_data, "http://localhost:8080")[0]['model_output']

print(output)