
from io import BytesIO
import base64
from PIL import Image

from vllm_qwen.vllm_request import evaluate_batch

def _encode_image(img: Image.Image):
    img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode("utf-8")



image_path_1 = '/path/to/image1'
image_path_2 = '/path/to/image2'

prompt = "" # prompt of given images

problem = f"""## Identity
You are a top-tier AI Image Content Evaluation Expert. Your task is to perform a hierarchical, multi-dimensional comparative analysis of Image 1 and Image 2 based on the provided Prompt.

## Evaluation Framework

### 1. Mandatory Starting Categories
For every evaluation, you MUST address these three core areas, but you should **independently define 3-5 sub-dimensions** for each based on what makes the images distinct:
- **A. Semantic Alignment & Accuracy**: Evaluate how well the images capture the prompt's subjects, actions, and constraints.
- **B. Image Quality & Realism**: Evaluate technical execution, physical logic, and visual clarity.
- **C. Aesthetics & Artistry**: Evaluate artistic appeal, color harmony, and compositional mastery.
*Note: If the prompt involves unique traits, you are encouraged to add a personalized Category D.*

### 2. Scoring & Reasoning Rules
- **Dynamic Dimensions**: Do not rely on a fixed list. Choose sub-dimensions that best highlight the differences between the two images.
- **Sum-of-10 Constraint**: For every sub-dimension, the scores for Image 1 and Image 2 MUST total exactly 10 (e.g., 8+2, 5+5).
- **Evidence-Based Reasoning**: Provide professional, critical analysis for each score. Avoid generic praise; point out specific visual evidence.

## Input Data
**Prompt:** [{prompt}]

**Content to be Evaluated:**
[Image 1] 
[Image 2] 

## Output Format
Output the results as a single, complete JSON object.

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
      "cat_winner": "Image 1/2"
    }}
  ],
  "reason": "[Overall analysis]",
  "winner": "Image 1/2"
}}
"""

left_image = Image.open(image_path_1)
right_image = Image.open(image_path_2)

left_image = _encode_image(left_image)
right_image = _encode_image(right_image)

input_data = [{
    'problem': problem,
    'images': [left_image, right_image]
}]

output = evaluate_batch(input_data, "http://localhost:8080")[0]['model_output']

output_path = "pair_rank_image_output.json"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(output)

print(output)
