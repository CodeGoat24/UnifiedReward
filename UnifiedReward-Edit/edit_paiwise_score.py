from PIL import Image
import torch
import tqdm
import os
import random
import json
from io import BytesIO
import base64
from vllm_request import evaluate_batch
import re

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

input_data=[]

instruction = ''

problem = f"You are tasked with assigning scores to two edited images, comparing each with the original source image. \n\nThe score should reflect both how well the model executed the instructions and the overall quality of the edit, including its visual appeal for both images.\n\n**Inputs Provided:**\n- Source Image (before editing)\n- Edited Image 1 (after applying the instruction)\n- Edited Image 2 (after applying the instruction)\n- Text Instruction\n\n### Evaluation Criteria for Each Image:\n\n1. **Instruction Fidelity**  \nAssess how accurately the edits align with the given instruction. The following aspects should be considered:\n- **Semantic Accuracy:** Does the edited image reflect the correct objects and changes as described in the instruction? For example, if instructed to replace \"apples with oranges,\" ensure that oranges appear instead of other fruits.\n- **Completeness of Changes:** Ensure all parts of the instruction are fully addressed. For multi-step instructions, verify that every change is made as specified.\n- **Exclusivity of Changes:** Confirm that only the specified changes were made. Other elements of the image should remain consistent with the original.\n\n2. **Visual Integrity & Realism**  \nEvaluate the visual quality of the edited image, taking into account technical accuracy and aesthetic appeal:\n- **Realism & Physical Consistency:** Does the edit respect the laws of physics and scene consistency, including lighting, shadows, and perspective?\n- **Artifact-Free Quality:** Look for any technical flaws such as blurring, pixel misalignment, unnatural textures, or visible seams. The image should be clean and free from distractions.\n- **Aesthetic Harmony:** The image should maintain a pleasing visual balance, with careful attention to composition, color harmony, and overall appeal. The changes should enhance the image rather than detract from it.\n\n### Scoring Guidelines:\n- The score can range from **positive to negative** based on how well the edit follows the instruction and maintains visual quality.\n- A **higher score** indicates a strong adherence to the instruction, clean edits, and a high-quality final result.\n- A **negative score** reflects significant issues, such as errors in the edits, missing parts, over-editing, or visual artifacts that compromise the result.\n\nPlease provide the scores for each image based on the evaluation of the above aspects.\n\nText instruction - {instruction}\n"

source_img = _encode_image('/path/to/image')
left_img =  _encode_image('/path/to/image')
right_img =  _encode_image('/path/to/image')

input_data.append({
    'problem': problem,
    'images': [source_img, left_img, right_img],
})

output = evaluate_batch(input_data, "http://localhost:8080")

print(item['model_output'])

# Extract scores from model_output:

# pattern = r"([0-9]+\.[0-9]+)"

# scores = re.findall(pattern, item['model_output'])

# image_1_score = float(scores[0])
# image_2_score = float(scores[1])

# print(image_1_score)
# print(image_2_score)
        