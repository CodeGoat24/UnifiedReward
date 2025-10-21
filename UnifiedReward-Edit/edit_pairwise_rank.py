from PIL import Image
import torch
import tqdm
import os
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

input_data=[]

instruction = ''

problem = f"You are tasked with comparing two edited images and determining which one is better based on the given criteria.\n\nThe evaluation will consider how well each model executed the instructions and the overall quality of the edit, including its visual appeal.\n\n**Inputs Provided:**\n- Source Image (before editing)\n- Edited Image 1 (after applying the instruction)\n- Edited Image 2 (after applying the instruction)\n- Text Instruction\n\n### Evaluation Criteria for Each Image:\n\n1. **Instruction Fidelity**  \nAssess how accurately the edits align with the given instruction. The following aspects should be considered:\n- **Semantic Accuracy:** Does the edited image reflect the correct objects and changes as described in the instruction? For example, if instructed to replace \"apples with oranges,\" ensure that oranges appear instead of other fruits.\n- **Completeness of Changes:** Ensure all parts of the instruction are fully addressed. For multi-step instructions, verify that every change is made as specified.\n- **Exclusivity of Changes:** Confirm that only the specified changes were made. Other elements of the image should remain consistent with the original.\n\n2. **Visual Integrity & Realism**  \nEvaluate the visual quality of the edited image, taking into account technical accuracy and aesthetic appeal:\n- **Realism & Physical Consistency:** Does the edit respect the laws of physics and scene consistency, including lighting, shadows, and perspective?\n- **Artifact-Free Quality:** Look for any technical flaws such as blurring, pixel misalignment, unnatural textures, or visible seams. The image should be clean and free from distractions.\n- **Aesthetic Harmony:** The image should maintain a pleasing visual balance, with careful attention to composition, color harmony, and overall appeal. The changes should enhance the image rather than detract from it.\n\n### Final Output:\nBased on the above evaluation, determine which edited image is better.\n\nText instruction - {instruction}\n"

source_img = _encode_image('/path/to/image')
left_img =  _encode_image('/path/to/image')
right_img =  _encode_image('/path/to/image')

input_data.append({
    'problem': problem,
    'images': [source_img, left_img, right_img],
})

output = evaluate_batch(input_data, "http://localhost:8080")

print(item['model_output'])