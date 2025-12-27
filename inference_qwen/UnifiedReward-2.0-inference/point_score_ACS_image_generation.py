
from PIL import Image





from io import BytesIO
import base64
from vllm_qwen.vllm_request import evaluate_batch

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

input_data = []

prompt = ''

problem = (
    "You are presented with a generated image and its associated text caption. "
    "Your task is to analyze the image across multiple dimensions in relation to the caption. Specifically:\n"
    "Provide overall assessments for the image along the following axes (each rated from 1 to 5):\n"
    "- Alignment Score: How well the image matches the caption in terms of content.\n"
    "- Coherence Score: How logically consistent the image is (absence of visual glitches, object distortions, etc.).\n"
    "- Style Score: How aesthetically appealing the image looks, regardless of caption accuracy.\n\n"
    "Output your evaluation using the format below:\n\n"
    "Alignment Score (1-5): X\n"
    "Coherence Score (1-5): Y\n"
    "Style Score (1-5): Z\n\n"
    "Your task is provided as follows:\n"
    f"Text Caption: [{prompt}]"
)
image_path = ''

images = [image_path]

input_data.append({
    'problem': problem,
    'images': images
})

 
output = evaluate_batch(input_data, "http://localhost:8080", image_root=None)

print(output[0]['model_output'])

'''Example output:

Alignment Score (1-5): 2.694700002670288
Coherence Score (1-5): 2.371500015258789
Style Score (1-5): 2.802500009536743
'''
