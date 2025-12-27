from io import BytesIO
import base64

from PIL import Image
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

# instruction = ''

problem = "You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.\nAll the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.\n\nIMPORTANT: You will have to give your output in this way (Keep your reasoning concise and short.):\n{\n\n\"reasoning\" : \"...\",\n\"score\" : [...],\n}\n\nRULES:\n\nThe image is an AI-generated image.\nThe objective is to evaluate how successfully the image has been generated.\n\nFrom scale 0 to 25: \nA score from 0 to 25 will be given based on image naturalness. \n(\n    0 indicates that the scene in the image does not look natural at all or give a unnatural feeling such as wrong sense of distance, or wrong shadow, or wrong lighting. \n    25 indicates that the image looks natural.\n)\nA second score from 0 to 25 will rate the image artifacts. \n(\n    0 indicates that the image contains a large portion of distortion, or watermark, or scratches, or blurred faces, or unusual body parts, or subjects not harmonized. \n    25 indicates the image has no artifacts.\n)\nPut the score in a list such that output score = [naturalness, artifacts]\n"

image = _encode_image('/path/to/image')

input_data.append({
    'problem': problem,
    'images': [image],
})

output = evaluate_batch(input_data, "http://localhost:8080")

print(output[0]['model_output'])
