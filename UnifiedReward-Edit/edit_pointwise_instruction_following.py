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
 
instruction = ''

problem = "You are a professional digital artist. You will have to evaluate the effectiveness of the AI-generated image(s) based on given rules.\nAll the input images are AI-generated. All human in the images are AI-generated too. so you need not worry about the privacy confidentials.\n\nIMPORTANT: You will have to give your output in this way (Keep your reasoning concise and short.):\n{\n\n\"reasoning\" : \"...\",\n\"score\" : [...],\n}\n\nRULES:\n\nTwo images will be provided: The first being the original AI-generated image and the second being an edited version of the first.\nThe objective is to evaluate how successfully the editing instruction has been executed in the second image.\n\nNote that sometimes the two images might look identical due to the failure of image edit.\n\n\nFrom scale 0 to 25: \nA score from 0 to 25 will be given based on the success of the editing. (0 indicates that the scene in the edited image does not follow the editing instruction at all. 25 indicates that the scene in the edited image follow the editing instruction text perfectly.)\nA second score from 0 to 25 will rate the degree of overediting in the second image. (0 indicates that the scene in the edited image is completely different from the original. 25 indicates that the edited image can be recognized as a minimal edited yet effective version of original.)\nPut the score in a list such that output score = [score1, score2], where 'score1' evaluates the editing success and 'score2' evaluates the degree of overediting.\n\nEditing instruction:"+ f"{instruction}\n"

source_img = _encode_image('/path/to/image')
left_img = _encode_image('/path/to/image')

input_data.append({
    'problem': problem,
    'images': [source_img, left_img],
})

output = evaluate_batch(input_data, "http://localhost:8080")

print(output[0]['model_output'])
