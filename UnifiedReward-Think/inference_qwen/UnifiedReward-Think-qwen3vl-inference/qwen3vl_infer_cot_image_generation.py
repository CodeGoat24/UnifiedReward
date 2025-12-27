
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

problem = f'''You are an objective and precise evaluator for image quality comparison. I will provide you with a text caption and two images generated based on this caption. You must analyze the two images carefully and determine which image is better.

        Evaluation procedure:

        1. The caption for the generated images is: 「{prompt}」. You must evaluate the two images across these core dimensions:
        - Semantic consistency (how closely the image content aligns with the caption)
        - Aesthetics (composition, color usage, artistic expression)
        - Authenticity (realism and attention to detail)

        2. You are also encouraged to add up to two additional evaluation dimensions if they are relevant to the specific caption or images (e.g., creativity, spatial layout, fine-grained detail). If no extra dimensions are relevant, just keep the three core dimensions.

        3. For each evaluation dimension:
        - Provide a score between 1–10 for both Image 1 and Image 2
        - Provide a short rationale for each score (2–5 short sentences)
        - The evaluation must follow exactly this format with line breaks and indentation:
            Dimension name: 
                Image 1 (x/10) - rationale; 
                Image 2 (y/10) - rationale

        4. After evaluating all dimensions, calculate the total score for each image and show the calculation explicitly, following this exact format:
            Total score:
            Image 1: x+x+x=total
            Image 2: y+y+y=total

        5. Wrap all reasoning and scoring strictly within <think> and </think> tags.

        6. After </think>, output the final judgment strictly inside <answer> and </answer> tags, containing only one of:
        - Image 1 is better
        - Image 2 is better

        Constraints:
        - You must strictly follow the line breaks, indentation, and formatting shown in the example below.
        - Do not merge multiple dimensions into one line. Each dimension must follow the 3-line block format shown below.
        - Do not use Markdown formatting, bullet points, bold text, or headings.
        - Do not output explanations outside <think> and <answer>.
        - The <answer> tag must contain only the final string with no extra words.

        Required output format:

        <think>
        1. Semantic consistency: 
            Image 1 (9/10) - ...; 
            Image 2 (7/10) - ...
        2. Aesthetics: 
            Image 1 (8/10) - ...; 
            Image 2 (8/10) - ...
        3. Authenticity: 
            Image 1 (8/10) - ...; 
            Image 2 (5/10) - ...
        [Additional dimension if any]: 
            Image 1 (7/10) - ...; 
            Image 2 (8/10) - ...
        [Additional dimension if any]: 
            Image 1 (6/10) - ...; 
            Image 2 (7/10) - ...
        Total score:
        Image 1: 9+8+8+7+6=38
        Image 2: 7+8+5+8+7=35
        </think>
        <answer>Image 1 is better</answer>

        Note: The example above is only to illustrate the exact format (line breaks, indentation, symbols, and style). Your actual evaluation must follow this format exactly, but be based on the given caption and images.
        '''

left_image = Image.open(image_path_1)
right_image = Image.open(image_path_2)

left_image = _encode_image(left_image)
right_image = _encode_image(right_image)

input_data = [{
    'problem': problem,
    'images': [left_image, right_image]
}]

output = evaluate_batch(input_data, "http://localhost:8080")[0]['model_output']

print(output)