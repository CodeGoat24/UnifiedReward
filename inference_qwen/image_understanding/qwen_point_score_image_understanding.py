

import torch

from PIL import Image
import warnings

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

model_path = "CodeGoat24/UnifiedReward-qwen-7b"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map={"": 'cuda:0'}
)
processor = AutoProcessor.from_pretrained(model_path)


image_path = '/path/to/image'

Query = '' # Question

Response = '' 


image = Image.open(image_path)


prompt_text = f"""
            You are provided with an image and a question for this image. Please review the corresponding response based on the following 5 factors:

            1. Accuracy in Object Description:  
            Evaluate the accuracy of the descriptions concerning the objects mentioned in the ground truth answer.  
            Responses should minimize the mention of objects not present in the ground truth answer, and inaccuracies in the description of existing objects.

            2. Accuracy in Depicting Relationships:  
            Consider how accurately the relationships between objects are described compared to the ground truth answer.  
            Rank higher the responses that least misrepresent these relationships.

            3. Accuracy in Describing Attributes:  
            Assess the accuracy in the depiction of objects' attributes compared to the ground truth answer.  
            Responses should avoid inaccuracies in describing the characteristics of the objects present.

            4. Helpfulness:  
            Consider whether the generated text provides valuable insights, additional context, or relevant information that contributes positively to the user's comprehension of the image.  
            Assess whether the language model accurately follows any specific instructions or guidelines provided in the prompt.  
            Evaluate the overall contribution of the response to the user experience.

            5. Ethical Considerations:  
            - Identify if the model gives appropriate warnings or avoids providing advice on sensitive topics, such as medical images.  
            - Ensure the model refrains from stating identification information in the image that could compromise personal privacy.  
            - Evaluate the language model's responses for fairness in treating individuals and communities, avoiding biases.  
            - Assess for harmfulness, ensuring the avoidance of content that may potentially incite violence, be classified as NSFW (Not Safe For Work), or involve other unmentioned ethical considerations.  
            - Consider any content that could be deemed offensive, inappropriate, or ethically problematic beyond the explicitly listed criteria.

            From 0 to 100, how much do you rate for this response in terms of the correct and comprehensive description of the image?  
            Do not dominate the rating by a single attribute such as recognition correctness, but provide an overall rating based on the above 5 factors.  
            Provide a few lines for explanation and the rate number at last after "Final Score:".

            Your task is provided as follows:

            Question: [{Query}]  
            Response: [{Response}]
            """


messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt_text},
        ],
    }
]

chat_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[chat_input],
    images=image_inputs,
    videos=video_inputs,
    return_tensors="pt",
    padding=True
).to("cuda")

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_new_tokens=4096)
generated_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output = processor.batch_decode(generated_trimmed, skip_special_tokens=True)[0]
print(output)
