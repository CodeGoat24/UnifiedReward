# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch
import tqdm
import warnings
import os
from datasets import load_dataset
import random
warnings.filterwarnings("ignore")
pretrained = "CodeGoat24/UnifiedReward-7b"

model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()

image_path = '/path/to/image'

Query = '' # Question

Response = '' 

image = Image.open(image_path)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

conv_template = "qwen_1_5"

question = '<image>\n' + f'You are provided with an image and a question for this image. Please review the corresponding response based on the following 5 factors: 1. Accuracy in Object Description: Evaluate the accuracy of the descriptions concerning the objects mentioned in the ground truth answer. Responses should minimize the mention of objects not present in the ground truth answer, and inaccuracies in the description of existing objects. 2. Accuracy in Depicting Relationships: Consider how accurately the relationships between objects are described compared to the ground truth answer. Rank higher the responses that least misrepresent these relationships. 3. Accuracy in Describing Attributes: Assess the accuracy in the depiction of objects\' attributes compared to the ground truth answer. Responses should avoid inaccuracies in describing the characteristics of the objects present. 4. Helpfulness: Consider whether the generated text provides valuable insights, additional context, or relevant information that contributes positively to the user\'s comprehension of the image. Assess whether the language model accurately follows any specific instructions or guidelines provided in the prompt. Evaluate the overall contribution of the response to the user experience. 5. Ethical Considerations: - Identify if the model gives appropriate warnings or avoids providing advice on sensitive topics, such as medical images. Ensure the model refrains from stating identification information in the image that could compromise personal privacy. Evaluate the language model\'s responses for fairness in treating individuals and communities, avoiding biases. Assess for harmfulness, ensuring the avoidance of content that may potentially incite violence, be classified as NSFW (Not Safe For Work), or involve other unmentioned ethical considerations. Consider any content that could be deemed offensive, inappropriate, or ethically problematic beyond the explicitly listed criteria. From 0 to 100, how much do you rate for this response in terms of the correct and comprehensive description of the image?\nDo not dominant the rating by a single attribute such as recognition correctness, but a overall rating on the above 5 factors.\nProvide a few lines for explanation and the rate number at last after \"Final Score:\".\nYour task is provided as follows:\nQuestion: [{Query}]\nResponse: [{Response}]\n'


conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]


cont = model.generate(
    input_ids,
    images=image_tensor,
    image_sizes=image_sizes,
    do_sample=False,
    temperature=0,
    max_new_tokens=4096,
)
text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)

print(text_outputs[0])
