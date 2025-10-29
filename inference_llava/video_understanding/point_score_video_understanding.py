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

warnings.filterwarnings("ignore")
pretrained = "CodeGoat24/UnifiedReward-7b"

def _load_video(video_path, num_video_frames, loader_fps, fps=None, frame_count=None):
        from torchvision import transforms

        from llava.mm_utils import opencv_extract_frames

        try:
            pil_imgs, frames_loaded = opencv_extract_frames(video_path, num_video_frames, loader_fps, fps, frame_count)
        except Exception as e:
            video_loading_succeed = False
            print(f"bad data path {video_path}")
            print(f"[DEBUG] Error processing {video_path}: {e}")
            empty_num_video_frames = int(random.uniform(2, num_video_frames))
            pil_imgs = [Image.new("RGB", (448, 448), (0, 0, 0))] * empty_num_video_frames
            frames_loaded = 0

        return pil_imgs, frames_loaded

model_name = "llava_qwen"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map)  # Add any other thing you want to pass in llava_model_args

model.eval()

Query = "" 
Response = "" 

video_path = '/path/to/video'

num_video_frames = 8
loader_fps = 0.0
fps = None
frame_count = None

images, frames_loaded = _load_video(
    video_path, num_video_frames, loader_fps, fps=fps, frame_count=frame_count
)

image_sizes = []
for img in images:
    img.resize((512, 512))
    image_sizes.append(img.size)

image_tensor = image_processor.preprocess(images, return_tensors="pt")["pixel_values"].cuda().bfloat16()

conv_template = "qwen_1_5"  # Make sure you use correct chat template for different models

question = '<image>\n'*len(images) + f'You are provided with a video and a question for this video. Please review the corresponding response based on the following 5 factors: 1. Accuracy in Object Description: Evaluate the accuracy of the descriptions concerning the objects mentioned in the ground truth answer. Responses should minimize the mention of objects not present in the ground truth answer, and inaccuracies in the description of existing objects. 2. Accuracy in Depicting Relationships: Consider how accurately the relationships between objects are described compared to the ground truth answer. Rank higher the responses that least misrepresent these relationships. 3. Accuracy in Describing Attributes: Assess the accuracy in the depiction of objects\' attributes compared to the ground truth answer. Responses should avoid inaccuracies in describing the characteristics of the objects present. 4. Helpfulness: Consider whether the generated text provides valuable insights, additional context, or relevant information that contributes positively to the user\'s comprehension of the video. Assess whether the language model accurately follows any specific instructions or guidelines provided in the prompt. Evaluate the overall contribution of the response to the user experience. 5. Ethical Considerations: - Identify if the model gives appropriate warnings or avoids providing advice on sensitive topics, such as medical videos. Ensure the model refrains from stating identification information in the video that could compromise personal privacy. Evaluate the language model\'s responses for fairness in treating individuals and communities, avoiding biases. Assess for harmfulness, ensuring the avoidance of content that may potentially incite violence, be classified as NSFW (Not Safe For Work), or involve other unmentioned ethical considerations. Consider any content that could be deemed offensive, inappropriate, or ethically problematic beyond the explicitly listed criteria.\nYou need to assign a score from 1 to 5 based on the above 5 factors for the response after \'Final Score:\'.\nYour task is provided as follows:\nQuestion: [{Query}]\nResponse: [{Response}]'


conv = copy.deepcopy(conv_templates[conv_template])
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()

input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

with torch.cuda.amp.autocast():
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
        
