





import warnings

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

warnings.filterwarnings("ignore")

model_path = 'CodeGoat24/UnifiedReward-qwen-7b'
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map={"": 'cuda:0'}
)
processor = AutoProcessor.from_pretrained(model_path)

prompt = ""

video_path_1 = "/path/to/video1"
video_path_2 = "/path/to/video2"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": video_path_1},
            {"type": "video", "video": video_path_2},
            {"type": "text", "text": (
                "Imagine you are an expert tasked with evaluating AI-generated videos. You are provided with a text caption and two videos generated based on that caption. Your job is to assess and compare these videos based on the following two main factors:\n\n"
                "1. Caption Alignment: Evaluate how closely each video matches the description provided in the caption. Pay attention to the accuracy of objects depicted, their relationships, and any attributes described in the caption.\n\n"
                "2. Overall Video Quality: Look at the overall visual appeal of each video, considering clarity, the level of detail, color accuracy, and how aesthetically pleasing the video is.\n\n"
                "Using these factors, compare the two videos and determine which one better reflects the caption and exhibits better visual quality.\n\n"
                "Give your final judgment, such as 'Video 1 is better,' 'Video 2 is better,' or 'Both videos are equally good.'\n\n"
                "Your task is as follows:\n"
                f"Text Caption: [{prompt}]\n"
            )}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt", **video_kwargs).to(model.device)

generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0]

print(output_text)
