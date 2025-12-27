





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

video_path = "/path/to/video"

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": f"Suppose you are an expert in judging and evaluating the quality of AI-generated videos, please watch the frames of a given video and see the text prompt for generating the video.\nThen give scores from 5 different dimensions:\n(1) visual quality: the quality of the video in terms of clearness, resolution, brightness, and color\n(2) temporal consistency, the consistency of objects or humans in video\n(3) dynamic degree, the degree of dynamic changes\n(4) text-to-video alignment, the alignment between the text prompt and the video content\n(5) factual consistency, the consistency of the video content with the common-sense and factual knowledge\n\nFor each dimension, output a number from [1,2,3,4], \nin which \'1\' means \'Bad\', \'2\' means \'Average\', \'3\' means \'Good\', \n\'4\' means \'Real\' or \'Perfect\' (the video is like a real video)\nFinally, based on above 5 dimensions, assign a score from 1 to 10 after \'Final Score:\'\nHere is an output example:\nvisual quality: 4\ntemporal consistency: 4\ndynamic degree: 3\ntext-to-video alignment: 1\nfactual consistency: 2\nFinal Score: 6\n\n**Note: In the example above, scores are placeholders meant only to demonstrate the format. Your actual evaluation should be based on the quality of the given video.**\nYour task is provided as follows: Text Prompt: [{prompt}]"}
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
