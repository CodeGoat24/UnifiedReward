import random
import tqdm
from PIL import Image
import warnings
from datasets import load_dataset
from io import BytesIO
import base64

from vllm_request import evaluate_batch

warnings.filterwarnings("ignore")


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


dataset = load_dataset("MMInstruction/VL-RewardBench")["test"]


group_correct = {"general": 0, "hallucination": 0, "reasoning": 0}
group_total = {"general": 0, "hallucination": 0, "reasoning": 0}
group_mapping = {
    "vlfeedback": "general",
    "povid": "hallucination",
    "reasoning_tasks": "reasoning",
    "rlhf-v": "hallucination",
    "rlaif-v": "hallucination",
    "wildvision-battle": "general",
}

correct = 0
random.seed(0)


batched_input = []
gold_answers = []          
group_types = []         
id_list = []                 

for i in tqdm.trange(len(dataset)):
    data = dataset[i]

    image = data["image"]

    if data["human_ranking"][0] == 0:
        if random.random() < 0.5:
            R1, R2, answer = data["response"][0], data["response"][1], "Answer 1 is better"
        else:
            R1, R2, answer = data["response"][1], data["response"][0], "Answer 2 is better"
    else:
        if random.random() < 0.5:
            R1, R2, answer = data["response"][0], data["response"][1], "Answer 2 is better"
        else:
            R1, R2, answer = data["response"][1], data["response"][0], "Answer 1 is better"

    Query = data["query"]

    # UnifiedReward-Think-qwen3vl Models Input Template for Image Understanding
    problem = f"You are an objective and precise evaluator for image-based question answering. I will provide you with a question, a reference image, and two candidate answers. You must analyze the two answers carefully and determine which one is better.\n\n        Instructions (MUST follow strictly):\n        1. All reasoning, analysis, explanations, and scores MUST be written strictly inside <think> and </think> tags. \n        2. The <think> block must start immediately with the first evaluation dimension. Do NOT include any introduction, notes, or explanations before the first numbered dimension.\n        3. After </think>, output the final judgment strictly inside <answer> and </answer> tags, containing only one of:\n        - Answer 1 is better\n        - Answer 2 is better\n        4. Do NOT output anything outside <think> and <answer>. No extra explanations, notes, or prefaces.\n\n        Evaluation procedure:\n        1. The question is: \u300c{Query}\u300d. The reference image is provided. The two candidate answers are:\n        Answer 1: \u300c{R1}\u300d\nAnswer 2: \u300c{R2}\u300d\n\n2. You must evaluate the two answers across these core dimensions:\n        - Semantic accuracy (how well the answer reflects the visual content of the image)\n        - Correctness (whether the answer is logically and factually correct)\n        - Clarity (whether the answer is clearly and fluently expressed)\n\n        3. You are also encouraged to add up to two additional evaluation dimensions if they are relevant (e.g., reasoning ability, attention to detail, multimodal grounding). If no extra dimensions are relevant, just keep the three core dimensions.\n\n        4. For each evaluation dimension:\n        - Provide a score between 1\u201310 for both Answer 1 and Answer 2\n        - Provide a short rationale for each score (2\u20135 short sentences)\n        - Each dimension must follow exactly this 3-line block format with numbering, line breaks, and indentation:\n            N. Dimension name: \n                Answer 1 (x/10) - rationale; \n                Answer 2 (y/10) - rationale\n\n        5. After evaluating all dimensions, calculate the total score for each answer and show the calculation explicitly, following this exact format:\n            Total score:\n            Answer 1: x+x+x=total\n            Answer 2: y+y+y=total\n\n        Required output format:\n\n        <think>\n        1. Semantic accuracy: \n            Answer 1 (9/10) - ...; \n            Answer 2 (7/10) - ...\n        2. Correctness: \n            Answer 1 (8/10) - ...; \n            Answer 2 (7/10) - ...\n        3. Clarity: \n            Answer 1 (9/10) - ...; \n            Answer 2 (8/10) - ...\n        [Additional dimension if any]: \n            Answer 1 (6/10) - ...; \n            Answer 2 (7/10) - ...\n        [Additional dimension if any]: \n            Answer 1 (9/10) - ...; \n            Answer 2 (7/10) - ...\n        Total score:\n        Answer 1: 9+8+9+6+9=41\n        Answer 2: 7+7+8+7+7=36\n        </think>\n        <answer>Answer 1 is better</answer>\n\n        Note: The example above is only to illustrate the exact format (numbering, line breaks, indentation, and style). Your actual evaluation must follow this format exactly, but be based on the given question, reference image, and candidate answers.\n        "


    batched_input.append({
        "problem": problem,
        "images": [_encode_image(image)]
    })

    gold_answers.append(answer)
    id_list.append(data["id"])



outputs = evaluate_batch(
    batched_input,
    "http://localhost:8080",
    image_root=None
)


for i, output_data in enumerate(outputs):
    output = output_data["model_output"]
    answer = gold_answers[i]
    id_value = id_list[i]

    # Determine group type
    split_index = min((id_value.find("_"), id_value.find("-")),
                      key=lambda x: x if x != -1 else float("inf"))
    id_prefix = id_value[:split_index] if split_index != -1 else id_value
    dtype = {
        "RLAIF": "rlaif-v",
        "RLHF": "rlhf-v",
        "mathverse": "reasoning_tasks",
        "mmmu": "reasoning_tasks",
        "wildvision": "wildvision-battle"
    }.get(id_prefix, "vlfeedback")
    group = group_mapping[dtype]

    group_total[group] += 1

    if answer in output:
        correct += 1
        group_correct[group] += 1
    else:
        print("❌ Wrong output:")
        print("Model:", output)
        print("GT:", answer)


accuracy = correct / len(dataset)
print(f"\n==== FINAL RESULTS ====")
print(f"Acc.: {correct}/{len(dataset)} = {accuracy:.4f}")

task_list = ["reasoning", "hallucination", "general"]
macro_average = sum(group_correct[k] / group_total[k] for k in task_list) / 3

print("reasoning:", group_correct["reasoning"] / group_total["reasoning"])
print("hallucination:", group_correct["hallucination"] / group_total["hallucination"])
print("general:", group_correct["general"] / group_total["general"])
print("macro:", macro_average)
