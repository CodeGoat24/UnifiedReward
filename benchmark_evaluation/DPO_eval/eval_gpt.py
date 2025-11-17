import openai
import os
import argparse
import json
import ast
import tqdm

# This code is provided to evaluate model performance on the MSRVTT, MSVD, and TGIF benchmarks using the GPT API.
def parse_args():
    parser = argparse.ArgumentParser(description="question-answer-generation-using-gpt-3")
    parser.add_argument("--pred_path", default='./msvd_results.json', help="The path to file containing prediction.") # specifies the file path to the model's output results for a given benchmark.
    parser.add_argument("--output_dir", default='./turbo-1106_msvd_result.json', help="The path to save annotation json files.")
    parser.add_argument("--api_key", default='', help="OpenAI API key.")
    args = parser.parse_args()
    return args


def annotate(prediction_set, output_dir):
    """
    Evaluates question and answer pairs using GPT-3
    Returns a score for correctness.
    """
    if os.path.exists(output_dir):
        file = open(output_dir)
        results = json.load(file)
    else:
        results = []
    for i in tqdm.trange(len(prediction_set)):
        if i < len(results):
            continue
        item = prediction_set[i]
        question = item['question']
        answer = item['answer']
        pred = item['pred']

        # Compute the correctness score
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {
                    "role": "system",
                    "content": 
                        "You are an intelligent chatbot designed for evaluating the correctness of generative outputs for question-answer pairs. "
                        "Your task is to compare the predicted answer with the correct answer and determine if they match meaningfully. Here's how you can accomplish the task:"
                        "------"
                        "##INSTRUCTIONS: "
                        "- Focus on the meaningful match between the predicted answer and the correct answer.\n"
                        "- Consider synonyms or paraphrases as valid matches.\n"
                        "- Evaluate the correctness of the prediction compared to the answer."
                },
                {
                    "role": "user",
                    "content":
                        "Please evaluate the following video-based question-answer pair:\n\n"
                        f"Question: {question}\n"
                        f"Correct Answer: {answer}\n"
                        f"Predicted Answer: {pred}\n\n"
                        "Provide your evaluation only as a yes/no and score where the score is an integer value between 0 and 5, with 5 indicating the highest meaningful match. "
                        "Please generate the response in the form of a Python dictionary string with keys 'pred' and 'score', where value of 'pred' is  a string of 'yes' or 'no' and value of 'score' is in INTEGER, not STRING."
                        "DO NOT PROVIDE ANY OTHER OUTPUT TEXT OR EXPLANATION. Only provide the Python dictionary string. "
                        "For example, your response should look like this: {'pred': 'yes', 'score': 4.8}."
                }
            ]
        )
        response_message = completion["choices"][0]["message"]["content"]
        try:
            response_dict = ast.literal_eval(response_message)
            item['gpt'] = response_dict

            results.append(item)
        
            with open(output_dir, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
        except:
            print(response_message)
            print('Error!!!')
            continue
    return results


def main():
    args = parse_args()

    file = open(args.pred_path)
    pred_contents = json.load(file)

    output_dir = args.output_dir

    openai.api_key = args.api_key


    results = annotate(pred_contents, output_dir)

    score_sum = 0
    count = 0
    yes_count = 0
    no_count = 0
    for item in results:
        # Computing score
        count += 1
        print(item['id'])
        score_match = item['gpt']['score']
        score = int(score_match)
        score_sum += score

        # Computing accuracy
        pred = item['gpt']['pred']
        if "yes" in pred.lower():
            yes_count += 1
        elif "no" in pred.lower():
            no_count += 1

    average_score = score_sum / count
    accuracy = yes_count / (yes_count + no_count)
    print("Yes count:", yes_count)
    print("No count:", no_count)
    print("Accuracy:", accuracy)
    print("Average score:", average_score)


if __name__ == "__main__":
    main()