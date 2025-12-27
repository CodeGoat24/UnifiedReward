import base64
from io import BytesIO
import json
import os
import time
from multiprocessing import Manager
import concurrent.futures

from PIL import Image
import requests
from tqdm import tqdm


class VLMessageClient:
    def __init__(self, api_url):
        self.api_url = api_url
        self.session = requests.Session()

    def _encode_image(self, image):
        with Image.open(image) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def build_messages(self, item, image_root=None):
        content = []
        images = list(item.get("images", []))
        if image_root:
            images = [os.path.join(image_root, image) for image in images]

        for image in images:
            if os.path.exists(image):
                base64_image = self._encode_image(image)
                image_url = f"data:image/jpeg;base64,{base64_image}"
            elif image.startswith(("http://", "https://")):
                image_url = image
            else:
                image_url = f"data:image/jpeg;base64,{image}"
            content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })

        content.append({"type": "text", "text": item["problem"]})

        return [
            {
                "role": "user",
                "content": content
            }
        ]

    def process_item(self, item, image_root, output_file, total_counter, lock, max_retries=10):
        attempt = 0
        result = None

        while attempt < max_retries:
            try:
                attempt += 1
                raw_messages = self.build_messages(item, image_root)

                payload = {
                    "model": "UnifiedReward",
                    "messages": raw_messages,
                    "do_sample": False,
                    "max_tokens": 4096,
                }

                response = self.session.post(
                    f"{self.api_url}/v1/chat/completions",
                    json=payload,
                    timeout=30 + attempt * 5
                )
                response.raise_for_status()

                output = response.json()["choices"][0]["message"]["content"]

                with lock:
                    total_counter.value += 1

                item["model_output"] = output
                item["success"] = True
                result = item

                break

            except Exception as e:
                if attempt == max_retries:
                    print(f"请求失败（已达最大重试次数）: {str(e)}")
                    result = {
                        "question": item["problem"],
                        "image_path": item.get("images", []),
                        "error": str(e),
                        "attempt": attempt,
                        "success": False
                    }
                else:
                    sleep_time = min(2 ** attempt, 10)
                    time.sleep(sleep_time)
        if result:
            with lock:
                with open(output_file, "a") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
        return result, result.get("success", False) if result else False


def evaluate_batch(batch_data, api_url, image_root=None, output_file="./results.json", max_workers=128, max_retries=10):
    with Manager() as manager:
        total_counter = manager.Value("i", 0)
        lock = manager.Lock()
        total_result = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            client = VLMessageClient(api_url)
            index = 0
            for item in batch_data:
                if "idx" not in item:
                    item["idx"] = str(index)
                    index += 1
                futures.append(
                    executor.submit(
                        client.process_item,
                        item=item,
                        image_root=image_root,
                        output_file=output_file,
                        total_counter=total_counter,
                        lock=lock,
                        max_retries=max_retries
                    )
                )
            with tqdm(total=len(batch_data), desc="推理进度") as pbar:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        result, _ = future.result()
                        total_result.append(result)
                    except Exception as e:
                        print(f"任务异常: {str(e)}")
                    finally:
                        pbar.update(1)
                        current_total = total_counter.value
                        processed_info = f"{current_total}/{len(batch_data)}"
                        pbar.set_postfix({
                            "processed": processed_info
                        })

    total_result.sort(key=lambda x: int(x["idx"]))

    return total_result


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", default="http://localhost:8080")
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--output_path", default="./results.json")
    parser.add_argument("--max_workers", type=int, default=128)
    parser.add_argument("--max_retries", type=int, default=10)
    args = parser.parse_args()

    with open(args.prompt_path, "r") as f:
        test_data = json.load(f)

    open(args.output_path, "w").close()
    results = evaluate_batch(
        test_data,
        args.api_url,
        image_root=args.image_root,
        output_file=args.output_path,
        max_workers=args.max_workers,
        max_retries=args.max_retries
    )

    success_count = sum(1 for item in results if item and item.get("success"))
    print("\nStatics:")
    print(f"Total data: {len(test_data)}")
    print(f"Success ratio: {success_count} ({success_count/len(test_data):.2%})")


if __name__ == "__main__":
    main()
