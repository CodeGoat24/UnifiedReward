import base64
from io import BytesIO
import json
import os
import statistics
import time
import concurrent.futures
import random
import threading

from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from tqdm import tqdm


class VLMessageClient:
    def __init__(
        self,
        api_url,
        *,
        timeout_base=60,
        max_retries=10,
        backoff_base=2,
        backoff_cap=10,
        pool_maxsize=8,
        max_cache_items=1024,
    ):
        self.api_url = api_url
        self.timeout_base = timeout_base
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self.pool_maxsize = pool_maxsize
        self.max_cache_items = max_cache_items
        self._local = threading.local()
        self._cache_lock = threading.Lock()
        self._encode_cache = {}

    def _get_session(self):
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            adapter = HTTPAdapter(
                pool_connections=self.pool_maxsize,
                pool_maxsize=self.pool_maxsize,
            )
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            self._local.session = session
        return session

    def _encode_image(self, image):
        with Image.open(image) as img:
            img = img.convert("RGB")
            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _get_cached_base64(self, image_path):
        with self._cache_lock:
            cached = self._encode_cache.get(image_path)
        if cached is not None:
            return cached
        encoded = self._encode_image(image_path)
        with self._cache_lock:
            if len(self._encode_cache) >= self.max_cache_items:
                self._encode_cache.clear()
            self._encode_cache[image_path] = encoded
        return encoded

    def build_messages(self, item, image_root=None):
        content = []
        images = list(item.get("images", []))
        if image_root:
            images = [os.path.join(image_root, image) for image in images]

        for image in images:
            if os.path.exists(image):
                base64_image = self._get_cached_base64(image)
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

    def process_item(self, item, image_root):
        attempt = 0
        result = None
        start_time = time.monotonic()

        while attempt < self.max_retries:
            try:
                attempt += 1
                session = self._get_session()
                raw_messages = self.build_messages(item, image_root)

                payload = {
                    "model": "UnifiedReward",
                    "messages": raw_messages,
                    "do_sample": False,
                    "max_tokens": 4096,
                }

                response = session.post(
                    f"{self.api_url}/v1/chat/completions",
                    json=payload,
                    timeout=self.timeout_base + attempt * 5,
                )
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"Retryable HTTP {response.status_code}",
                        response=response,
                    )
                response.raise_for_status()

                output = response.json()["choices"][0]["message"]["content"]

                item["model_output"] = output
                item["success"] = True
                result = item

                break

            except Exception as e:
                if attempt == self.max_retries:
                    print(f"请求失败（已达最大重试次数）: {str(e)}")
                    result = {
                        "idx": item.get("idx"),
                        "question": item["problem"],
                        "image_path": item.get("images", []),
                        "error": str(e),
                        "attempt": attempt,
                        "success": False
                    }
                else:
                    sleep_time = min(
                        self.backoff_base ** attempt + random.uniform(0, 1),
                        self.backoff_cap,
                    )
                    time.sleep(sleep_time)
        if result is None:
            result = {
                "idx": item.get("idx"),
                "question": item.get("problem"),
                "image_path": item.get("images", []),
                "error": "empty result",
                "attempt": attempt,
                "success": False
            }
        result["elapsed"] = round(time.monotonic() - start_time, 3)
        return result, result.get("success", False) if result else False


def evaluate_batch(
    batch_data,
    api_url,
    image_root=None,
    output_file="./results.json",
    error_file=None,
    max_workers=None,
    max_retries=10,
    timeout_base=160,
    backoff_base=2,
    backoff_cap=10,
    flush_every=20,
    log_stats=None,
):
    success_count = 0
    total_result = []
    durations = []
    output_buffer = []
    error_buffer = []

    if max_workers is None:
        max_workers = int(os.getenv("VLLM_MAX_WORKERS", "16"))
    max_workers = max(1, min(max_workers, len(batch_data)))

    client = VLMessageClient(
        api_url,
        timeout_base=timeout_base,
        max_retries=max_retries,
        backoff_base=backoff_base,
        backoff_cap=backoff_cap,
        pool_maxsize=max_workers,
    )

    if log_stats is None:
        log_stats = os.getenv("VLLM_LOG_STATS", "0") == "1"

    def flush_buffer(buffer, path):
        if not buffer or not path:
            return
        with open(path, "a", encoding="utf-8") as f:
            for row in buffer:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        buffer.clear()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
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
                )
            )
        with tqdm(total=len(batch_data), desc="推理进度") as pbar:
            for future in concurrent.futures.as_completed(futures):
                try:
                    result, _ = future.result()
                    total_result.append(result)
                    if result and result.get("elapsed") is not None:
                        durations.append(result["elapsed"])
                    output_buffer.append(result)
                    if result and result.get("success"):
                        success_count += 1
                    else:
                        error_buffer.append(result)
                except Exception as e:
                    print(f"任务异常: {str(e)}")
                finally:
                    pbar.update(1)
                    processed_info = f"{success_count}/{len(batch_data)}"
                    pbar.set_postfix({
                        "processed": processed_info
                    })
                    if len(output_buffer) >= flush_every:
                        flush_buffer(output_buffer, output_file)
                    if error_file and len(error_buffer) >= flush_every:
                        flush_buffer(error_buffer, error_file)

    flush_buffer(output_buffer, output_file)
    if error_file:
        flush_buffer(error_buffer, error_file)

    total_result.sort(
        key=lambda x: int(x["idx"]) if x and x.get("idx") is not None else -1
    )

    if log_stats and durations:
        durations_sorted = sorted(durations)
        count = len(durations_sorted)
        p50 = durations_sorted[int(0.50 * (count - 1))]
        p95 = durations_sorted[int(0.95 * (count - 1))]
        mean = statistics.mean(durations_sorted)
        print(
            f"\nLatency stats (s): mean={mean:.2f} p50={p50:.2f} p95={p95:.2f}"
        )

    return total_result


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", default="http://localhost:8080")
    parser.add_argument("--prompt_path", required=True)
    parser.add_argument("--image_root", default=None)
    parser.add_argument("--output_path", default="./results.json")
    parser.add_argument("--error_path", default=None)
    parser.add_argument("--max_workers", type=int, default=None)
    parser.add_argument("--max_retries", type=int, default=10)
    parser.add_argument("--timeout_base", type=int, default=60)
    args = parser.parse_args()

    with open(args.prompt_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    open(args.output_path, "w", encoding="utf-8").close()
    error_path = args.error_path
    if error_path is None:
        error_path = f"{args.output_path}.errors.jsonl"
    open(error_path, "w", encoding="utf-8").close()
    results = evaluate_batch(
        test_data,
        args.api_url,
        image_root=args.image_root,
        output_file=args.output_path,
        error_file=error_path,
        max_workers=args.max_workers,
        max_retries=args.max_retries,
        timeout_base=args.timeout_base,
    )

    success_count = sum(1 for item in results if item and item.get("success"))
    print("\nStatics:")
    print(f"Total data: {len(test_data)}")
    print(f"Success ratio: {success_count} ({success_count/len(test_data):.2%})")


if __name__ == "__main__":
    main()
