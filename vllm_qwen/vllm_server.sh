vllm serve CodeGoat24/UnifiedReward-2.0-qwen3vl-8b \
    --host /ip/address \
    --trust-remote-code \
    --served-model-name UnifiedReward \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 4 \
    --pipeline-parallel-size 1 \
    --limit-mm-per-prompt.image 2 \
    --port 8080
