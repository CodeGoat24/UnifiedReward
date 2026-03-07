vllm serve CodeGoat24/UnifiedReward-Think-qwen3vl-8b \
    --host localhost \
    --trust-remote-code \
    --served-model-name UnifiedReward \
    --gpu-memory-utilization 0.9 \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 1 \
    --limit-mm-per-prompt.image 32 \
    --port 8080 \
    --enable-prefix-caching \
    --disable-log-requests \
    --mm_processor_cache_gb=500

# Qwen3.5
export VLLM_DISABLE_FLASHINFER_GDN_PREFILL=1
export TOKENIZERS_PARALLELISM=false
vllm serve CodeGoat24/UnifiedReward-Think-qwen35-9b \
 --host localhost \
 --port 8080 \
 --trust-remote-code \
 --served-model-name UnifiedReward \
 --gpu-memory-utilization 0.95 \
 --mm-encoder-tp-mode data \
 --mm-processor-cache-type shm \
 --enable-prefix-caching \
 --tensor-parallel-size 8 \
 --default-chat-template-kwargs '{"enable_thinking": false}'