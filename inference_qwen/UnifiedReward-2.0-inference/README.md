# 😊 vLLM Setup and Usage Guide

## 🔧 1. Create a new vLLM environment and install required packages

```
conda create -n vllm python=3.11 -y
conda activate vllm

pip install vllm>=0.11.0

pip install qwen-vl-utils==0.0.14
```

## 💪 2. Start the vLLM service in a new window (e.g., tmux)

```
tmux new -s vllm

conda activate vllm
cd inference_qwen/UnifiedReward-2.0-inference/

bash vllm_server.sh
```

## 🚀 3. Send a request in a new window

```
tmux new -s infer
conda activate <infer_env>
cd inference_qwen/UnifiedReward-2.0-inference/

python point_score_ACS_image_generation.py
```
