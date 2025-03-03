## ENV
```cmd
conda create -n vllm python=3.12 -y
conda activate vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 pip install --editable .
```

## Dowdload Model
```cmd
mkdir vlm_models
huggingface-cli download Qwen/Qwen2-VL-2B-Instruct --local-dir vlm_models/Qwen/Qwen2-VL-2B-Instruct
```

## Run
```bash
CUDA_VISIBLE_DEVICES=2 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-3B-Instruct-AWQ \
    --download-dir /other/vlm_models \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 8192 \
    --trust-remote-code
```

```bash
CUDA_VISIBLE_DEVICES=3 python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
    --download-dir /other/vlm_models \
    --host 0.0.0.0 \
    --port 8001 \
    --dtype float16 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 8192 \
    --trust-remote-code
```

## Test
```cmd
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-VL-3B-Instruct-AWQ",
        "messages": [
            {"role": "user", "content": "你好,最近怎么样?"}
        ],
        "max_tokens": 100
    }'
```

## Change Yaml Setting
```
LangCoop/vlmdrive/hypes_yaml/api_vlm_drive.yaml
```

```yaml
  api_model_name: Qwen/Qwen2-VL-2B-Instruct
  api_base_url: http://localhost:8000/v1
  api_key: dummy_key
```