#!/bin/bash
set -e

echo "============================================"
echo "  Bright Masker — RunPod Startup"
echo "============================================"

if [ -f /app/.env ]; then
  export $(grep -v '^#' /app/.env | xargs)
fi

MODEL_NAME="${MODEL_DEPLOYED_NAME:-Qwen/Qwen3-8B-Instruct}"
VLLM_PORT="${VLLM_PORT:-8001}"
APP_PORT="${PORT:-8000}"
GPU_MEM="${VLLM_GPU_UTIL:-0.75}"
MAX_CTX="${VLLM_MAX_MODEL_LEN:-8192}"

echo "  LLM model   : $MODEL_NAME"
echo "  vLLM port   : $VLLM_PORT"
echo "  App port    : $APP_PORT"
echo ""

# Check if model is already in HuggingFace cache
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}/hub"
MODEL_SLUG=$(echo "$MODEL_NAME" | tr '/' '--')
if ls "$HF_CACHE"/models--"$MODEL_SLUG" > /dev/null 2>&1; then
  echo "[1/3] Model found in cache — skipping download"
else
  echo "[1/3] Model not in cache — will download on first load (~5GB, ~2min on RunPod)"
fi

# Start vLLM
echo "[2/3] Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$VLLM_PORT" \
  --max-model-len "$MAX_CTX" \
  --gpu-memory-utilization "$GPU_MEM" \
  --disable-log-requests \
  --dtype bfloat16 \
  > /var/log/vllm.log 2>&1 &

VLLM_PID=$!
echo "  vLLM PID: $VLLM_PID"

# Wait for vLLM to be ready
WAITED=0
until curl -sf "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; do
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "ERROR: vLLM process died. Logs:"
    tail -30 /var/log/vllm.log
    exit 1
  fi
  echo -n "."
  sleep 5
  WAITED=$((WAITED + 5))
  if [ $WAITED -gt 600 ]; then
    echo "ERROR: vLLM did not start within 10 minutes"
    exit 1
  fi
done
echo ""
echo "  vLLM ready on port $VLLM_PORT ✓"

# Start Bright Masker
echo "[3/3] Starting Bright Masker on port $APP_PORT..."
cd /app
UVICORN_RELOAD=0 exec uvicorn app:app --host 0.0.0.0 --port "$APP_PORT"
