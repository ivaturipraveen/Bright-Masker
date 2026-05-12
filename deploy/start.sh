#!/bin/bash
set -e

echo "============================================"
echo "  Bright Masker — RunPod Startup"
echo "============================================"

# ── Step 1: Clone repo if /app is missing ────────────────────────────────────
if [ ! -f /app/app.py ]; then
  echo "[setup] /app not found — cloning repo..."
  git clone https://github.com/ivaturipraveen/Bright-Masker.git /app
  echo "[setup] Repo cloned."
else
  echo "[setup] Repo already present — skipping clone."
fi

cd /app

# ── Step 2: Write .env from RunPod environment variables ─────────────────────
echo "[setup] Writing .env from environment..."
python3 -c "
import os
lines = [
  'DEFAULT_MODEL=' + os.getenv('DEFAULT_MODEL', 'deployed'),
  'MODEL_Q25_NAME=' + os.getenv('MODEL_Q25_NAME', 'qwen/qwen-2.5-7b-instruct'),
  'MODEL_Q25_BASE_URL=' + os.getenv('MODEL_Q25_BASE_URL', 'https://openrouter.ai/api/v1'),
  'MODEL_Q25_API_KEY=' + os.getenv('MODEL_Q25_API_KEY', ''),
  'MODEL_Q25_MAX_TOKENS=' + os.getenv('MODEL_Q25_MAX_TOKENS', '1024'),
  'MODEL_Q25_TIMEOUT=' + os.getenv('MODEL_Q25_TIMEOUT', '25.0'),
  'MODEL_Q25_MAX_RETRIES=' + os.getenv('MODEL_Q25_MAX_RETRIES', '2'),
  'MODEL_7B_NAME=' + os.getenv('MODEL_7B_NAME', 'qwen/qwen3-8b'),
  'MODEL_7B_BASE_URL=' + os.getenv('MODEL_7B_BASE_URL', 'https://openrouter.ai/api/v1'),
  'MODEL_7B_API_KEY=' + os.getenv('MODEL_7B_API_KEY', ''),
  'MODEL_7B_MAX_TOKENS=' + os.getenv('MODEL_7B_MAX_TOKENS', '1024'),
  'MODEL_7B_TIMEOUT=' + os.getenv('MODEL_7B_TIMEOUT', '25.0'),
  'MODEL_7B_MAX_RETRIES=' + os.getenv('MODEL_7B_MAX_RETRIES', '2'),
  'MODEL_72B_NAME=' + os.getenv('MODEL_72B_NAME', 'qwen/qwen3-32b'),
  'MODEL_72B_BASE_URL=' + os.getenv('MODEL_72B_BASE_URL', 'https://openrouter.ai/api/v1'),
  'MODEL_72B_API_KEY=' + os.getenv('MODEL_72B_API_KEY', ''),
  'MODEL_72B_MAX_TOKENS=' + os.getenv('MODEL_72B_MAX_TOKENS', '1024'),
  'MODEL_72B_TIMEOUT=' + os.getenv('MODEL_72B_TIMEOUT', '45.0'),
  'MODEL_72B_MAX_RETRIES=' + os.getenv('MODEL_72B_MAX_RETRIES', '2'),
  'MODEL_DEPLOYED_DISPLAY=Qwen 3 8B (Deployed vLLM)',
  'MODEL_DEPLOYED_NAME=' + os.getenv('MODEL_DEPLOYED_NAME', 'Qwen/Qwen3-8B-Instruct'),
  'MODEL_DEPLOYED_BASE_URL=' + os.getenv('MODEL_DEPLOYED_BASE_URL', 'http://127.0.0.1:8002/v1'),
  'MODEL_DEPLOYED_API_KEY=' + os.getenv('MODEL_DEPLOYED_API_KEY', 'no-key-needed'),
  'MODEL_DEPLOYED_MAX_TOKENS=' + os.getenv('MODEL_DEPLOYED_MAX_TOKENS', '1024'),
  'MODEL_DEPLOYED_TIMEOUT=' + os.getenv('MODEL_DEPLOYED_TIMEOUT', '30.0'),
  'MODEL_DEPLOYED_MAX_RETRIES=' + os.getenv('MODEL_DEPLOYED_MAX_RETRIES', '2'),
  'MODEL_DEPLOYED_DISABLE_REASONING=true',
  'GLINER_MODEL_NAME=' + os.getenv('GLINER_MODEL_NAME', 'urchade/gliner_large-v2.1'),
  'GLINER_THRESHOLD=' + os.getenv('GLINER_THRESHOLD', '0.25'),
  'GLINER_MAX_CHUNK_CHARS=' + os.getenv('GLINER_MAX_CHUNK_CHARS', '2000'),
  'GLINER_CHUNK_OVERLAP_CHARS=' + os.getenv('GLINER_CHUNK_OVERLAP_CHARS', '150'),
  'SPACY_MODEL_NAME=' + os.getenv('SPACY_MODEL_NAME', 'en_core_web_lg'),
  'ENTITIES_CONFIG_PATH=./entities_config.yaml',
  'LOG_LEVEL=' + os.getenv('LOG_LEVEL', 'INFO'),
  'TRANSFORMERS_OFFLINE=0',
  'ENABLE_ASYNC_LAYERS=true',
  'BATCH_MAX_CONCURRENCY=4',
  'PRESIDIO_MIN_SCORE=0.6',
  'PRESIDIO_NLP_ENGINE=spacy',
  'PRESIDIO_LANGUAGE=en',
  'LLM_MODEL_NAME=' + os.getenv('LLM_MODEL_NAME', 'qwen/qwen3-8b'),
  'LLM_TIMEOUT_SECONDS=25.0',
  'LLM_MAX_RETRIES=2',
  'LLM_MAX_TOKENS=1024',
  'LLM_TEMPERATURE=0.0',
  'LLM_CONTEXT_CHARS=80',
  'MAX_LLM_BATCH_SIZE=10',
  'FAKER_SEED=42',
  'ENCRYPTION_KEY=' + os.getenv('ENCRYPTION_KEY', 'change-this-to-a-random-secret-key-in-production'),
  'OPENROUTER_API_KEY=' + os.getenv('OPENROUTER_API_KEY', ''),
  'OPENROUTER_BASE_URL=https://openrouter.ai/api/v1',
]
with open('/app/.env', 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('.env written.')
"

# ── Step 3: Install Python dependencies ──────────────────────────────────────
if ! python -c "import fastapi" &>/dev/null; then
  echo "[setup] Installing Python dependencies..."
  pip install -r requirements.txt --quiet
else
  echo "[setup] Python deps already installed — skipping."
fi

# ── Step 4: Install spaCy model ──────────────────────────────────────────────
if ! python -c "import spacy; spacy.load('en_core_web_lg')" &>/dev/null; then
  echo "[setup] Downloading spaCy en_core_web_lg..."
  python -m spacy download en_core_web_lg
else
  echo "[setup] spaCy en_core_web_lg already present — skipping."
fi

# ── Step 5: Install vLLM if missing ──────────────────────────────────────────
if ! python -c "import vllm" &>/dev/null; then
  echo "[setup] Installing vLLM..."
  pip install vllm --quiet
else
  echo "[setup] vLLM already installed — skipping."
fi
# vLLM pulls a newer transformers; GLiNER needs <5.2 — pin after vLLM install
echo "[setup] Pinning transformers for GLiNER compatibility..."
pip install "transformers>=4.51.3,<5.2.0" --quiet

# ── Step 6: Pre-download GLiNER model ────────────────────────────────────────
GLINER_MODEL="${GLINER_MODEL_NAME:-urchade/gliner_large-v2.1}"
if ! python -c "from gliner import GLiNER; GLiNER.from_pretrained('$GLINER_MODEL')" &>/dev/null; then
  echo "[setup] Downloading GLiNER model $GLINER_MODEL ..."
  python -c "from gliner import GLiNER; GLiNER.from_pretrained('$GLINER_MODEL'); print('GLiNER ready.')"
else
  echo "[setup] GLiNER model already cached — skipping."
fi

echo ""
echo "============================================"
echo "  Setup complete — starting services"
echo "============================================"

# ── Step 7: Start vLLM ───────────────────────────────────────────────────────
MODEL_NAME="${MODEL_DEPLOYED_NAME:-Qwen/Qwen3-8B-Instruct}"
# RunPod (and some images) bind nginx on 8001 — use 8002+ for vLLM
VLLM_PORT="${VLLM_PORT:-8002}"
APP_PORT="${PORT:-8000}"
GPU_MEM="${VLLM_GPU_UTIL:-0.75}"
MAX_CTX="${VLLM_MAX_MODEL_LEN:-8192}"

echo "[1/2] Starting vLLM ($MODEL_NAME) on port $VLLM_PORT..."
python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_NAME" \
  --host 0.0.0.0 \
  --port "$VLLM_PORT" \
  --max-model-len "$MAX_CTX" \
  --gpu-memory-utilization "$GPU_MEM" \
  --no-enable-log-requests \
  --dtype bfloat16 \
  > /var/log/vllm.log 2>&1 &

VLLM_PID=$!
echo "  vLLM PID: $VLLM_PID"

WAITED=0
until curl -sf "http://127.0.0.1:$VLLM_PORT/v1/models" | grep -q '"data"'; do
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "ERROR: vLLM process died. Last logs:"
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

# ── Step 8: Start Bright Masker ──────────────────────────────────────────────
echo "[2/2] Starting Bright Masker on port $APP_PORT..."
cd /app
TRANSFORMERS_OFFLINE=1 exec uvicorn app:app --host 0.0.0.0 --port "$APP_PORT"
