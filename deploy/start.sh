#!/bin/bash
set -e

# RunPod: container root FS is wiped every restart — only /workspace persists.
# We install all deps into a venv on the volume so "pip install" runs once.
PYTHON_BOOT="${PYTHON_BOOT:-${PYTHON:-python3}}"
# vLLM wheels are huge; default pip timeouts often fail on RunPod.
if [ -z "$PIP_BIG" ]; then
  PIP_BIG="--default-timeout=900 --retries 15"
fi

echo "============================================"
echo "  Bright Masker — RunPod Startup"
echo "============================================"

# Repo root = parent of deploy/ (works for /app or /workspace/Bright-Masker)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export APP_ROOT
echo "  APP_ROOT: $APP_ROOT"

# ── Step 1: Clone or pull latest code ────────────────────────────────────────
if [ ! -f "$APP_ROOT/app.py" ]; then
  echo "[setup] Repo missing — cloning into $APP_ROOT ..."
  mkdir -p "$(dirname "$APP_ROOT")"
  git clone https://github.com/ivaturipraveen/Bright-Masker.git "$APP_ROOT"
  echo "[setup] Repo cloned."
else
  echo "[setup] Pulling latest code from GitHub..."
  cd "$APP_ROOT"
  git fetch origin main 2>/dev/null && git reset --hard origin/main 2>/dev/null || echo "[setup] git pull skipped (no network or detached)"
  echo "[setup] Code up to date."
fi

cd "$APP_ROOT"

# Persistent Python env (survives pod restarts; lives on the RunPod volume).
VENV="${BRIGHT_MASKER_VENV:-/workspace/.bright-masker-venv}"
if [ ! -x "$VENV/bin/python" ]; then
  echo "[setup] Creating venv at $VENV (first boot only; lives on /workspace volume)..."
  "$PYTHON_BOOT" -m venv "$VENV"
fi
PY="$VENV/bin/python"
echo "  PY=$PY  — all pip installs go here, not system Python (override path: BRIGHT_MASKER_VENV)"

# Hugging Face cache on volume so models are not re-downloaded every boot
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
mkdir -p "$HF_HOME"
PIP_CACHE="--cache-dir /workspace/.pip-cache"
mkdir -p /workspace/.pip-cache

# RunPod / base images sometimes set HF offline — GLiNER + vLLM need Hub on first run.
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0

# ── Step 2: Write .env from RunPod environment variables ─────────────────────
echo "[setup] Writing .env from environment..."
python3 -c "
import os
app_root = os.environ['APP_ROOT']
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
  'MODEL_DEPLOYED_DISPLAY=Qwen3-8B-Deployed-vLLM',
  'MODEL_DEPLOYED_NAME=' + os.getenv('MODEL_DEPLOYED_NAME', 'Qwen/Qwen3-8B'),
  'MODEL_DEPLOYED_BASE_URL=' + os.getenv('MODEL_DEPLOYED_BASE_URL', 'http://127.0.0.1:8002/v1'),
  'MODEL_DEPLOYED_API_KEY=' + os.getenv('MODEL_DEPLOYED_API_KEY', 'no-key-needed'),
  'MODEL_DEPLOYED_MAX_TOKENS=' + os.getenv('MODEL_DEPLOYED_MAX_TOKENS', '512'),
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
  'LLM_MAX_TOKENS=' + os.getenv('LLM_MAX_TOKENS', '512'),
  'LLM_TEMPERATURE=0.0',
  'LLM_CONTEXT_CHARS=80',
  'MAX_LLM_BATCH_SIZE=10',
  'FAKER_SEED=42',
  'ENCRYPTION_KEY=' + os.getenv('ENCRYPTION_KEY', 'change-this-to-a-random-secret-key-in-production'),
  'OPENROUTER_API_KEY=' + os.getenv('OPENROUTER_API_KEY', ''),
  'OPENROUTER_BASE_URL=https://openrouter.ai/api/v1',
]
with open(os.path.join(app_root, '.env'), 'w') as f:
    f.write('\n'.join(lines) + '\n')
print('.env written.')
"

# ── Step 3: Install Python dependencies ──────────────────────────────────────
if ! "$PY" -c "import fastapi" &>/dev/null; then
  echo "[setup] Installing Python dependencies into venv ($PY -m pip)..."
  "$PY" -m pip install $PIP_BIG $PIP_CACHE --upgrade pip setuptools wheel
  "$PY" -m pip install $PIP_BIG $PIP_CACHE -r requirements.txt
  echo "[setup] Python dependencies installed."
else
  echo "[setup] Python deps already installed — skipping."
fi

# ── Step 4: Install spaCy model ──────────────────────────────────────────────
if ! "$PY" -c "import spacy; spacy.load('en_core_web_lg')" &>/dev/null; then
  echo "[setup] Downloading spaCy en_core_web_lg..."
  "$PY" -m spacy download en_core_web_lg
else
  echo "[setup] spaCy en_core_web_lg already present — skipping."
fi

# ── Step 5: Install torch cu124 + vLLM 0.7.x (compatible pair for CUDA 12.x) ─
VLLM_TARGET="0.7.3"
TORCH_CUDA_OK=$( "$PY" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False" )
VLLM_VER=$( "$PY" -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "none" )

if [ "$VLLM_VER" != "$VLLM_TARGET" ] || [ "$TORCH_CUDA_OK" != "True" ]; then
  echo "[setup] Installing torch 2.5.1+cu124 and vLLM $VLLM_TARGET (cached on /workspace)..."
  mkdir -p /workspace/.pip-cache
  # Install torch for cu124 first so vLLM does not pull cu130
  "$PY" -m pip install $PIP_BIG $PIP_CACHE \
    "torch==2.5.1" "torchvision==0.20.1" \
    --index-url https://download.pytorch.org/whl/cu124
  # Install vLLM pinned — must not upgrade torch
  "$PY" -m pip install $PIP_BIG $PIP_CACHE "vllm==$VLLM_TARGET"
  # vLLM may re-pull torch; force cu124 again
  "$PY" -m pip install $PIP_BIG $PIP_CACHE \
    "torch==2.5.1" "torchvision==0.20.1" \
    --index-url https://download.pytorch.org/whl/cu124
else
  echo "[setup] torch+vLLM $VLLM_VER already correct — skipping."
fi
"$PY" -c "import torch; print('[setup] torch CUDA:', torch.cuda.is_available(), torch.version.cuda)"

# vLLM 0.7.x + GLiNER both need transformers in a compatible range
echo "[setup] Pinning transformers for vLLM 0.7 + GLiNER compatibility..."
"$PY" -m pip install $PIP_BIG $PIP_CACHE "transformers>=4.45.0,<5.0.0"

# ── Step 6: Pre-download GLiNER model ────────────────────────────────────────
GLINER_MODEL="${GLINER_MODEL_NAME:-urchade/gliner_large-v2.1}"
if ! "$PY" -c "from gliner import GLiNER; GLiNER.from_pretrained('$GLINER_MODEL')" &>/dev/null; then
  echo "[setup] Downloading GLiNER model $GLINER_MODEL ..."
  "$PY" -c "from gliner import GLiNER; GLiNER.from_pretrained('$GLINER_MODEL'); print('GLiNER ready.')"
else
  echo "[setup] GLiNER model already cached — skipping."
fi

echo ""
echo "============================================"
echo "  Setup complete — starting services"
echo "============================================"

# Reduce allocator fragmentation when GLiNER + vLLM share one GPU
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Step 7: Start vLLM ───────────────────────────────────────────────────────
MODEL_NAME="${MODEL_DEPLOYED_NAME:-Qwen/Qwen3-8B}"
# RunPod (and some images) bind nginx on 8001 — use 8002+ for vLLM
VLLM_PORT="${VLLM_PORT:-8002}"
APP_PORT="${PORT:-8000}"
# Qwen3-8B bf16 weights alone are ~16 GB; 0.70×24 GB ≈ 16.8 GB leaves almost no KV
# headroom and often fails with "Engine process failed to start". Default higher;
# if /mask GLiNER OOMs, lower VLLM_GPU_UTIL (e.g. 0.82) or shorten context.
GPU_MEM="${VLLM_GPU_UTIL:-0.90}"
MAX_CTX="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
# Optional extra CLI flags (space-separated), e.g. "--enforce-eager"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

# v1 multiprocess engine often fails on RunPod; legacy engine is more stable.
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

echo "[setup] vLLM: gpu-memory-utilization=$GPU_MEM max-model-len=$MAX_CTX dtype=$VLLM_DTYPE"
echo "[setup] torch / CUDA check:"
"$PY" -c "import torch; print('  torch', torch.__version__, 'cuda=', torch.cuda.is_available(), getattr(torch.version, 'cuda', None))" || true

# ── Start vLLM (skip if already running) ─────────────────────────────────────
if curl -sf "http://127.0.0.1:$VLLM_PORT/v1/models" | grep -q '"data"' 2>/dev/null; then
  echo "[1/2] vLLM already running on port $VLLM_PORT — skipping."
else
  echo "[1/2] Starting vLLM ($MODEL_NAME) on port $VLLM_PORT..."
  # shellcheck disable=SC2086
  "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --max-model-len "$MAX_CTX" \
    --gpu-memory-utilization "$GPU_MEM" \
    --disable-log-requests \
    --dtype "$VLLM_DTYPE" \
    $VLLM_EXTRA_ARGS \
    > /var/log/vllm.log 2>&1 &

  VLLM_PID=$!
  echo "  vLLM PID: $VLLM_PID"
  WAITED=0
  until curl -sf "http://127.0.0.1:$VLLM_PORT/v1/models" | grep -q '"data"'; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
      echo "ERROR: vLLM process died. Full log: /var/log/vllm.log (excerpt below)"
      tail -250 /var/log/vllm.log
      echo ""
      echo "Hints: raise VRAM for weights+KV → set VLLM_GPU_UTIL=0.90 (default) or 0.92;"
      echo "  OOM during masking → lower VLLM_GPU_UTIL or VLLM_MAX_MODEL_LEN (e.g. 3072);"
      echo "  older GPU → VLLM_DTYPE=float16; unstable worker → VLLM_EXTRA_ARGS='--enforce-eager'"
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
fi

# ── Start Bright Masker (skip if already running) ─────────────────────────────
if curl -sf "http://127.0.0.1:$APP_PORT/health" 2>/dev/null | grep -q '.'; then
  echo "[2/2] Bright Masker already running on port $APP_PORT — skipping."
  # Keep container alive
  echo "  All services running. Sleeping to keep container alive..."
  tail -f /var/log/vllm.log
else
  echo "[2/2] Starting Bright Masker on port $APP_PORT..."
  cd "$APP_ROOT"
  TRANSFORMERS_OFFLINE=1 exec "$PY" -m uvicorn app:app --host 0.0.0.0 --port "$APP_PORT"
fi
