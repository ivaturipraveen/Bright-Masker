#!/bin/bash
set -e

# ── Self-reload after git pull ────────────────────────────────────────────────
# Bash buffers the script into memory at launch. Any code changes made by
# "git reset --hard" during Step 1 would NOT take effect in the current process.
# Solution: after the pull, re-exec this script with _BRIGHT_REEXEC=1 set so
# the fresh file is read from disk. The guard prevents an infinite loop.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_ROOT_TMP="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ "${_BRIGHT_REEXEC:-0}" != "1" ] && [ -f "$APP_ROOT_TMP/app.py" ]; then
  cd "$APP_ROOT_TMP"
  if git fetch origin main 2>/dev/null; then
    git reset --hard origin/main
    echo "[setup] Code updated — re-execing with latest start.sh..."
    export _BRIGHT_REEXEC=1
    exec bash "${BASH_SOURCE[0]}" "$@"
  fi
fi
unset _BRIGHT_REEXEC

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

# ── Step 1: Clone (first boot) or confirm pull already done ──────────────────
if [ ! -f "$APP_ROOT/app.py" ]; then
  echo "[setup] Repo missing — cloning into $APP_ROOT ..."
  mkdir -p "$(dirname "$APP_ROOT")"
  if ! git clone https://github.com/ivaturipraveen/Bright-Masker.git "$APP_ROOT"; then
    echo "ERROR: git clone failed — check network and repo URL."
    exit 1
  fi
  echo "[setup] Repo cloned."
else
  # Pull already happened at top of script (re-exec path) or was just done by
  # the bootstrap clone above. Log current HEAD for traceability.
  echo "[setup] Code up to date — $(cd "$APP_ROOT" && git log -1 --oneline)"
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
echo "[disk] workspace at boot:"; df -h /workspace | tail -1

# Clean up corrupted torch artifacts left by a previously interrupted install
if ls "$VENV/lib/"python*/site-packages/ 2>/dev/null | grep -q '^~'; then
  echo "[setup] Removing corrupted package artifacts (~ prefix) from prior failed install..."
  rm -rf "$VENV"/lib/python*/site-packages/~*
  echo "[setup] Cleanup done."
fi

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
  'DEFAULT_MODEL=deployed',
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
  'MODEL_DEPLOYED_DISPLAY=Qwen3-8B AWQ · Local vLLM',
  'MODEL_DEPLOYED_NAME=Qwen/Qwen3-8B-AWQ',
  'MODEL_DEPLOYED_BASE_URL=' + os.getenv('MODEL_DEPLOYED_BASE_URL', 'http://127.0.0.1:8002/v1'),
  'MODEL_DEPLOYED_API_KEY=no-key-needed',
  'MODEL_DEPLOYED_MAX_TOKENS=512',
  'MODEL_DEPLOYED_TIMEOUT=60.0',
  'MODEL_DEPLOYED_MAX_RETRIES=2',
  'MODEL_DEPLOYED_DISABLE_REASONING=true',
  'GLINER_MODEL_NAME=urchade/gliner_large-v2.1',
  'GLINER_DEVICE=cuda',
  'GLINER_THRESHOLD=' + os.getenv('GLINER_THRESHOLD', '0.25'),
  'GLINER_MAX_CHUNK_CHARS=1200',
  'GLINER_CHUNK_OVERLAP_CHARS=' + os.getenv('GLINER_CHUNK_OVERLAP_CHARS', '150'),
  'SPACY_MODEL_NAME=' + os.getenv('SPACY_MODEL_NAME', 'en_core_web_lg'),
  'ENTITIES_CONFIG_PATH=./entities_config.yaml',
  'LOG_LEVEL=' + os.getenv('LOG_LEVEL', 'INFO'),
  'TRANSFORMERS_OFFLINE=1',
  'ENABLE_ASYNC_LAYERS=true',
  'BATCH_MAX_CONCURRENCY=4',
  'PRESIDIO_MIN_SCORE=0.6',
  'PRESIDIO_NLP_ENGINE=spacy',
  'PRESIDIO_LANGUAGE=en',
  'LLM_MODEL_NAME=' + os.getenv('LLM_MODEL_NAME', 'qwen/qwen3-8b'),
  'LLM_TIMEOUT_SECONDS=60.0',
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

# ── Step 5: Install vLLM 0.8.x (requires torch>=2.6.0, CUDA 12.x) ────────────
# vLLM 0.7.x does NOT support Qwen3 (released April 2025); 0.8.5+ is required.
# We do NOT pin/downgrade torch — if a newer torch is already present it is fine.
VLLM_TARGET="0.8.5"
TORCH_MIN="2.6.0"
TORCH_CUDA_OK=$( "$PY" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False" )
VLLM_VER=$( "$PY" -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "none" )
TORCH_OK=$( "$PY" -c "
from packaging.version import Version
import torch
print('True' if Version(torch.__version__.split('+')[0]) >= Version('$TORCH_MIN') else 'False')
" 2>/dev/null || echo "False" )

if [ "$VLLM_VER" != "$VLLM_TARGET" ] || [ "$TORCH_CUDA_OK" != "True" ]; then
  echo "[setup] Installing vLLM $VLLM_TARGET (cached on /workspace)..."
  mkdir -p /workspace/.pip-cache
  if [ "$TORCH_OK" != "True" ]; then
    # Only install torch if below the minimum — never downgrade a newer version
    echo "[setup]   torch < $TORCH_MIN detected — installing torch $TORCH_MIN+cu124..."
    "$PY" -m pip install $PIP_BIG $PIP_CACHE \
      "torch>=$TORCH_MIN" "torchvision" \
      --index-url https://download.pytorch.org/whl/cu124
  else
    echo "[setup]   torch already >= $TORCH_MIN — skipping torch install."
  fi
  "$PY" -m pip install $PIP_BIG $PIP_CACHE "vllm==$VLLM_TARGET"
else
  echo "[setup] torch+vLLM $VLLM_VER already correct — skipping."
fi
"$PY" -c "import torch; print('[setup] torch CUDA:', torch.cuda.is_available(), torch.version.cuda)"
echo "[disk] after vLLM install:"; df -h /workspace | tail -1

# vLLM 0.8.x + GLiNER both need transformers in a compatible range
echo "[setup] Pinning transformers for vLLM 0.8 + GLiNER compatibility..."
"$PY" -m pip install $PIP_BIG $PIP_CACHE "transformers>=4.45.0,<5.0.0"
echo "[disk] after transformers pin:"; df -h /workspace | tail -1

# ── Step 6a: Remove stale full-precision model if AWQ version is present ─────
# Qwen3-8B bf16 = 16 GB; AWQ = 5.7 GB. If both exist, the bf16 is wasted space.
BF16_CACHE="$HF_HOME/hub/models--Qwen--Qwen3-8B"
AWQ_CACHE="$HF_HOME/hub/models--Qwen--Qwen3-8B-AWQ"
if [ -d "$BF16_CACHE" ] && [ -d "$AWQ_CACHE" ]; then
  echo "[setup] Removing unused Qwen3-8B bf16 model (16 GB) — AWQ is already cached..."
  rm -rf "$BF16_CACHE"
  echo "[disk] after bf16 cleanup:"; df -h /workspace | tail -1
fi

# ── Step 6b: Pre-download vLLM model (AWQ) ───────────────────────────────────
# Download before starting vLLM so first boot gives a clear progress message.
LLM_MODEL="Qwen/Qwen3-8B-AWQ"
LLM_CACHE="$HF_HOME/hub/models--Qwen--Qwen3-8B-AWQ"
if [ ! -d "$LLM_CACHE" ]; then
  echo "[setup] Downloading $LLM_MODEL (~5.7 GB, first boot only)..."
  "$PY" -c "
from huggingface_hub import snapshot_download
snapshot_download('$LLM_MODEL', ignore_patterns=['*.bin'])
print('LLM model ready.')
" || { echo "ERROR: Failed to download $LLM_MODEL"; exit 1; }
  echo "[disk] after LLM download:"; df -h /workspace | tail -1
else
  echo "[setup] LLM model already cached ($LLM_MODEL) — skipping."
fi

# ── Step 6c: Pre-download GLiNER model ───────────────────────────────────────
GLINER_MODEL="urchade/gliner_large-v2.1"
if ! "$PY" -c "from gliner import GLiNER; GLiNER.from_pretrained('$GLINER_MODEL')" &>/dev/null; then
  echo "[setup] Downloading GLiNER model $GLINER_MODEL (~1.7 GB, first boot only)..."
  "$PY" -c "from gliner import GLiNER; GLiNER.from_pretrained('$GLINER_MODEL'); print('GLiNER ready.')"
else
  echo "[setup] GLiNER model already cached — skipping."
fi
echo "[disk] after model cache:"; df -h /workspace | tail -1

# ── Step 6d: Disk safety check ───────────────────────────────────────────────
DISK_FREE_GB=$(df /workspace | awk 'NR==2 {printf "%.0f", $4/1024/1024}')
if [ "$DISK_FREE_GB" -lt 5 ]; then
  echo "WARNING: Only ${DISK_FREE_GB} GB free on /workspace. Consider upgrading volume size."
fi

# Purge pip cache — wheels are large and the volume has limited space.
echo "[setup] Purging pip cache to free disk space..."
rm -rf /workspace/.pip-cache
echo "[disk] after pip cache purge:"; df -h /workspace | tail -1

echo ""
echo "============================================"
echo "  Setup complete — starting services"
echo "============================================"

# Reduce allocator fragmentation when GLiNER + vLLM share one GPU
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# ── Step 7: Start vLLM ───────────────────────────────────────────────────────
# Default to AWQ quantized model — 4-bit weights are ~4.5 GB vs ~15.3 GB for bf16,
# leaving room for GLiNER on CUDA and a generous KV cache within 24 GB VRAM.
MODEL_NAME="Qwen/Qwen3-8B-AWQ"
# RunPod (and some images) bind nginx on 8001 — use 8002+ for vLLM
VLLM_PORT="${VLLM_PORT:-8002}"
APP_PORT="${PORT:-8000}"
# RTX 4090 (24 GB): AWQ weights ~4.5 GB + KV cache + GLiNER ~1.5 GB = ~20 GB (82%)
GPU_MEM="0.72"
MAX_CTX="8192"
VLLM_DTYPE="bfloat16"
# CUDA graphs ON (VLLM_ENFORCE_EAGER=0) — safe at 0.72 GPU util with AWQ.
# Override: set VLLM_ENFORCE_EAGER=1 in pod env only if vLLM crashes at startup.
if [ "${VLLM_ENFORCE_EAGER:-0}" = "1" ]; then
  EAGER_FLAG="--enforce-eager"
  echo "[setup] enforce-eager ON — CUDA graphs disabled."
else
  EAGER_FLAG=""
  echo "[setup] CUDA graphs ENABLED."
fi
# Optional extra CLI flags (space-separated)
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

# Legacy engine (V0) is more stable on RunPod single-GPU setups.
export VLLM_USE_V1="${VLLM_USE_V1:-0}"
# Reduce CUDA allocator fragmentation when GLiNER + vLLM share one GPU.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[setup] vLLM: gpu-memory-utilization=$GPU_MEM max-model-len=$MAX_CTX dtype=$VLLM_DTYPE enforce-eager=${VLLM_ENFORCE_EAGER:-0}"
echo "[setup] torch / CUDA check:"
"$PY" -c "import torch; print('  torch', torch.__version__, 'cuda=', torch.cuda.is_available(), getattr(torch.version, 'cuda', None))" || true

# Pre-check: load tokenizer to catch missing model files / bad model id early
echo "[setup] Verifying model tokenizer ($MODEL_NAME)..."
"$PY" -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$MODEL_NAME', trust_remote_code=True)
print('  tokenizer OK — vocab size:', tok.vocab_size)
" || { echo "ERROR: Cannot load tokenizer for '$MODEL_NAME'."; echo "  Check MODEL_DEPLOYED_NAME env var and that HF_HOME=/workspace/.cache/huggingface is accessible."; exit 1; }

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
    --trust-remote-code \
    --enable-prefix-caching \
    $EAGER_FLAG \
    $VLLM_EXTRA_ARGS \
    > /var/log/vllm.log 2>&1 &

  VLLM_PID=$!
  echo "  vLLM PID: $VLLM_PID  (live log → /var/log/vllm.log)"

  # Stream log to stdout so worker subprocess crashes appear in RunPod pod logs
  tail -f /var/log/vllm.log &
  TAIL_PID=$!

  WAITED=0
  until curl -sf "http://127.0.0.1:$VLLM_PORT/v1/models" | grep -q '"data"'; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
      kill $TAIL_PID 2>/dev/null
      echo ""
      echo "ERROR: vLLM process died — see log above and full file: /var/log/vllm.log"
      exit 1
    fi
    sleep 5
    WAITED=$((WAITED + 5))
    if [ $WAITED -gt 600 ]; then
      kill $TAIL_PID 2>/dev/null
      echo "ERROR: vLLM did not start within 10 minutes"
      exit 1
    fi
  done
  kill $TAIL_PID 2>/dev/null
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
