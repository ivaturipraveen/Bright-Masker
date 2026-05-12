# Running Bright Masker on RunPod

## RunPod start command (recommended)

Use this in **Edit pod → Start command**. It clones once, then always runs `deploy/start.sh` (which **git pull**s latest `main` on every boot).

```bash
bash -c 'git clone https://github.com/ivaturipraveen/Bright-Masker.git /workspace/Bright-Masker 2>/dev/null || true; bash /workspace/Bright-Masker/deploy/start.sh'
```

You do **not** need to change this line when we push repo updates: `start.sh` pulls `origin/main` before starting services.

---

## Optional RunPod environment variables

Set these in **Edit pod → Environment variables** if defaults are wrong for your GPU.

| Variable | Purpose | Default in `start.sh` |
|----------|---------|------------------------|
| `DEFAULT_MODEL` | Active LLM profile | `deployed` |
| `MODEL_DEPLOYED_BASE_URL` | vLLM OpenAI base URL | `http://127.0.0.1:8002/v1` |
| `MODEL_DEPLOYED_NAME` | HF model id for vLLM | `Qwen/Qwen3-8B` |
| `VLLM_GPU_UTIL` | vLLM GPU memory fraction | **`0.80`** (leave VRAM for GLiNER; raise only if you OOM on vLLM) |
| `VLLM_MAX_MODEL_LEN` | vLLM context window | `4096` |
| `PORT` | Uvicorn port | `8000` |
| `LLM_MAX_TOKENS` | Max completion tokens (must fit under `VLLM_MAX_MODEL_LEN` with prompt) | **`512`** |
| `HF_HUB_OFFLINE` | Set `0` for first-time downloads | `start.sh` forces `0` during setup |

**Do not** put spaces in values you later `export` from `.env` in a shell (e.g. avoid spaces in display names).

---

## Three levels of “keep it running”

### 1. Same pod, browser closed

Processes keep running until you **Stop** the pod. Check:

```bash
curl -s http://127.0.0.1:8002/v1/models
curl -s http://127.0.0.1:8000/docs | head -5
```

### 2. Survive SSH / web terminal disconnect

Use **screen** (example uses **0.80** GPU util for GLiNER headroom):

```bash
screen -S vllm -dm bash -c '
python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B \
  --host 0.0.0.0 --port 8002 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.80 \
  --disable-log-requests \
  --dtype bfloat16 \
  > /var/log/vllm.log 2>&1'

screen -S app -dm bash -c '
until curl -sf http://127.0.0.1:8002/v1/models | grep -q "data"; do sleep 5; done
cd /workspace/Bright-Masker
TRANSFORMERS_OFFLINE=1 python3 -m uvicorn app:app --host 0.0.0.0 --port 8000'
```

Attach: `screen -r vllm` / `screen -r app` — detach with **Ctrl+A**, then **D**.

### 3. Auto-start every pod boot

Use the **start command** above. First boot can take 15–25+ minutes (pip + models). Later boots are faster thanks to `/workspace` and pip cache under `/workspace/.pip-cache`.

---

## Where is the public URL?

**Connect → HTTP services → Port 8000** → link like:

`https://<pod-id>-8000.proxy.runpod.net`

- **UI:** `https://<pod-id>-8000.proxy.runpod.net/`
- **Swagger:** `https://<pod-id>-8000.proxy.runpod.net/docs`

---

## Quick API test

```bash
curl -s -X POST "https://<pod-id>-8000.proxy.runpod.net/mask" \
  -H "Content-Type: application/json" \
  -d '{"text": "My name is John Smith and my email is john@example.com"}' | python3 -m json.tool
```

---

## If you removed the start command

Run once in the pod:

```bash
cd /workspace/Bright-Masker && git pull origin main && bash deploy/start.sh
```

Or re-add the same start command in RunPod **Edit pod** when you want automatic boots again.
