from __future__ import annotations

import asyncio
import json
import os
import time
import warnings
import httpx
from contextlib import asynccontextmanager
from pathlib import Path

# Suppress noisy deprecation warnings from third-party libraries
warnings.filterwarnings("ignore", message=".*resume_download.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*byte fallback.*", category=UserWarning)

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import AppConfig, Config
from exceptions import LlmApiError
from pipeline.orchestrator import PiiMaskingPipeline
from utils.logger import configure_logging, get_logger

log = get_logger(__name__)

_app_config: AppConfig | None = None
_pipeline: PiiMaskingPipeline | None = None

# ---------------------------------------------------------------------------
# Model registry — all values driven from .env, no hardcodings
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict] = {
    "qwen25_7b": {
        "display":     "Qwen 2.5 · 7B  (OpenRouter)",
        "model_name":  os.getenv("MODEL_Q25_NAME", "qwen/qwen-2.5-7b-instruct"),
        "base_url":    os.getenv("MODEL_Q25_BASE_URL", "https://openrouter.ai/api/v1"),
        "api_key":     os.getenv("MODEL_Q25_API_KEY", ""),
        "max_tokens":  int(os.getenv("MODEL_Q25_MAX_TOKENS", "1024")),
        "timeout":     float(os.getenv("MODEL_Q25_TIMEOUT", "25.0")),
        "max_retries": int(os.getenv("MODEL_Q25_MAX_RETRIES", "2")),
        "extra_body":  {},
        "speed":       "fast",
        "provider":    "openrouter",
    },
    "qwen3_8b": {
        "display":     "Qwen 3 · 8B  (OpenRouter)",
        "model_name":  os.getenv("MODEL_7B_NAME", ""),
        "base_url":    os.getenv("MODEL_7B_BASE_URL", ""),
        "api_key":     os.getenv("MODEL_7B_API_KEY", ""),
        "max_tokens":  int(os.getenv("MODEL_7B_MAX_TOKENS", "1024")),
        "timeout":     float(os.getenv("MODEL_7B_TIMEOUT", "25.0")),
        "max_retries": int(os.getenv("MODEL_7B_MAX_RETRIES", "2")),
        "extra_body":  {"reasoning": {"enabled": False}},
        "speed":       "fast",
        "provider":    "openrouter",
    },
    "qwen3_32b": {
        "display":     "Qwen 3 · 32B  (OpenRouter)",
        "model_name":  os.getenv("MODEL_72B_NAME", ""),
        "base_url":    os.getenv("MODEL_72B_BASE_URL", ""),
        "api_key":     os.getenv("MODEL_72B_API_KEY", ""),
        "max_tokens":  int(os.getenv("MODEL_72B_MAX_TOKENS", "1024")),
        "timeout":     float(os.getenv("MODEL_72B_TIMEOUT", "45.0")),
        "max_retries": int(os.getenv("MODEL_72B_MAX_RETRIES", "2")),
        "extra_body":  {"reasoning": {"enabled": False}},
        "speed":       "slow",
        "provider":    "openrouter",
    },
    "deployed": {
        "display":     "Qwen3-8B AWQ · Local vLLM",
        "model_name":  "Qwen/Qwen3-8B-AWQ",
        "base_url":    "http://127.0.0.1:8002/v1",
        "api_key":     "no-key-needed",
        "max_tokens":  1536,
        "timeout":     60.0,
        "max_retries": 2,
        "extra_body":  {"chat_template_kwargs": {"enable_thinking": False}},
        "speed":       "fast",
        "provider":    "deployed",
    },
}

_active_model_key: str = "deployed"


def _apply_model_config(key: str) -> None:
    """Push a MODEL_REGISTRY entry into the live config and rebuild the LLM client."""
    global _active_model_key
    cfg = MODEL_REGISTRY[key]
    _app_config._settings.llm_model_name      = cfg["model_name"]
    _app_config._settings.openrouter_base_url  = cfg["base_url"]
    _app_config._settings.openrouter_api_key   = cfg["api_key"]
    _app_config._settings.llm_max_tokens       = cfg["max_tokens"]
    _app_config._settings.llm_timeout_seconds  = cfg["timeout"]
    _app_config._settings.llm_max_retries      = cfg["max_retries"]
    _app_config._settings.llm_extra_body       = cfg["extra_body"]
    _app_config._settings.llm_stop_sequences   = cfg.get("stop_sequences", [])
    if _pipeline is not None and _pipeline.llm_layer is not None:
        _pipeline.llm_layer.rebuild_client()
    _active_model_key = key


async def _vllm_keepalive_loop() -> None:
    """Ping local vLLM every 90 s to prevent GPU P-state downscaling and prefix cache eviction."""
    vllm_url = "http://127.0.0.1:8002/v1/chat/completions"
    payload = {
        "model": "Qwen/Qwen3-8B-AWQ",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "temperature": 0.0,
        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
    }
    headers = {"Authorization": "Bearer no-key-needed"}
    while True:
        await asyncio.sleep(90)
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                await client.post(vllm_url, json=payload, headers=headers)
            log.debug("vllm_keepalive_ok")
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _app_config, _pipeline
    settings = Config()
    configure_logging(settings.log_level)
    _app_config = AppConfig(settings=settings)
    _pipeline = PiiMaskingPipeline(_app_config)
    await _pipeline._ensure_initialized()
    # Apply default model from env so LLM client is always env-configured
    default_key = os.getenv("DEFAULT_MODEL", "qwen3_8b")
    if default_key in MODEL_REGISTRY:
        _apply_model_config(default_key)
    log.info("server_ready", models={
        "ner_gliner": settings.gliner_model_name,
        "llm_review": _app_config.llm_model_name,
        "spacy": settings.spacy_model_name,
    })
    # Warm up pattern + NER engines so first real request doesn't pay cold-start.
    # Always runs regardless of which LLM is active — pattern/spaCy cold start is 4s+.
    # Also warms local vLLM CUDA graphs if deployed model is active.
    _warmup_text = "Warm-up: John Smith, john@example.com, (555) 010-0100."
    try:
        await _pipeline.process(_warmup_text)
        log.info("warmup_complete", model=_app_config.llm_model_name)
    except Exception as _e:
        log.warning("warmup_failed", error=str(_e)[:200])
    # If DEFAULT_MODEL is not deployed, also fire a silent warmup via the deployed
    # model so local vLLM is exercised before first user switch.
    # Non-fatal: if vLLM is down or crashes this must not block startup.
    if default_key != "deployed" and "deployed" in MODEL_REGISTRY:
        deployed_base = MODEL_REGISTRY["deployed"].get("base_url", "")
        if deployed_base:
            vllm_alive = await _check_vllm_health(deployed_base, timeout=5.0)
            if vllm_alive:
                try:
                    _apply_model_config("deployed")
                    await _pipeline.process(_warmup_text)
                    log.info("vllm_warmup_complete")
                except Exception as _e:
                    # vLLM may have crashed between health check and first request —
                    # this is a known issue when CUDA graph memory pushes over VRAM budget.
                    # Log clearly and continue; the /health/vllm endpoint can diagnose later.
                    log.warning("vllm_warmup_failed",
                                error=str(_e)[:200],
                                hint="vLLM may have crashed due to VRAM OOM after CUDA graph capture. "
                                     "Check /var/log/vllm.log or set VLLM_ENFORCE_EAGER=1.")
                finally:
                    _apply_model_config(default_key)
            else:
                log.warning("vllm_warmup_skipped", reason="vLLM not reachable at startup",
                            base_url=deployed_base)
    keepalive_task = asyncio.create_task(_vllm_keepalive_loop())
    yield
    keepalive_task.cancel()
    log.info("server_shutdown")


app = FastAPI(title="Bright Masker API", version="2.0.0", lifespan=lifespan)

_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class MaskRequest(BaseModel):
    text: str


class SpanInfo(BaseModel):
    entity_id: str
    display_name: str
    original: str
    masked: str
    confidence: float
    source: str
    strategy: str


class MaskResponse(BaseModel):
    masked_text: str
    original_text: str
    spans: list[SpanInfo]
    stats: dict
    response_time_ms: float


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_mask_response(result, app_config: AppConfig, response_time_ms: float) -> MaskResponse:
    masked_lookup: dict[tuple[str, str], str] = {
        (ms.entity_id, ms.original): ms.masked for ms in result.masked_spans
    }
    spans: list[SpanInfo] = []
    for span in result.detected_spans:
        entity_cfg = app_config.entities_by_id.get(span.entity_id)
        strategy = entity_cfg.masking.strategy.value if entity_cfg else "unknown"
        masked_val = masked_lookup.get((span.entity_id, span.text), span.text)
        spans.append(SpanInfo(
            entity_id=span.entity_id,
            display_name=span.display_name,
            original=span.text,
            masked=masked_val,
            confidence=round(span.confidence, 4),
            source=span.source,
            strategy=strategy,
        ))
    return MaskResponse(
        masked_text=result.masked_text,
        original_text=result.original_text,
        spans=spans,
        stats=result.stats.model_dump(),
        response_time_ms=round(response_time_ms, 2),
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def _branding_file(filename: str) -> FileResponse:
    path = _static_dir / filename
    if not path.is_file():
        raise HTTPException(404, f"Missing static asset: {filename}")
    return FileResponse(str(path), media_type="image/webp")


@app.get("/branding/logo.webp", include_in_schema=False)
async def branding_logo():
    """Serve logo; explicit route so the UI always resolves images."""
    return _branding_file("brightcone-logo.webp")


@app.get("/branding/wordmark.webp", include_in_schema=False)
async def branding_wordmark():
    return _branding_file("brightcone-wordmark.webp")


@app.get("/", include_in_schema=False)
async def index():
    html_path = _static_dir / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return JSONResponse({"message": "Bright Masker API — see /docs"})


@app.get("/health")
async def health():
    if _app_config is None:
        raise HTTPException(503, "Pipeline not initialized")
    return {
        "status": "ok",
        "active_model_key": _active_model_key,
        "models": {
            "ner_gliner": _app_config.gliner_model_name,
            "llm_review": _app_config.llm_model_name,
            "spacy": _app_config.spacy_model_name,
        },
        "openrouter_endpoint": _app_config.openrouter_base_url,
        "entities_loaded": len(_app_config.entities),
    }


@app.get("/entities")
async def list_entities():
    if _app_config is None:
        raise HTTPException(503, "Pipeline not initialized")
    return {
        "entities": [
            {
                "id": e.id,
                "display_name": e.display_name,
                "strategy": e.masking.strategy.value,
                "format": e.masking.format,
                "layers": _entity_layers(e),
                "priority": e.priority,
                "confidence_threshold": e.confidence_threshold,
            }
            for e in _app_config.entities
        ]
    }


def _entity_layers(entity) -> list[str]:
    layers = []
    if entity.presidio_type:
        layers.append("pattern")
    layers.append("ner")
    layers.append("llm")
    return layers


class ModelSelectRequest(BaseModel):
    model_key: str


@app.get("/config/models")
async def list_models():
    """Return all available LLM models and which one is currently active."""
    return {
        "active": _active_model_key,
        "models": [
            {
                "key":      k,
                "display":  v["display"],
                "provider": v["provider"],
                "speed":    v["speed"],
                "active":   k == _active_model_key,
            }
            for k, v in MODEL_REGISTRY.items()
        ],
    }


async def _check_vllm_health(base_url: str, timeout: float = 5.0) -> bool:
    """Ping vLLM /v1/models — returns True if reachable and serving."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.get(f"{base_url.rstrip('/')}/models")
            return r.status_code == 200 and "data" in r.text
    except Exception:
        return False


@app.get("/health/vllm")
async def vllm_health():
    """Check whether the local vLLM process is reachable."""
    cfg = MODEL_REGISTRY.get("deployed", {})
    base_url = cfg.get("base_url", "http://127.0.0.1:8002/v1")
    healthy = await _check_vllm_health(base_url)
    if not healthy:
        raise HTTPException(503, f"vLLM not reachable at {base_url}")
    return {"ok": True, "base_url": base_url}


@app.post("/config/model")
async def set_model(body: ModelSelectRequest):
    """Hot-swap the LLM backend at runtime — no restart required."""
    if _pipeline is None or _app_config is None:
        raise HTTPException(503, "Pipeline not initialized")

    key = body.model_key
    if key not in MODEL_REGISTRY:
        raise HTTPException(400, f"Unknown model key '{key}'. Valid: {list(MODEL_REGISTRY)}")

    # For the deployed (local vLLM) model, verify the endpoint is reachable before switching
    cfg = MODEL_REGISTRY[key]
    if cfg.get("provider") == "deployed":
        base_url = cfg.get("base_url", "")
        if base_url and not await _check_vllm_health(base_url):
            raise HTTPException(503,
                f"Cannot switch to '{key}': vLLM not reachable at {base_url}. "
                "Check pod logs — vLLM may have crashed.")

    _apply_model_config(key)
    log.info("model_switched", key=key, model=cfg["model_name"], base_url=cfg["base_url"])

    return {
        "ok":     True,
        "active": key,
        "display": cfg["display"],
        "model":  cfg["model_name"],
    }


@app.post("/mask", response_model=MaskResponse)
async def mask_text(request: MaskRequest):
    """
    Mask PII in text of any length. Large texts are chunked internally
    across all detection layers — no entity is missed at chunk boundaries
    because each layer uses overlapping windows.
    """
    if _pipeline is None or _app_config is None:
        raise HTTPException(503, "Pipeline not initialized")
    if not request.text.strip():
        raise HTTPException(400, "text must not be empty")

    t0 = time.perf_counter()
    try:
        result = await _pipeline.process(request.text)
    except LlmApiError as exc:
        raise HTTPException(503, detail=str(exc))
    response_time_ms = (time.perf_counter() - t0) * 1000
    return _build_mask_response(result, _app_config, response_time_ms)


# ---------------------------------------------------------------------------
# Streaming endpoint — SSE progress events then final result
# ---------------------------------------------------------------------------

@app.post("/mask/stream")
async def mask_text_stream(request: MaskRequest):
    """
    Server-Sent Events endpoint. Same as /mask but streams step-by-step
    progress so callers can show a live progress indicator.

    For large texts the LLM layer emits one event per chunk:
        {"type":"progress","step":3,"name":"llm_chunk","chunk":N,"total_chunks":M,...}

    Final event:
        {"type":"complete","result":{...full MaskResponse...}}

    Error event:
        {"type":"error","message":"..."}
    """
    if _pipeline is None or _app_config is None:
        raise HTTPException(503, "Pipeline not initialized")
    if not request.text.strip():
        raise HTTPException(400, "text must not be empty")

    queue: asyncio.Queue = asyncio.Queue()
    result_holder: list = []
    error_holder: list = []

    async def _run() -> None:
        try:
            res = await _pipeline.process(request.text, progress_queue=queue)
            result_holder.append(res)
        except Exception as exc:
            error_holder.append(exc)
        finally:
            await queue.put(None)  # sentinel — always sent

    async def generate():
        t0 = time.perf_counter()
        asyncio.create_task(_run())

        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

        if error_holder:
            yield f"data: {json.dumps({'type': 'error', 'message': str(error_holder[0])})}\n\n"
            return

        if result_holder:
            response_time_ms = (time.perf_counter() - t0) * 1000
            resp = _build_mask_response(result_holder[0], _app_config, response_time_ms)
            yield f"data: {json.dumps({'type': 'complete', 'result': resp.model_dump()})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
