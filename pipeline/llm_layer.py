from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from config import AppConfig
from exceptions import LlmApiError
from models.schemas import DetectedSpan
from utils.logger import get_logger, log_llm_chunk

log = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_DETECT_CHUNK_SIZE = 3500
_DETECT_CHUNK_OVERLAP = 200
_LLM_CONFIDENCE = 0.93
_MAX_PARALLEL_CHUNKS = 3


class LlmLayer:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._client = self._build_client()
        self._va_system = _load_template("llm_validate_augment_system.txt")
        self._va_user = _load_template("llm_validate_augment_user.txt")

    def _build_client(self) -> Any:
        try:
            import openai
            return openai.AsyncOpenAI(
                base_url=self._config.openrouter_base_url,
                api_key=self._config.openrouter_api_key,
            )
        except ImportError as exc:
            raise LlmApiError("openai package not installed") from exc

    def rebuild_client(self) -> None:
        """Hot-swap the OpenAI client after a runtime model switch."""
        self._client = self._build_client()

    async def _call_api(self, system_prompt: str, user_prompt: str) -> tuple[str, bool]:
        """Returns (response_text, success). success=False means API call failed."""
        for attempt in range(self._config.llm_max_retries):
            try:
                extra_body = getattr(self._config, "llm_extra_body", None) or {}
                stop_seqs = getattr(self._config, "llm_stop_sequences", None) or []
                response = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=self._config.llm_model_name,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        max_tokens=self._config.llm_max_tokens,
                        temperature=self._config.llm_temperature,
                        **({"stop": stop_seqs} if stop_seqs else {}),
                        **({"extra_body": extra_body} if extra_body else {}),
                    ),
                    timeout=self._config.llm_timeout_seconds,
                )
                return response.choices[0].message.content or "", True
            except asyncio.TimeoutError:
                log.warning("llm_timeout", attempt=attempt + 1,
                            max_retries=self._config.llm_max_retries)
                continue
            except Exception as exc:
                err_str = str(exc)
                if "429" in err_str or "rate_limit" in err_str.lower():
                    wait = 2 ** attempt
                    log.warning("llm_rate_limit", attempt=attempt + 1, wait_seconds=wait)
                    await asyncio.sleep(wait)
                    continue
                if "402" in err_str:
                    log.error("llm_error_insufficient_credits",
                              error=err_str[:200],
                              fix="top_up_OpenRouter_credits_OR_reduce_LLM_MAX_TOKENS_in_.env")
                elif "401" in err_str or "authentication" in err_str.lower():
                    log.error("llm_error_auth_failed",
                              error=err_str[:200],
                              fix="check_OPENROUTER_API_KEY_in_.env")
                elif "model_not_found" in err_str.lower() or "404" in err_str:
                    log.error("llm_error_model_not_found",
                              error=err_str[:200],
                              fix="check_LLM_MODEL_NAME_in_.env")
                else:
                    log.error("llm_error_api",
                              error=err_str[:200])
                return "", False
        log.warning("llm_all_retries_exhausted", retries=self._config.llm_max_retries)
        return "", False

    async def validate_and_augment(
        self,
        text: str,
        candidate_spans: list[DetectedSpan],
        entities_by_id: dict,
        progress_queue: asyncio.Queue | None = None,
    ) -> tuple[list[DetectedSpan], bool]:
        """Validate NER/pattern candidates and augment with any missed PII.

        Returns (spans, llm_succeeded). llm_succeeded=False means all API calls failed.
        The orchestrator treats this as a hard error and raises LlmApiError.
        """
        if not text.strip():
            return [], True

        entity_lines = [
            f"  {eid} ({cfg.display_name})"
            for eid, cfg in entities_by_id.items()
        ]
        system_prompt = self._va_system.replace("{entity_list}", "\n".join(entity_lines))

        chunks_with_pos = _chunk_text_with_positions(text, _DETECT_CHUNK_SIZE, _DETECT_CHUNK_OVERLAP)
        total_chunks = len(chunks_with_pos)
        log.debug("llm_va_start",
                 chunks=total_chunks,
                 chars=len(text),
                 candidates=len(candidate_spans))

        semaphore = asyncio.Semaphore(_MAX_PARALLEL_CHUNKS)
        chunk_timings: list[float] = [0.0] * total_chunks

        async def _process_chunk(chunk: str, chunk_start: int, idx: int) -> tuple[list[dict], bool]:
            async with semaphore:
                t_chunk = asyncio.get_event_loop().time()
                chunk_end = chunk_start + len(chunk)
                chunk_candidates = [
                    {"entity_id": s.entity_id, "text": s.text, "source": s.source}
                    for s in candidate_spans
                    if s.start < chunk_end and s.end > chunk_start
                ]
                candidates_json = json.dumps(chunk_candidates, ensure_ascii=False)
                user_prompt = (
                    self._va_user
                    .replace("{candidates_json}", candidates_json)
                    .replace("{text}", chunk)
                )
                raw, api_ok = await self._call_api(system_prompt, user_prompt)
                chunk_ms = (asyncio.get_event_loop().time() - t_chunk) * 1000
                chunk_timings[idx] = chunk_ms
                detections, parse_ok = _parse_json(raw)
                # Chunk fails if API call failed OR JSON was present but unparseable
                chunk_ok = api_ok and parse_ok
                log.debug("llm_va_chunk", chunk=idx + 1, of=total_chunks,
                          candidates_in=len(chunk_candidates),
                          chunk_ok=chunk_ok,
                          detections=len(detections),
                          chunk_ms=round(chunk_ms, 1),
                          preview=raw[:300] if raw else "empty")
                log_llm_chunk(idx + 1, total_chunks, len(detections), chunk_ms, chunk_ok)
                if progress_queue is not None:
                    await progress_queue.put({
                        "type": "progress",
                        "step": 3,
                        "name": "llm_chunk",
                        "chunk": idx + 1,
                        "total_chunks": total_chunks,
                        "ms": round(chunk_ms, 1),
                        "detections": len(detections),
                        "ok": chunk_ok,
                    })
                return detections, chunk_ok

        chunk_results = await asyncio.gather(*[
            _process_chunk(chunk, start, i)
            for i, (chunk, start) in enumerate(chunks_with_pos)
        ])

        all_detections: list[dict] = []
        any_chunk_succeeded = False
        for chunk_detections, chunk_success in chunk_results:
            all_detections.extend(chunk_detections)
            if chunk_success:
                any_chunk_succeeded = True

        llm_succeeded = any_chunk_succeeded
        log.debug("llm_va_raw", count=len(all_detections),
                 llm_succeeded=llm_succeeded,
                 entities=list({d.get("entity_id") for d in all_detections}))

        spans = _locate_spans(all_detections, text, entities_by_id, _LLM_CONFIDENCE)
        log.debug("llm_va_done", spans_located=len(spans), llm_succeeded=llm_succeeded)
        return spans, llm_succeeded


# ── Helpers ───────────────────────────────────────────────────────────────────

def _chunk_text_with_positions(
    text: str, chunk_size: int, overlap: int
) -> list[tuple[str, int]]:
    """Split text into overlapping chunks at whitespace. Returns (chunk, start_offset) pairs."""
    if len(text) <= chunk_size:
        return [(text, 0)]

    result: list[tuple[str, int]] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            break_at = text.rfind("\n", start, end)
            if break_at == -1 or break_at < start + chunk_size // 2:
                break_at = text.rfind(" ", start, end)
            if break_at > start:
                end = break_at + 1

        result.append((text[start:end], start))
        if end >= len(text):
            break
        start = end - overlap

    return result


def _parse_json(raw: str) -> tuple[list[dict], bool]:
    """Parse JSON array from LLM response.

    Returns (items, parse_ok). parse_ok=False means the response contained
    content but was unparseable — callers should treat this as a chunk failure.
    Empty-but-valid responses (e.g. legitimate ``[]``) return ([], True).
    """
    if not raw:
        return [], True  # no response is not a parse error

    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        raw = "\n".join(inner)

    start = raw.find("[")
    if start == -1:
        log.warning("llm_no_json_array", preview=raw[:200])
        return [], False

    end = raw.rfind("]")

    # Repair truncated JSON — token-limit cut-off produces valid prefix but no closing ]
    if end == -1 or end < start:
        repaired = raw[start:].rstrip().rstrip(",") + "]"
        try:
            data = json.loads(repaired)
            items = [d for d in data if isinstance(d, dict)]
            log.warning("llm_json_truncated_repaired",
                        items_recovered=len(items),
                        hint="increase MODEL_XXX_MAX_TOKENS if this recurs")
            return items, True
        except json.JSONDecodeError:
            log.warning("llm_json_truncated_unrecoverable", preview=raw[:200])
            return [], False

    try:
        data = json.loads(raw[start : end + 1])
        return [d for d in data if isinstance(d, dict)], True
    except json.JSONDecodeError as exc:
        log.warning("llm_json_parse_error", error=str(exc), preview=raw[:200])
        return [], False


def _locate_spans(
    detections: list[dict],
    text: str,
    entities_by_id: dict,
    default_confidence: float,
) -> list[DetectedSpan]:
    """Map detected entity texts back to their character offsets in the document."""
    spans: list[DetectedSpan] = []
    seen: set[tuple[int, int, str]] = set()

    for item in detections:
        entity_id = item.get("entity_id", "")
        entity_text = item.get("text", "")

        if not entity_id or not entity_text:
            continue
        if entity_id not in entities_by_id:
            log.debug("llm_unknown_entity_id", entity_id=entity_id, text=entity_text[:40])
            continue

        entity_cfg = entities_by_id[entity_id]
        confidence = float(item.get("confidence", default_confidence))

        for start, end in _find_all_occurrences(entity_text, text):
            key = (start, end, entity_id)
            if key in seen:
                continue
            seen.add(key)
            spans.append(DetectedSpan(
                text=text[start:end],
                start=start,
                end=end,
                entity_id=entity_id,
                display_name=entity_cfg.display_name,
                confidence=confidence,
                source="llm",
            ))

    return spans


def _find_all_occurrences(entity_text: str, document: str) -> list[tuple[int, int]]:
    """Return (start, end) for every occurrence of entity_text in document."""
    results: list[tuple[int, int]] = []

    pos = 0
    while True:
        idx = document.find(entity_text, pos)
        if idx == -1:
            break
        results.append((idx, idx + len(entity_text)))
        pos = idx + 1

    if results:
        return results

    for m in re.finditer(re.escape(entity_text), document, re.IGNORECASE):
        results.append((m.start(), m.end()))

    return results


def _first_sentence(text: str) -> str:
    end = text.find(".")
    return (text[:end].strip() if end > 0 else text[:120].strip())


def _load_template(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8")
