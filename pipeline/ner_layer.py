from __future__ import annotations

import asyncio
from typing import Any

from config import AppConfig
from exceptions import LayerInitError
from models.schemas import DetectedSpan
from utils.logger import get_logger
from utils.text_utils import chunk_text

log = get_logger(__name__)

_MAX_PARALLEL_NER_CHUNKS = 4


def _best_device() -> str:
    # GLINER_DEVICE=cuda forces GPU; default is CPU to leave VRAM headroom for vLLM
    import os
    forced = os.getenv("GLINER_DEVICE", "").lower()
    if forced in ("cuda", "mps", "cpu"):
        return forced
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        # Default to CPU — 503GB RAM is available and vLLM needs the VRAM
        return "cpu"
    except Exception:
        pass
    return "cpu"


class NerLayer:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._model: Any = None

    def _load_model(self) -> None:
        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise LayerInitError(
                "gliner package not installed — run: pip install gliner"
            ) from exc
        device = _best_device()
        log.info("gliner_loading", model=self._config.gliner_model_name, device=device)
        self._model = GLiNER.from_pretrained(self._config.gliner_model_name)
        try:
            self._model.to(device)
        except Exception:
            pass  # fallback to CPU if MPS move fails
        log.info("gliner_loaded", model=self._config.gliner_model_name, device=device)

    async def initialize(self) -> None:
        await asyncio.to_thread(self._load_model)

    def _predict_chunk(self, chunk: str, offset: int) -> list[DetectedSpan]:
        label_map = self._config.gliner_label_to_entity_id
        labels = list(label_map.keys())
        if not labels or self._model is None:
            return []

        try:
            entities = self._model.predict_entities(
                chunk, labels, threshold=self._config.gliner_threshold
            )
        except Exception as exc:
            log.warning("gliner_predict_failed", error=str(exc))
            return []

        spans: list[DetectedSpan] = []
        for ent in entities:
            entity_id = label_map.get(ent["label"])
            if not entity_id:
                continue
            entity_cfg = self._config.entities_by_id.get(entity_id)
            if not entity_cfg:
                continue
            spans.append(DetectedSpan(
                text=ent["text"],
                start=offset + ent["start"],
                end=offset + ent["end"],
                entity_id=entity_id,
                display_name=entity_cfg.display_name,
                confidence=float(ent["score"]),
                source="ner",
            ))

        return spans

    async def analyze(self, text: str) -> list[DetectedSpan]:
        if self._model is None:
            return []

        chunks = chunk_text(
            text,
            max_chars=self._config.gliner_max_chunk_chars,
            overlap_chars=self._config.gliner_chunk_overlap_chars,
        )

        log.debug("ner_layer_start", chars=len(text), chunks=len(chunks))

        semaphore = asyncio.Semaphore(_MAX_PARALLEL_NER_CHUNKS)

        async def _run_chunk(idx: int, chunk_text_str: str, offset: int) -> list[DetectedSpan]:
            async with semaphore:
                try:
                    spans = await asyncio.to_thread(
                        self._predict_chunk, chunk_text_str, offset
                    )
                    log.debug("ner_chunk_done",
                              chunk=idx + 1,
                              offset=offset,
                              chars=len(chunk_text_str),
                              spans=len(spans),
                              entities=[s.entity_id for s in spans])
                    return spans
                except Exception as exc:
                    log.warning("ner_chunk_failed", chunk=idx + 1, offset=offset, error=str(exc))
                    return []

        chunk_results = await asyncio.gather(*[
            _run_chunk(i, chunk_text_str, offset)
            for i, (chunk_text_str, offset) in enumerate(chunks)
        ])
        all_spans: list[DetectedSpan] = [s for result in chunk_results for s in result]

        # deduplicate: same (start, end, entity_id) → keep highest confidence
        seen: dict[tuple[int, int, str], DetectedSpan] = {}
        for span in all_spans:
            key = (span.start, span.end, span.entity_id)
            if key not in seen or seen[key].confidence < span.confidence:
                seen[key] = span

        deduped = list(seen.values())

        by_entity: dict[str, list[str]] = {}
        for s in deduped:
            by_entity.setdefault(s.entity_id, []).append(s.text)

        log.debug("step_2b_ner_done",
                 chunks=len(chunks),
                 raw_spans=len(all_spans),
                 deduped_spans=len(deduped),
                 by_entity={k: len(v) for k, v in sorted(by_entity.items())},
                 detail=[{"entity": s.entity_id, "text": s.text, "conf": round(s.confidence, 2)}
                         for s in sorted(deduped, key=lambda x: x.start)])

        return deduped
