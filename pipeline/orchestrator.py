from __future__ import annotations

import asyncio
import time
import uuid
from typing import Optional

import structlog

from config import AppConfig
from exceptions import LlmApiError
from models.schemas import DetectedSpan, PipelineOutput, ProcessingStats
from pipeline.llm_layer import LlmLayer
from pipeline.masking_engine import MaskingEngine
from pipeline.ner_layer import NerLayer
from pipeline.pattern_layer import PatternLayer
from pipeline.preprocessor import Preprocessor
from pipeline.span_merger import SpanMerger
from utils.logger import (
    get_logger,
    log_line,
    log_pipeline_summary,
    log_request_start,
    log_step_header,
    log_step_timing,
)

log = get_logger(__name__)


def _entity_counts(spans: list[DetectedSpan]) -> str:
    counts: dict[str, int] = {}
    for s in spans:
        counts[s.entity_id] = counts.get(s.entity_id, 0) + 1
    return "  |  ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "—"


_ENT_INDENT = "               "  # 15 spaces — continuation indent for wrapped entity lines


def _entity_count_lines(spans: list[DetectedSpan]) -> list[str]:
    """Wrap entity counts into multiple indented lines so they don't truncate in terminal."""
    counts: dict[str, int] = {}
    for s in spans:
        counts[s.entity_id] = counts.get(s.entity_id, 0) + 1
    if not counts:
        return [_ENT_INDENT + "—"]
    items = [f"{k}:{v}" for k, v in sorted(counts.items())]
    sep = "  |  "
    lines: list[str] = []
    row: list[str] = []
    row_len = 0
    limit = 65
    for item in items:
        added = len(item) + (len(sep) if row else 0)
        if row_len + added > limit and row:
            lines.append(_ENT_INDENT + sep.join(row))
            row = [item]
            row_len = len(item)
        else:
            row.append(item)
            row_len += added
    if row:
        lines.append(_ENT_INDENT + sep.join(row))
    return lines


def _trim_multiline_spans(
    spans: list[DetectedSpan],
    no_multiline_ids: frozenset[str],
) -> list[DetectedSpan]:
    result: list[DetectedSpan] = []
    trimmed_count = 0
    for span in spans:
        if span.entity_id in no_multiline_ids and "\n" in span.text:
            nl = span.text.index("\n")
            trimmed = span.text[:nl].rstrip()
            if len(trimmed) < 2:
                trimmed_count += 1
                continue
            log.debug("trim_multiline_span", entity=span.entity_id,
                      before=span.text[:40].replace("\n", "\\n"), after=trimmed)
            result.append(DetectedSpan(
                text=trimmed,
                start=span.start,
                end=span.start + len(trimmed),
                entity_id=span.entity_id,
                display_name=span.display_name,
                confidence=span.confidence,
                source=span.source,
            ))
            trimmed_count += 1
        else:
            result.append(span)
    if trimmed_count:
        log.debug("multiline_span_trimmed", count=trimmed_count, kept=len(result))
    return result


class PiiMaskingPipeline:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._initialized = False
        self.preprocessor: Optional[Preprocessor] = None
        self.pattern_layer: Optional[PatternLayer] = None
        self.ner_layer: Optional[NerLayer] = None
        self.llm_layer: Optional[LlmLayer] = None
        self.span_merger: Optional[SpanMerger] = None
        self.masking_engine: Optional[MaskingEngine] = None
        self._no_multiline_ids: frozenset[str] = frozenset()

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        gc = self._config.global_config
        self._no_multiline_ids = frozenset(gc.no_multiline_entity_ids)

        self.preprocessor = Preprocessor(self._config)
        self.pattern_layer = PatternLayer(self._config)
        self.ner_layer = NerLayer(self._config)
        await self.ner_layer.initialize()
        self.llm_layer = LlmLayer(self._config)
        self.span_merger = SpanMerger()
        self.masking_engine = MaskingEngine(self._config)
        self._initialized = True
        log.info("pipeline_initialized",
                 entities=len(self._config.entities),
                 no_multiline_ids=len(self._no_multiline_ids))

    def _apply_filters(self, spans: list[DetectedSpan]) -> list[DetectedSpan]:
        return _trim_multiline_spans(spans, self._no_multiline_ids)

    async def process(
        self,
        text: str,
        progress_queue: asyncio.Queue | None = None,
    ) -> PipelineOutput:
        await self._ensure_initialized()

        doc_id = uuid.uuid4().hex[:8]
        structlog.contextvars.bind_contextvars(doc_id=doc_id)

        try:
            return await self._process(text, doc_id, progress_queue)
        finally:
            structlog.contextvars.clear_contextvars()

    async def _process(
        self,
        text: str,
        doc_id: str,
        progress_queue: asyncio.Queue | None = None,
    ) -> PipelineOutput:

        async def emit(event: dict) -> None:
            if progress_queue is not None:
                await progress_queue.put(event)

        # ── Request banner ────────────────────────────────────────────────────
        t0 = log_request_start(doc_id, len(text), text[:60].replace("\n", " "))
        log.info("pipeline_start", chars=len(text), preview=text[:80].replace("\n", " "))

        # ── [1/5] Preprocessor ────────────────────────────────────────────────
        log_step_header(1, 5, "PREPROCESSOR")
        t1 = time.perf_counter()

        preprocessed = await self.preprocessor.process(text)

        pre_s = time.perf_counter() - t1
        log_line(f"  language : {preprocessed.language}")
        log_line(f"  format   : {preprocessed.format}")
        log_line(f"  chars    : {len(preprocessed.text):,}")
        log_step_timing(pre_s, True)

        log.debug("step_1_preprocessor",
                 language=preprocessed.language,
                 format=preprocessed.format,
                 chars_in=len(text),
                 chars_out=len(preprocessed.text))

        await emit({"type": "progress", "step": 1, "name": "preprocessor",
                    "ms": round(pre_s * 1000, 1),
                    "language": preprocessed.language,
                    "format": preprocessed.format,
                    "chars": len(preprocessed.text)})

        # ── [2/5] Pattern + NER (parallel) ───────────────────────────────────
        log_step_header(2, 5, "PATTERN + NER", "parallel local")
        t2 = time.perf_counter()

        async def _timed_pattern():
            t = time.perf_counter()
            spans = await self.pattern_layer.analyze(preprocessed.text, preprocessed.language)
            return spans, (time.perf_counter() - t) * 1000

        async def _timed_ner():
            t = time.perf_counter()
            spans = await self.ner_layer.analyze(preprocessed.text)
            return spans, (time.perf_counter() - t) * 1000

        (pattern_spans, pattern_ms), (ner_spans, ner_ms) = await asyncio.gather(
            _timed_pattern(), _timed_ner()
        )

        local_s = time.perf_counter() - t2  # wall-clock of the parallel phase
        local_ms = local_s * 1000

        pattern_entities = sorted({s.entity_id for s in pattern_spans})
        ner_entities = sorted({s.entity_id for s in ner_spans})
        only_in_ner = sorted(set(ner_entities) - set(pattern_entities))
        only_in_pattern = sorted(set(pattern_entities) - set(ner_entities))

        log_line(f"  pattern  : {len(pattern_spans):>3} spans  ({pattern_ms:.0f} ms)")
        for _l in _entity_count_lines(pattern_spans):
            log_line(_l)
        log_line(f"  ner      : {len(ner_spans):>3} spans  ({ner_ms:.0f} ms)")
        for _l in _entity_count_lines(ner_spans):
            log_line(_l)
        if only_in_ner:
            log_line(f"  ner only : {only_in_ner}")
        if only_in_pattern:
            log_line(f"  pat only : {only_in_pattern}")
        log_step_timing(local_s, True)

        log.debug("step_2_local_detection_summary",
                 local_ms=round(local_ms, 1),
                 pattern_spans=len(pattern_spans),
                 ner_spans=len(ner_spans),
                 pattern_entities=pattern_entities,
                 ner_entities=ner_entities,
                 ner_only_entities=only_in_ner,
                 pattern_only_entities=only_in_pattern)

        await emit({"type": "progress", "step": 2, "name": "pattern_ner",
                    "ms": round(local_ms, 1),
                    "pattern_spans": len(pattern_spans),
                    "ner_spans": len(ner_spans),
                    "pattern_entities": pattern_entities,
                    "ner_entities": ner_entities})

        # ── [3/5] LLM Validate + Augment ─────────────────────────────────────
        log_step_header(3, 5, "LLM VALIDATE + AUGMENT",
                        f"{len(ner_spans)} NER candidates  ·  model: {self._config.llm_model_name}")
        log_line(f"  model    : {self._config.llm_model_name}")
        t3 = time.perf_counter()

        llm_spans, llm_ok = await self.llm_layer.validate_and_augment(
            preprocessed.text, ner_spans, self._config.entities_by_id,
            progress_queue=progress_queue,
        )

        llm_s = time.perf_counter() - t3
        llm_ms = llm_s * 1000

        ner_entity_set = {s.entity_id for s in ner_spans}
        llm_entity_set = {s.entity_id for s in llm_spans}
        ner_removed = sorted(ner_entity_set - llm_entity_set)
        llm_added = sorted(llm_entity_set - ner_entity_set)

        log_line()
        log_line(f"  spans out   : {len(llm_spans)}")
        log_line(f"  dropped FPs : {len(ner_removed)}")
        log_line(f"  new added   : {len(llm_added)}")
        log_step_timing(llm_s, llm_ok)

        log.debug("step_3_llm_done",
                 llm_succeeded=llm_ok,
                 llm_ms=round(llm_ms, 1),
                 spans_out=len(llm_spans),
                 entities_kept=sorted(llm_entity_set & ner_entity_set),
                 entities_dropped_by_llm=ner_removed,
                 entities_added_by_llm=llm_added,
                 detail=[{"entity": s.entity_id, "text": s.text}
                         for s in sorted(llm_spans, key=lambda x: x.start)])

        await emit({"type": "progress", "step": 3, "name": "llm",
                    "ms": round(llm_ms, 1),
                    "llm_ok": llm_ok,
                    "spans_out": len(llm_spans),
                    "dropped": len(ner_removed),
                    "added": len(llm_added)})

        if not llm_ok:
            log.error("llm_mandatory_failed",
                      model=self._config.llm_model_name,
                      llm_ms=round(llm_ms, 1))
            raise LlmApiError(
                f"LLM validation failed after all retries "
                f"(model: {self._config.llm_model_name}) — request rejected"
            )

        merge_input = pattern_spans + llm_spans

        # ── [4/5] Span Merge + Filter ─────────────────────────────────────────
        log_step_header(4, 5, "SPAN MERGE + FILTER")
        t4 = time.perf_counter()

        merged = self.span_merger.merge_all(merge_input, self._config.entities_by_id)
        merged = self._apply_filters(merged)

        t4_s = time.perf_counter() - t4
        merge_by_entity: dict[str, list[str]] = {}
        for s in merged:
            merge_by_entity.setdefault(s.entity_id, []).append(s.text)

        log_line(f"  input    : {len(merge_input)} spans")
        log_line(f"  output   : {len(merged)} spans")
        log_line(f"  by entity:")
        for _l in _entity_count_lines(merged):
            log_line(_l)
        log_step_timing(t4_s, True)

        log.debug("step_4_merge_done",
                 spans_in=len(merge_input),
                 spans_out=len(merged),
                 dropped=len(merge_input) - len(merged),
                 by_entity={k: len(v) for k, v in sorted(merge_by_entity.items())},
                 final_spans=[{"entity": s.entity_id, "text": s.text, "source": s.source}
                               for s in sorted(merged, key=lambda x: x.start)])

        await emit({"type": "progress", "step": 4, "name": "merge",
                    "ms": round(t4_s * 1000, 1),
                    "spans_in": len(merge_input),
                    "spans_final": len(merged),
                    "by_entity": {k: len(v) for k, v in sorted(merge_by_entity.items())}})

        # ── [5/5] Masking ─────────────────────────────────────────────────────
        log_step_header(5, 5, "MASKING")
        t5 = time.perf_counter()

        masked_text, masked_spans = self.masking_engine.mask(preprocessed.text, merged)

        t5_s = time.perf_counter() - t5
        log_line(f"  masked   : {len(masked_spans)} entities")
        log_line(f"  by entity:")
        for _l in _entity_count_lines(merged):
            log_line(_l)
        log_step_timing(t5_s, True)

        log.debug("step_5_masking_done",
                 total_masked=len(masked_spans),
                 masked_entities=sorted({ms.entity_id for ms in masked_spans}),
                 masked_values=[{"entity": ms.entity_id,
                                  "original": ms.original,
                                  "masked": ms.masked}
                                 for ms in masked_spans])

        await emit({"type": "progress", "step": 5, "name": "masking",
                    "ms": round(t5_s * 1000, 1),
                    "entities_masked": len(masked_spans)})

        # ── Summary ───────────────────────────────────────────────────────────
        total_s = time.perf_counter() - t0
        total_ms = total_s * 1000

        log_pipeline_summary(
            total_s=total_s,
            pattern_ner_s=local_s,
            llm_s=llm_s,
            llm_ok=llm_ok,
            llm_model=self._config.llm_model_name,
            entities=len(merged),
            language=preprocessed.language,
        )

        log.debug("pipeline_done",
                 total_ms=round(total_ms, 1),
                 pattern_ms=round(pattern_ms, 1),
                 ner_ms=round(ner_ms, 1),
                 local_ms=round(local_ms, 1),
                 llm_ms=round(llm_ms, 1),
                 llm_called=True,
                 llm_succeeded=llm_ok,
                 spans_pattern=len(pattern_spans),
                 spans_ner=len(ner_spans),
                 spans_llm=len(llm_spans),
                 spans_final=len(merged))

        return PipelineOutput(
            original_text=text,
            masked_text=masked_text,
            detected_spans=merged,
            masked_spans=masked_spans,
            stats=ProcessingStats(
                total_ms=total_ms,
                pattern_ms=pattern_ms,    # individual pattern layer time
                ner_ms=ner_ms,            # individual NER layer time
                local_ms=local_ms,        # wall-clock for parallel phase (≈ max of above two)
                llm_ms=llm_ms,
                llm_called=True,          # LLM is always attempted
                llm_succeeded=llm_ok,     # always True here — False causes LlmApiError above
                llm_model=self._config.llm_model_name,
                spans_pattern=len(pattern_spans),
                spans_ner=len(ner_spans),
                spans_llm=len(llm_spans),
                spans_total=len(merged),
                language=preprocessed.language,
            ),
        )

    async def process_batch(
        self, texts: list[str], max_concurrency: int | None = None
    ) -> list[PipelineOutput]:
        await self._ensure_initialized()
        semaphore = asyncio.Semaphore(max_concurrency or self._config.batch_max_concurrency)

        async def _bounded(text: str) -> PipelineOutput:
            async with semaphore:
                return await self.process(text)

        return await asyncio.gather(*[_bounded(t) for t in texts])

    def process_sync(self, text: str) -> PipelineOutput:
        return asyncio.run(self.process(text))
