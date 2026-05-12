from __future__ import annotations

from config import AppConfig, EntityConfig
from models.schemas import DetectedSpan
from utils.logger import get_logger

log = get_logger(__name__)

_SOURCE_ORDER = {"pattern": 0, "ner": 1, "llm": 2}


class SpanMerger:
    def merge(
        self,
        spans: list[DetectedSpan],
        threshold: float,
        entity_configs: dict[str, EntityConfig],
        app_config: AppConfig | None = None,
    ) -> tuple[list[DetectedSpan], list[DetectedSpan]]:
        deduped = self._remove_exact_duplicates(spans)
        log.debug("merger_dedup",
                 before=len(spans),
                 after=len(deduped),
                 removed=len(spans) - len(deduped))

        merged = self._resolve_overlaps(deduped, entity_configs)
        log.debug("merger_overlap",
                 before=len(deduped),
                 after=len(merged),
                 removed=len(deduped) - len(merged))

        high_conf, low_conf = self._split_by_threshold(merged, app_config, threshold)
        log.debug("merger_threshold_split",
                 threshold=threshold,
                 high_conf=len(high_conf),
                 low_conf=len(low_conf),
                 low_conf_spans=[(s.entity_id, round(s.confidence, 2)) for s in low_conf])

        return high_conf, low_conf

    def _remove_exact_duplicates(
        self, spans: list[DetectedSpan]
    ) -> list[DetectedSpan]:
        best: dict[tuple[int, int, str], DetectedSpan] = {}
        for span in spans:
            key = (span.start, span.end, span.entity_id)
            if key not in best or span.confidence > best[key].confidence:
                best[key] = span
        return list(best.values())

    def _resolve_overlaps(
        self,
        spans: list[DetectedSpan],
        entity_configs: dict[str, EntityConfig],
    ) -> list[DetectedSpan]:
        sorted_spans = sorted(spans, key=lambda s: s.start)
        result: list[DetectedSpan] = []

        for span in sorted_spans:
            if not result:
                result.append(span)
                continue

            last = result[-1]
            if span.start < last.end:
                winner = self._pick_winner(last, span, entity_configs)
                loser = span if winner is last else last
                log.debug("merger_overlap_resolved",
                          winner_entity=winner.entity_id,
                          winner_text=winner.text,
                          loser_entity=loser.entity_id,
                          loser_text=loser.text)
                result[-1] = winner
            else:
                result.append(span)

        return result

    def _pick_winner(
        self,
        a: DetectedSpan,
        b: DetectedSpan,
        entity_configs: dict[str, EntityConfig],
    ) -> DetectedSpan:
        if a.confidence != b.confidence:
            return a if a.confidence > b.confidence else b

        priority_a = entity_configs.get(a.entity_id, None)
        priority_b = entity_configs.get(b.entity_id, None)
        pa = priority_a.priority if priority_a else 5
        pb = priority_b.priority if priority_b else 5
        if pa != pb:
            return a if pa < pb else b

        sa = _SOURCE_ORDER.get(a.source, 99)
        sb = _SOURCE_ORDER.get(b.source, 99)
        if sa != sb:
            return a if sa <= sb else b

        # Final tiebreaker: prefer the longer span so containing spans (e.g., a full URL)
        # are not discarded in favour of a short substring (e.g., an extracted token).
        return a if (a.end - a.start) >= (b.end - b.start) else b

    def merge_all(
        self,
        spans: list[DetectedSpan],
        entity_configs: dict[str, EntityConfig],
    ) -> list[DetectedSpan]:
        """Merge all spans from all sources without threshold split."""
        deduped = self._remove_exact_duplicates(spans)
        log.debug("merger_all_dedup",
                 before=len(spans),
                 after=len(deduped),
                 removed=len(spans) - len(deduped))

        merged = self._resolve_overlaps(deduped, entity_configs)
        log.debug("merger_all_overlap",
                 before=len(deduped),
                 after=len(merged),
                 removed=len(deduped) - len(merged),
                 entities=sorted({s.entity_id for s in merged}))

        return merged

    def resolve_overlaps(
        self,
        spans: list[DetectedSpan],
        entity_configs: dict[str, EntityConfig],
    ) -> list[DetectedSpan]:
        """Public entry point for post-merge overlap resolution (e.g. final dedup)."""
        deduped = self._remove_exact_duplicates(spans)
        return self._resolve_overlaps(deduped, entity_configs)

    def _split_by_threshold(
        self,
        spans: list[DetectedSpan],
        app_config: AppConfig | None,
        global_threshold: float,
    ) -> tuple[list[DetectedSpan], list[DetectedSpan]]:
        high: list[DetectedSpan] = []
        low: list[DetectedSpan] = []

        for span in spans:
            threshold = global_threshold
            if app_config is not None:
                threshold = app_config.get_confidence_threshold(span.entity_id)

            if span.confidence >= threshold:
                high.append(span)
            else:
                low.append(span)

        return high, low
