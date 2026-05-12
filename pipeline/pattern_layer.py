from __future__ import annotations

import asyncio
import re

from config import AppConfig
from models.schemas import DetectedSpan
from utils.logger import get_logger

log = get_logger(__name__)


class PatternLayer:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._analyzer = self._build_analyzer()

    def _build_analyzer(self):
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        if not self._config.presidio_entity_map:
            return None

        nlp_config = {
            "nlp_engine_name": self._config.presidio_nlp_engine,
            "models": [
                {
                    "lang_code": self._config.presidio_language,
                    "model_name": self._config.spacy_model_name,
                }
            ],
        }
        provider = NlpEngineProvider(nlp_configuration=nlp_config)
        nlp_engine = provider.create_engine()

        # Load all recognizers so context-based confidence boosting works.
        # Results are filtered to config entity types in _run_presidio().
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()

        return AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)

    def _run_presidio(self, text: str, language: str) -> list[DetectedSpan]:
        if self._analyzer is None:
            return []

        allowed_types = set(self._config.presidio_entity_map.keys())
        lang = language if language == self._config.presidio_language else self._config.presidio_language

        try:
            results = self._analyzer.analyze(text=text, language=lang)
        except Exception as exc:
            log.warning("presidio_analysis_failed", error=str(exc))
            return []

        spans: list[DetectedSpan] = []
        for result in results:
            if result.score < self._config.presidio_min_score:
                continue
            if result.entity_type not in allowed_types:
                continue
            entity_id = self._config.presidio_entity_map.get(result.entity_type)
            if not entity_id:
                continue
            entity_cfg = self._config.entities_by_id.get(entity_id)
            if not entity_cfg:
                continue
            spans.append(
                DetectedSpan(
                    text=text[result.start : result.end],
                    start=result.start,
                    end=result.end,
                    entity_id=entity_id,
                    display_name=entity_cfg.display_name,
                    confidence=float(result.score),
                    source="pattern",
                )
            )
        return spans

    def _run_custom_patterns(self, text: str) -> list[DetectedSpan]:
        spans: list[DetectedSpan] = []
        for entity in self._config.entities:
            if not entity.patterns:
                continue
            entity_cfg = self._config.entities_by_id.get(entity.id)
            if not entity_cfg:
                continue
            for pattern_str in entity.patterns:
                try:
                    for m in re.finditer(pattern_str, text, re.IGNORECASE | re.MULTILINE):
                        if m.lastindex and m.lastindex >= 1:
                            start, end = m.start(1), m.end(1)
                            value = m.group(1)
                        else:
                            start, end = m.start(), m.end()
                            value = m.group()
                        if not value:
                            continue
                        spans.append(DetectedSpan(
                            text=value,
                            start=start,
                            end=end,
                            entity_id=entity.id,
                            display_name=entity_cfg.display_name,
                            confidence=1.0,
                            source="pattern",
                        ))
                except re.error as exc:
                    log.warning("custom_pattern_error",
                                entity_id=entity.id,
                                pattern=pattern_str[:60],
                                error=str(exc))
        return spans

    def _run_all(self, text: str, language: str) -> list[DetectedSpan]:
        presidio = self._run_presidio(text, language)
        custom = self._run_custom_patterns(text)
        all_spans = presidio + custom

        by_entity: dict[str, list[str]] = {}
        for s in all_spans:
            by_entity.setdefault(s.entity_id, []).append(s.text)

        log.debug("step_2a_pattern_done",
                 presidio_spans=len(presidio),
                 custom_spans=len(custom),
                 total=len(all_spans),
                 by_entity={k: len(v) for k, v in sorted(by_entity.items())},
                 detail=[{"entity": s.entity_id, "text": s.text, "source": s.source}
                         for s in sorted(all_spans, key=lambda x: x.start)])
        return all_spans

    async def analyze(self, text: str, language: str) -> list[DetectedSpan]:
        return await asyncio.to_thread(self._run_all, text, language)
