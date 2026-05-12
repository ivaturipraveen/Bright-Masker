from __future__ import annotations

from config import AppConfig
from exceptions import MaskingError
from models.schemas import DetectedSpan, MaskedSpan, MaskingStrategy
from strategies.masking_strategies import (
    build_token_map,
    encrypt_deterministic,
    hash_value,
    partial_redact,
    redact,
    substitute,
)
from utils.logger import get_logger

log = get_logger(__name__)


class MaskingEngine:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._cache: dict[str, str] = {}
        self._type_counters: dict[str, int] = {}
        self._encryption_key = config.encryption_key.encode()

        from faker import Faker
        self._faker = Faker()
        self._faker.seed_instance(config.faker_seed)

    def mask(
        self, text: str, spans: list[DetectedSpan]
    ) -> tuple[str, list[MaskedSpan]]:
        # Reset per-document state so each request gets sequential counters from 1.
        self._type_counters = {}
        self._cache = {}
        # Pass 1: assign counters in left-to-right order so {n} matches document order.
        for span in sorted(spans, key=lambda s: s.start):
            entity_cfg = self._config.entities_by_id.get(span.entity_id)
            if entity_cfg is None:
                continue
            cache_key = f"{span.entity_id}:{span.text}"
            if cache_key not in self._cache:
                try:
                    masked_value = self._apply_strategy(
                        span.text,
                        span.entity_id,
                        entity_cfg.masking.strategy,
                        entity_cfg.masking.format,
                    )
                    self._cache[cache_key] = masked_value
                except Exception as exc:
                    raise MaskingError(
                        f"Failed to mask '{span.entity_id}': {exc}"
                    ) from exc

        # Pass 2: apply substitutions right-to-left to preserve character offsets.
        sorted_spans = sorted(spans, key=lambda s: s.start, reverse=True)
        masked_spans: list[MaskedSpan] = []
        chars = list(text)

        for span in sorted_spans:
            entity_cfg = self._config.entities_by_id.get(span.entity_id)
            if entity_cfg is None:
                log.warning("unknown_entity_id", entity_id=span.entity_id)
                continue

            original = span.text
            cache_key = f"{span.entity_id}:{original}"
            masked_value = self._cache[cache_key]
            log.debug("masking_applied",
                      entity=span.entity_id,
                      strategy=entity_cfg.masking.strategy.value,
                      original=original,
                      masked=masked_value)

            chars[span.start : span.end] = list(masked_value)
            masked_spans.append(
                MaskedSpan(
                    original=original,
                    masked=masked_value,
                    entity_id=span.entity_id,
                    strategy=entity_cfg.masking.strategy,
                    start=span.start,
                    end=span.end,
                )
            )

        log.debug("masking_summary",
                 total=len(masked_spans),
                 results=[{"entity": ms.entity_id, "original": ms.original, "masked": ms.masked}
                          for ms in masked_spans])
        return "".join(chars), masked_spans

    def _apply_strategy(
        self, original: str, entity_id: str, strategy: MaskingStrategy, format_template: str
    ) -> str:
        n = self._type_counters.get(entity_id, 0) + 1
        self._type_counters[entity_id] = n
        token_map = build_token_map(original, entity_id, self._faker)
        token_map["n"] = n

        if strategy == MaskingStrategy.REDACT:
            return redact(original, format_template, token_map)
        if strategy == MaskingStrategy.SUBSTITUTE:
            return substitute(original, format_template, token_map, self._faker)
        if strategy == MaskingStrategy.HASH:
            return hash_value(original, format_template, token_map)
        if strategy == MaskingStrategy.PARTIAL_REDACT:
            return partial_redact(original, format_template, token_map)
        if strategy == MaskingStrategy.ENCRYPT:
            return encrypt_deterministic(
                original, format_template, token_map, self._encryption_key
            )
        raise MaskingError(f"Unknown masking strategy: {strategy}")
