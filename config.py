from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel

from exceptions import ConfigValidationError
from models.schemas import MaskingStrategy

load_dotenv()

NOT_PII_LABEL = "NOT_PII"


class MaskingConfig(BaseModel):
    strategy: MaskingStrategy
    format: str


class EntityConfig(BaseModel):
    id: str
    display_name: str
    description: str = ""
    enabled: bool = True
    policy: list[str] = ["general"]
    priority: int = 5
    confidence_threshold: Optional[float] = None
    presidio_type: Optional[str] = None
    gliner_label: Optional[str] = None  # override display_name for GLiNER zero-shot
    patterns: list[str] = []
    masking: MaskingConfig
    notes: str = ""

class GlobalConfig(BaseModel):
    default_confidence_threshold: float = 0.85
    default_masking_strategy: str = "redact"
    language: str = "auto"
    # Policy gates: list the policies to activate. Empty [] = all policies active.
    enabled_policies: list[str] = []
    # Field-label words that must never be masked instead of their values.
    # Normalised to lowercase, colon-stripped before matching.
    label_blocklist: list[str] = []
    # Entity IDs whose span text should never cross a newline boundary.
    no_multiline_entity_ids: list[str] = []


class Config:
    def __init__(self, **overrides: Any) -> None:
        def _s(key: str, env: str, default: str) -> str:
            return str(overrides[key]) if key in overrides else os.getenv(env, default)

        def _i(key: str, env: str, default: int) -> int:
            return int(overrides[key]) if key in overrides else int(os.getenv(env, str(default)))

        def _f(key: str, env: str, default: float) -> float:
            return float(overrides[key]) if key in overrides else float(os.getenv(env, str(default)))

        def _b(key: str, env: str, default: bool) -> bool:
            if key in overrides:
                return bool(overrides[key])
            return os.getenv(env, str(default)).lower() in ("true", "1", "yes")

        # ── OpenRouter / Ollama ─────────────────────────────────────────────
        self.openrouter_api_key = _s("openrouter_api_key", "OPENROUTER_API_KEY", "dummy")
        self.openrouter_base_url = _s("openrouter_base_url", "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

        # ── Model names ─────────────────────────────────────────────────────
        self.gliner_model_name = _s("gliner_model_name", "GLINER_MODEL_NAME", "urchade/gliner_medium-v2.1")
        self.llm_model_name = _s("llm_model_name", "LLM_MODEL_NAME", "meta-llama/llama-3.1-8b-instruct")
        self.spacy_model_name = _s("spacy_model_name", "SPACY_MODEL_NAME", "en_core_web_sm")

        # ── Entity config ───────────────────────────────────────────────────
        self.entities_config_path = Path(_s("entities_config_path", "ENTITIES_CONFIG_PATH", "entities_config.yaml"))

        # ── Pattern layer (Presidio) ────────────────────────────────────────
        self.presidio_min_score = _f("presidio_min_score", "PRESIDIO_MIN_SCORE", 0.6)
        self.presidio_nlp_engine = _s("presidio_nlp_engine", "PRESIDIO_NLP_ENGINE", "spacy")
        self.presidio_language = _s("presidio_language", "PRESIDIO_LANGUAGE", "en")

        # ── NER layer (GLiNER local model) ──────────────────────────────────
        self.gliner_threshold = _f("gliner_threshold", "GLINER_THRESHOLD", 0.4)
        self.gliner_max_chunk_chars = _i("gliner_max_chunk_chars", "GLINER_MAX_CHUNK_CHARS", 1200)
        self.gliner_chunk_overlap_chars = _i("gliner_chunk_overlap_chars", "GLINER_CHUNK_OVERLAP_CHARS", 100)

        # ── LLM review layer ─────────────────────────────────────────────────
        self.llm_timeout_seconds = _f("llm_timeout_seconds", "LLM_TIMEOUT_SECONDS", 60.0)
        self.llm_max_retries = _i("llm_max_retries", "LLM_MAX_RETRIES", 3)
        self.llm_max_tokens = _i("llm_max_tokens", "LLM_MAX_TOKENS", 512)
        self.llm_temperature = _f("llm_temperature", "LLM_TEMPERATURE", 0.0)
        self.llm_extra_body: dict = {}  # set at runtime by _apply_model_config
        self.llm_stop_sequences: list = []  # set at runtime by _apply_model_config
        self.llm_context_chars = _i("llm_context_chars", "LLM_CONTEXT_CHARS", 50)
        self.max_llm_batch_size = _i("max_llm_batch_size", "MAX_LLM_BATCH_SIZE", 10)

        # ── Masking engine ──────────────────────────────────────────────────
        self.faker_seed = _i("faker_seed", "FAKER_SEED", 42)
        self.encryption_key = _s("encryption_key", "ENCRYPTION_KEY", "change-this-key-in-production")

        # ── Pipeline ────────────────────────────────────────────────────────
        self.enable_async_layers = _b("enable_async_layers", "ENABLE_ASYNC_LAYERS", True)
        self.batch_max_concurrency = _i("batch_max_concurrency", "BATCH_MAX_CONCURRENCY", 4)

        # ── Logging ─────────────────────────────────────────────────────────
        self.log_level = _s("log_level", "LOG_LEVEL", "INFO")


class AppConfig:
    def __init__(
        self,
        settings: Optional[Config] = None,
        entities_config_path: Optional[Path] = None,
    ) -> None:
        self._settings = settings or Config()
        if entities_config_path is not None:
            self._settings.entities_config_path = Path(entities_config_path)
        self._entities: list[EntityConfig] = []
        self._global_config: GlobalConfig = GlobalConfig()
        self.load()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._settings, name)

    def load(self) -> "AppConfig":
        self._entities = self.load_entities()
        return self

    def load_entities(self) -> list[EntityConfig]:
        path = self._settings.entities_config_path
        if not path.exists():
            raise ConfigValidationError(f"entities_config.yaml not found at {path}")

        with open(path, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)

        self._global_config = GlobalConfig(**data.get("global", {}))

        active_policies: set[str] = set(self._global_config.enabled_policies)

        entities: list[EntityConfig] = []
        for raw in data.get("entities", []):
            entity_id = raw.get("id", "unknown")
            try:
                entity = EntityConfig(**raw)
            except Exception as exc:
                raise ConfigValidationError(
                    f"Invalid entity config for '{entity_id}': {exc}"
                ) from exc
            if not entity.enabled:
                continue
            # If enabled_policies is non-empty, skip entities whose policy
            # has no intersection with the active set.
            if active_policies and not set(entity.policy).intersection(active_policies):
                continue
            entities.append(entity)
        return entities

    @property
    def global_config(self) -> GlobalConfig:
        return self._global_config

    @property
    def entities(self) -> list[EntityConfig]:
        return self._entities

    @property
    def entities_by_id(self) -> dict[str, EntityConfig]:
        return {e.id: e for e in self._entities}

    @property
    def presidio_entity_map(self) -> dict[str, str]:
        return {e.presidio_type: e.id for e in self._entities if e.presidio_type}

    @property
    def gliner_labels(self) -> list[str]:
        return [e.display_name for e in self._entities]

    @property
    def gliner_label_to_entity_id(self) -> dict[str, str]:
        return {(e.gliner_label or e.display_name): e.id for e in self._entities}

    def get_confidence_threshold(self, entity_id: str) -> float:
        entity = self.entities_by_id.get(entity_id)
        if entity and entity.confidence_threshold is not None:
            return entity.confidence_threshold
        return self._global_config.default_confidence_threshold
