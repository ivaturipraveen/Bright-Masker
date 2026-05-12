from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel


class MaskingStrategy(str, Enum):
    REDACT = "redact"
    SUBSTITUTE = "substitute"
    HASH = "hash"
    ENCRYPT = "encrypt"
    PARTIAL_REDACT = "partial_redact"


class DetectedSpan(BaseModel):
    text: str
    start: int
    end: int
    entity_id: str
    display_name: str
    confidence: float
    source: Literal["pattern", "ner", "llm"]
    context: str = ""


class MaskedSpan(BaseModel):
    original: str
    masked: str
    entity_id: str
    strategy: MaskingStrategy
    start: int
    end: int


class ProcessingStats(BaseModel):
    total_ms: float
    pattern_ms: float        # actual elapsed time of pattern layer (runs in parallel with NER)
    ner_ms: float            # actual elapsed time of NER layer (runs in parallel with pattern)
    local_ms: float = 0.0   # wall-clock time for the parallel pattern+NER phase (≈ max of above two)
    llm_ms: Optional[float] = None
    llm_called: bool         # True — LLM is always attempted
    llm_succeeded: bool = False  # True if LLM returned valid results; False if API error / fallback used
    llm_model: str = ""
    spans_pattern: int
    spans_ner: int
    spans_llm: int
    spans_total: int
    language: str


class PipelineOutput(BaseModel):
    original_text: str
    masked_text: str
    detected_spans: list[DetectedSpan]
    masked_spans: list[MaskedSpan]
    stats: ProcessingStats


class PreprocessedText(BaseModel):
    text: str
    language: str
    format: Literal["plain", "json", "csv", "xml"]
    original_length: int
