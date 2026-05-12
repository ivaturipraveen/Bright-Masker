import logging
import os
import sys
import time
from datetime import datetime
from typing import Any

import structlog

_BAR_WIDE = "=" * 60   # major section borders
_BAR_THIN = "-" * 60   # step section borders


def configure_logging(level: str = "INFO") -> None:
    fmt = os.getenv("LOG_FORMAT", "json").lower()

    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="%H:%M:%S.%f" if fmt == "pretty" else "iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if fmt == "pretty":
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            )
        ]
    else:
        processors = shared_processors + [structlog.processors.JSONRenderer()]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> Any:
    return structlog.get_logger(name)


# ── Low-level print helper ────────────────────────────────────────────────────

def log_line(msg: str = "") -> None:
    """Print a single line to stderr (same stream as structlog)."""
    print(msg, file=sys.stderr, flush=True)


# ── Request banner ────────────────────────────────────────────────────────────

def log_request_start(req_id: str, chars: int, preview: str) -> float:
    """Print the request header banner. Returns perf_counter start time."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_line()
    log_line(_BAR_WIDE)
    log_line("  PII MASKING REQUEST")
    log_line(_BAR_WIDE)
    log_line(f"  req_id  : {req_id}")
    log_line(f"  chars   : {chars:,}")
    log_line(f'  preview : "{preview}"')
    log_line(f"  started : {now}")
    log_line(_BAR_WIDE)
    return time.perf_counter()


# ── Step section helpers ──────────────────────────────────────────────────────

def log_step_header(n: int, total: int, name: str, subtitle: str = "") -> None:
    """Print a step divider + title line."""
    sub = f"  ·  {subtitle}" if subtitle else ""
    log_line()
    log_line(_BAR_THIN)
    log_line(f"  [STEP {n}/{total}]  {name}{sub}")
    log_line(_BAR_THIN)


def log_step_timing(elapsed_s: float, ok: bool) -> None:
    status = "✓" if ok else "✗  FAILED"
    log_line(f"  timing   : {elapsed_s:.3f} s  {status}")
    log_line()


# ── LLM chunk progress ────────────────────────────────────────────────────────

def log_llm_chunk(chunk: int, total: int, detections: int, ms: float, ok: bool) -> None:
    status = "✓" if ok else "✗"
    log_line(f"  chunk [{chunk}/{total}]  →  {detections} detections  ·  {ms / 1000:.3f} s  {status}")


# ── Final summary banner ──────────────────────────────────────────────────────

def log_pipeline_summary(
    total_s: float,
    pattern_ner_s: float,
    llm_s: float,
    llm_ok: bool,
    llm_model: str,
    entities: int,
    language: str,
) -> None:
    llm_status = "succeeded" if llm_ok else "FAILED  →  raw NER fallback used"
    log_line()
    log_line(_BAR_WIDE)
    log_line("  PIPELINE COMPLETE")
    log_line(_BAR_WIDE)
    log_line(f"  total       : {total_s:.2f} s")
    log_line(f"  pattern+ner : {pattern_ner_s:.2f} s")
    log_line(f"  llm review  : {llm_s:.2f} s")
    log_line(f"  llm model   : {llm_model}")
    log_line(f"  llm status  : {llm_status}")
    log_line(f"  entities    : {entities}")
    log_line(f"  language    : {language}")
    log_line(_BAR_WIDE)
    log_line()
