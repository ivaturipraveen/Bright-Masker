from __future__ import annotations

import re
import unicodedata
from typing import Optional


# ── Encoding & character normalization ────────────────────────────────────────

def normalize_encoding(text: str) -> str:
    """Normalize encoding artifacts common in medical documents and copy-paste sources."""
    # Null bytes
    text = text.replace("\x00", "")
    # Windows / old-Mac line endings → Unix
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Non-breaking space, zero-width space, soft hyphen
    text = text.replace("\xa0", " ").replace("​", "").replace("­", "")
    # Smart quotes → standard ASCII quotes
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("‘", "'").replace("’", "'")
    # Em/en dash → hyphen (preserves SSN-style numbers like 078-05-1120)
    text = text.replace("–", "-").replace("—", "-")
    # Ellipsis character → three dots
    text = text.replace("…", "...")
    # Unicode canonical decomposition → composed form
    return unicodedata.normalize("NFC", text)


def normalize_whitespace(text: str) -> str:
    """Collapse multi-space runs, normalize excessive blank lines, strip edges."""
    # Multiple spaces/tabs on a line → single space (preserve newlines)
    text = re.sub(r"[ \t]+", " ", text)
    # 3+ consecutive blank lines → exactly 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Trailing space on each line
    text = re.sub(r" +\n", "\n", text)
    return text.strip()


# ── Format detection ──────────────────────────────────────────────────────────

def detect_format(text: str) -> str:
    """Detect high-level text format: json | xml | csv | plain."""
    stripped = text.strip()

    # JSON
    if stripped.startswith(("{", "[")):
        try:
            import json
            json.loads(stripped)
            return "json"
        except (ValueError, Exception):
            pass

    # XML / HTML
    if stripped.startswith("<?xml") or re.match(r"^<[A-Za-z!]", stripped):
        return "xml"

    # CSV — requires consistent column count across at least 3 data rows
    if "\n" in stripped:
        lines = [ln for ln in stripped.splitlines() if ln.strip()]
        if len(lines) >= 3:
            first_count = len(lines[0].split(","))
            if first_count >= 2 and all(
                len(ln.split(",")) == first_count for ln in lines[:5]
            ):
                return "csv"

    return "plain"


# ── Content stripping ─────────────────────────────────────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]{1,200}>")

def has_html(text: str) -> bool:
    return bool(_HTML_TAG_RE.search(text))


def strip_html(text: str) -> str:
    """Remove HTML tags and decode entities. Falls back to regex if bs4 unavailable."""
    if not has_html(text):
        return text
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(text, "html.parser").get_text(separator="\n")
    except ImportError:
        text = _HTML_TAG_RE.sub(" ", text)
        # Basic HTML entity decoding
        text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
        text = text.replace("&nbsp;", " ").replace("&quot;", '"').replace("&#39;", "'")
        return text


def strip_markdown(text: str) -> str:
    """Remove common Markdown formatting, preserving the underlying text content."""
    # Fenced code blocks — keep content
    text = re.sub(r"```[^\n]*\n(.*?)```", r"\1", text, flags=re.DOTALL)
    # Inline code: `value` → value
    text = re.sub(r"`+(.+?)`+", r"\1", text)
    # Bold+italic: ***text***
    text = re.sub(r"\*{3}(.+?)\*{3}", r"\1", text, flags=re.DOTALL)
    # Bold: **text** or __text__
    text = re.sub(r"\*{2}(.+?)\*{2}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"_{2}(.+?)_{2}", r"\1", text, flags=re.DOTALL)
    # ATX Headers: ## Title → Title
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Markdown links: [text](url) → text
    text = re.sub(r"\[([^\]]+)\]\([^\)]*\)", r"\1", text)
    # Images: ![alt](url) → alt
    text = re.sub(r"!\[([^\]]*)\]\([^\)]*\)", r"\1", text)
    # Horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Blockquotes: > text → text
    text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)
    return text


# ── Language detection ────────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    try:
        from langdetect import detect, LangDetectException
        return detect(text)
    except Exception:
        return "en"


# ── Text chunking ─────────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    max_chars: int = 1800,
    overlap_chars: int = 200,
) -> list[tuple[str, int]]:
    """Split text into overlapping chunks at whitespace. Returns (chunk, start_offset) pairs."""
    if len(text) <= max_chars:
        return [(text, 0)]

    chunks: list[tuple[str, int]] = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append((text[start:], start))
            break

        boundary = text.rfind(" ", start + max_chars // 2, end)
        if boundary == -1:
            boundary = end
        else:
            boundary += 1

        chunks.append((text[start:boundary], start))
        next_start = boundary - overlap_chars
        # Ensure we always advance to prevent an infinite loop when overlap >= progress
        start = max(next_start, start + 1)
        if start < 0:
            start = 0

    return chunks


# ── JSON walker ───────────────────────────────────────────────────────────────

def walk_json(data: object, path: str = "") -> list[tuple[str, str]]:
    """Recursively walk JSON-like structure, returning (field_path, string_value) pairs."""
    results: list[tuple[str, str]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            child_path = f"{path}.{key}" if path else key
            results.extend(walk_json(value, child_path))
    elif isinstance(data, list):
        for i, item in enumerate(data):
            child_path = f"{path}[{i}]"
            results.extend(walk_json(item, child_path))
    elif isinstance(data, str) and len(data) > 3:
        results.append((path, data))
    return results
