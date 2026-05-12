from __future__ import annotations

from config import AppConfig
from models.schemas import PreprocessedText
from utils.text_utils import (
    detect_format,
    detect_language,
    normalize_encoding,
    normalize_whitespace,
    strip_html,
    strip_markdown,
    walk_json,
)


class Preprocessor:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    async def process(self, text: str) -> PreprocessedText:
        original_length = len(text)

        # 1. Fix encoding artifacts (smart quotes, Windows line endings, etc.)
        text = normalize_encoding(text)

        # 2. Detect format on normalized-but-uncleaned text
        fmt = detect_format(text)

        # 3. Format-specific content stripping
        if fmt == "plain":
            text = strip_html(text)      # strip any HTML remnants
            text = strip_markdown(text)  # strip any Markdown formatting
        # json / csv / xml: leave content intact — format parsers own those

        # 4. Collapse excessive whitespace (always safe after content stripping)
        text = normalize_whitespace(text)

        # 5. Language detection
        language_hint = self._config.global_config.language
        language = detect_language(text) if language_hint == "auto" else language_hint

        return PreprocessedText(
            text=text,
            language=language,
            format=fmt,
            original_length=original_length,
        )

    async def split_structured(
        self, data: dict | list, path: str = ""
    ) -> list[tuple[str, str]]:
        """Recursively walk JSON/dict and return (field_path, string_value) pairs."""
        return walk_json(data, path)
