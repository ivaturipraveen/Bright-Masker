from __future__ import annotations

import hashlib
import hmac
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from faker import Faker


def _render(template: str, token_map: dict) -> str:
    class _SafeDict(dict):
        def __missing__(self, key: str) -> str:
            return f"{{{key}}}"

    return template.format_map(_SafeDict(token_map))


def build_token_map(original: str, entity_id: str, faker: "Faker") -> dict:
    digest = hashlib.sha256(original.encode()).hexdigest()
    return {
        "label": entity_id.upper(),
        "hash8": digest[:8],
        "hash4": digest[:4],
        "last4": original[-4:] if len(original) >= 4 else original,
        "fake_name": faker.name(),
        "fake_address": faker.street_address(),
        "fake_city": faker.city(),
        "fake_company": faker.company(),
        "fake_email": faker.email(),
        "fake_phone": faker.phone_number(),
    }


def redact(original: str, format_template: str, token_map: dict) -> str:
    return _render(format_template, token_map)


def substitute(original: str, format_template: str, token_map: dict, faker: "Faker") -> str:
    return _render(format_template, token_map)


def hash_value(original: str, format_template: str, token_map: dict) -> str:
    digest = hashlib.sha256(original.encode()).hexdigest()
    enriched = {**token_map, "hash8": digest[:8], "hash4": digest[:4]}
    return _render(format_template, enriched)


def partial_redact(original: str, format_template: str, token_map: dict) -> str:
    last4 = original[-4:] if len(original) >= 4 else original
    enriched = {**token_map, "last4": last4}
    return _render(format_template, enriched)


def encrypt_deterministic(
    original: str,
    format_template: str,
    token_map: dict,
    key: bytes,
) -> str:
    mac = hmac.new(key, original.encode(), digestmod=hashlib.sha256).hexdigest()
    enriched = {**token_map, "encrypted": mac}
    return _render(format_template, enriched)
