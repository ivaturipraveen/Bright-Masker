# PII Masker — Architecture

## Pipeline Overview

```
Raw Text
   │
   ▼
┌─────────────────────────────────────────────────────────────────┐
│ [1] Preprocessor                                        ~1ms    │
│                                                                 │
│  Input : raw string                                             │
│  Steps : normalize_encoding → detect_format                     │
│          → strip_html + strip_markdown (plain text only)        │
│          → normalize_whitespace → detect_language               │
│  Output: PreprocessedText(text, language, format,               │
│                           original_length)                      │
└──────────────────────────────┬──────────────────────────────────┘
                               │  preprocessed.text
               ┌───────────────┴───────────────┐
               │      asyncio.gather()          │  ← runs in parallel (no API)
               ▼                               ▼
┌──────────────────────────┐  ┌─────────────────────────────────────┐
│ [2a] PatternLayer  ~5ms  │  │ [2b] NerLayer                ~300ms │
│                          │  │                                     │
│  Input : text, language  │  │  Input : text                       │
│                          │  │  Model : urchade/gliner_large-v2.1  │
│  Presidio regex engine   │  │          local CPU, no API call      │
│  + custom YAML patterns  │  │                                     │
│  from entities_config    │  │  Labels passed to GLiNER:           │
│                          │  │    gliner_label (if set) or         │
│  spaCy: en_core_web_lg   │  │    display_name for every enabled   │
│                          │  │    entity in entities_config.yaml   │
│  source = "pattern"      │  │    e.g. "person name, full name",   │
│                          │  │    "prescription medication name    │
│  Output: list[           │  │     with dosage", "GPS coordinates" │
│    DetectedSpan]         │  │                                     │
│                          │  │  Chunked: 1200 chars / 100 overlap  │
│                          │  │  Threshold: 0.25 (low — candidate   │
│                          │  │    generator, LLM validates later)  │
│                          │  │  source = "ner"                     │
│                          │  │                                     │
│                          │  │  Output: list[DetectedSpan]         │
└──────────────────────────┘  └─────────────────────────────────────┘
        pattern_spans                      ner_spans
               │                               │
               │                               ▼
               │          ┌─────────────────────────────────────────┐
               │          │ [3] LlmLayer.validate_and_augment() ~5s │
               │          │                                         │
               │          │  Input : text, ner_spans only           │
               │          │  ↑ Pattern spans bypass LLM —          │
               │          │    regex is deterministic, no need      │
               │          │    for model validation                 │
               │          │                                         │
               │          │  SYSTEM PROMPT:                         │
               │          │    {entity_list} → per enabled entity:  │
               │          │      entity_id (Display Name) —         │
               │          │      first sentence of description      │
               │          │    VALIDATE rules (false positive       │
               │          │      removal: headers, role words,      │
               │          │      credentials, geo context)          │
               │          │    AUGMENT rules (find missed PII)      │
               │          │    Date disambiguation rules            │
               │          │                                         │
               │          │  USER PROMPT:                           │
               │          │    {candidates_json} ← NER spans only   │
               │          │    {text}            ← document chunk   │
               │          │                                         │
               │          │  Chunking: 3500 chars / 200 overlap     │
               │          │  Parallel: asyncio.Semaphore(3)         │
               │          │  Model: configurable at runtime via      │
               │          │    DEFAULT_MODEL env / POST /config/model│
               │          │                                         │
               │          │  JSON repair: if response truncated      │
               │          │    (no closing ]), repaired before      │
               │          │    parse; if unrecoverable → chunk      │
               │          │    marked failed → request rejected 503 │
               │          │                                         │
               │          │  Output: list[DetectedSpan]             │
               │          │          source = "llm"                 │
               │          └────────────────┬────────────────────────┘
               │                           │  llm_spans
               └──────────────┬────────────┘
                              │  LLM is mandatory — request fails (503)
                              │  if LLM validation cannot complete
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ [4] SpanMerger.merge_all()                              ~1ms    │
│                                                                 │
│  Input : pattern_spans + llm_spans, entities_by_id             │
│  Step 1: Remove exact duplicates (same start+end+entity_id)     │
│           keep highest confidence                               │
│  Step 2: Resolve overlapping spans — winner order:              │
│           1. confidence (higher wins)                           │
│           2. entity priority (lower number wins)                │
│           3. source order  pattern > llm > ner                  │
│           4. span length  (longer span wins on full tie)        │
│  Output: list[DetectedSpan]  non-overlapping, sorted by start   │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ [5] _trim_multiline_spans()                             ~1ms    │
│                                                                 │
│  Structural constraint only — no PII logic                      │
│  17 entity types (ssn, phone, email, date_of_birth, etc.)       │
│  are trimmed to the first line if the span crosses a newline    │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ [6] MaskingEngine                                       ~1ms    │
│                                                                 │
│  Input : text, merged_spans                                     │
│  Reads masking strategy per entity from entities_config.yaml    │
│  Processes spans right-to-left (preserves char offsets)         │
│  Session cache: same original text → same masked token          │
│  Numbered tokens: [PERSON 1], [PERSON 2] per document           │
│  Output: masked_text, list[MaskedSpan]                          │
└──────────────────────────────┬──────────────────────────────────┘
                               │
                               ▼
                      PipelineOutput
              (masked_text, spans, stats, timing)
```

---

## Production Models

All model configs live in `.env` under `MODEL_XXX_*` keys — zero Python changes to swap or tune.

### Fixed local models

| Model | Env Var | Where | Avg Latency |
|---|---|---|---|
| `urchade/gliner_large-v2.1` | `GLINER_MODEL_NAME` | Local CPU | ~300ms |
| `en_core_web_lg` | `SPACY_MODEL_NAME` | Local CPU | ~5ms |

### Selectable LLM backends (runtime switchable)

Three LLM backends are registered in `MODEL_REGISTRY` in `app.py`. The active one is set by `DEFAULT_MODEL` in `.env` and can be hot-swapped at runtime via `POST /config/model` without restart.

| Key | Model | Provider | Latency | Retries | Max Tokens |
|---|---|---|---|---|---|
| `openrouter_7b` *(default)* | qwen/qwen-2.5-7b-instruct | OpenRouter | ~5–10s | 2 | 1024 |
| `openrouter_72b` | qwen/qwen-2.5-72b-instruct | OpenRouter | ~15–30s | 2 | 512 |
| `private_35b` | qwen3.6:35b-a3b | Private Cloud | ~60–180s | 1 | 2048 |

All per-model values (`NAME`, `BASE_URL`, `API_KEY`, `MAX_TOKENS`, `TIMEOUT`, `MAX_RETRIES`) are env vars — no hardcodings in Python.

**Why these models:**
- **GLiNER large** (not medium): benchmark showed medium misses ~22% more entities; large generates better candidates for LLM validation
- **Qwen 2.5 7B** (default): fastest; strong structured JSON output; sufficient for NER validation task since pattern spans bypass LLM
- **Qwen 2.5 72B**: best instruction following; use for complex/ambiguous documents
- **Private Cloud 35B MOE**: data never leaves your infrastructure; slower but fully air-gapped
- **spaCy lg** (not sm): significantly better person/org NER in Presidio pattern matching

**Tested alternatives (benchmark 2026-05-08):**

| GLiNER | LLM | Avg recall | Avg latency | Notes |
|---|---|---|---|---|
| gliner_large | qwen-2.5-72b | ~95% | ~10s | Best quality |
| gliner_large | qwen-2.5-7b | ~90% | ~8s | **Default** — best speed/quality balance |
| gliner_large | llama-3.3-70b:free | ~90% | ~35s | Free, rate-limited on long docs |
| gliner_medium | qwen-2.5-72b | ~78% | ~8s | Medium GLiNER hurts recall |

---

## What Passes Through Each Step

```
Raw text
  │
  │ normalize_encoding(text)          fix smart quotes, CRLF, NBSP, em-dash
  │ strip_html / strip_markdown       remove tags and formatting symbols
  │ normalize_whitespace              collapse multi-spaces, cap blank lines
  │ detect_language                   → "en" (or override via config)
  ▼
preprocessed.text   [clean plain text, same char positions as input]
  │
  ├─→ PatternLayer.analyze(text, language)
  │     Presidio engine + spaCy + custom YAML regex patterns
  │     84 entities have at least one pattern
  │     Output: [{entity_id, text, start, end, confidence, source="pattern"}]
  │
  └─→ NerLayer.analyze(text)
        text chunked at 1200 chars / 100 overlap
        GLiNER receives: (chunk_text, [gliner_label or display_name per entity])
        gliner_label overrides display_name for 40 entities with cleaner NER labels
        GLiNER returns raw predictions with start/end within chunk
        Offsets adjusted back to document positions
        Threshold: 0.25 (low — candidate generator; LLM validates)
        Output: [{entity_id, text, start, end, confidence, source="ner"}]
  │
  │  pattern_spans bypass LLM (deterministic regex needs no model validation)
  │  only ner_spans go to LLM
  │
  └─→ LlmLayer.validate_and_augment(text, ner_spans, entities_by_id)
        SYSTEM prompt contains:
          - Full entity list: entity_id (Display Name) — first sentence of description
          - All 103 enabled entities included
          - VALIDATE rules (what to remove: headers, role words, geo context, credentials)
          - AUGMENT rules (what to add: PII missed by NER)
          - Date disambiguation (clinical_date vs DOB vs card_expiration_date)
          - Org subtype rules (hospital_name vs bank_name vs organization_name)
        USER prompt contains:
          - ner_spans serialised as JSON array (NER candidates only)
          - full document chunk
        LLM returns JSON array of validated + augmented spans
        _locate_spans() converts text → (start, end) offsets
        Output: [{entity_id, text, start, end, confidence, source="llm"}]
        JSON repair: truncated responses (no closing ]) repaired automatically;
        if unrecoverable → chunk marked failed → request rejected with 503
  │
  merged_input = pattern_spans + llm_spans  (LLM is mandatory — no fallback)
  │
  └─→ SpanMerger.merge_all(merged_input)
        dedup by (start, end, entity_id) — keep highest confidence
        overlap resolution: confidence → entity priority → source → span length
        source order: pattern > llm > ner
        span length: longer span wins on full tie (e.g. full URL beats token substring)
        Output: non-overlapping sorted span list
  │
  └─→ _trim_multiline_spans()
        structural trim for 17 entity types (ssn, phone, email, date_of_birth, etc.)
  │
  └─→ MaskingEngine.mask(text, spans)
        per-entity strategy from entities_config.yaml
        Output: masked_text, [{entity_id, original, masked, start, end}]
```

---

## Prompt Files

```
prompts/
  llm_validate_augment_system.txt   ← PART A validate + PART B augment
  llm_validate_augment_user.txt     ← {candidates_json} + {text}
```

**Template placeholders** (use `.replace()`, NOT `.format()` — prompts contain JSON with `{` braces):

| Placeholder | Filled with | File |
|---|---|---|
| `{entity_list}` | `entity_id (Display Name) — first sentence of description` per entity | system |
| `{candidates_json}` | JSON array of NER candidate spans only (not pattern spans) | user |
| `{text}` | Document chunk text | user |

---

## Entity Configuration (`entities_config.yaml`)

Single source of truth for all PII behaviour. **103 entities enabled** across 5 policy groups.

```yaml
global:
  default_confidence_threshold: 0.85
  default_masking_strategy: redact
  no_multiline_entity_ids:          # structural constraint — 17 IDs
    - ssn
    - phone_number
    - email_address
    - date_of_birth
    - bank_account_number
    # … 12 more
  label_blocklist: []               # empty — LLM handles label vs value in context
  enabled_policies: []              # empty = all 5 policies active
                                    # set [hipaa, pci_dss] to load only those

entities:
  - id: person_name
    display_name: "Person / Full Name / Alias"
    description: "Human personal names only: first name, last name, full name, aliases"
    gliner_label: "person name, full name, alias"  # optional: cleaner label for GLiNER
    enabled: true
    priority: 2
    presidio_type: PERSON            # optional: enables PatternLayer detection via Presidio
    policy: hipaa
    patterns:                        # optional: custom YAML regex (capturing group = value)
      - '(?:Dr\.|Prof\.) ([A-Z][a-z]+ [A-Z][a-z]+)'
    masking:
      strategy: redact
      format: "[PERSON {n}]"         # {n} = per-entity counter in document
```

### Entity detection methods per entity

Every entity uses at least one detection method. Most use two or three:

| Method | How it works | Count |
|---|---|---|
| **Pattern** | Regex in entities_config.yaml (structured IDs: SSN, VIN, phone, email, …) | 84 entities |
| **Presidio** | Presidio NLP engine via `presidio_type` field | 14 entities |
| **GLiNER** | Zero-shot NER with `gliner_label` or `display_name` as label | all 103 entities |
| **LLM** | Validates GLiNER output + augments with `description` as context | all 103 entities |

### Entity groups by detection strategy

| Group | Examples | Primary detection |
|---|---|---|
| Structured IDs | SSN, credit card, VIN, phone, email, NPI | Pattern regex (exact format) |
| Semantic entities | person name, city, medication, race, religion | GLiNER + LLM |
| CJIS records | warrant, probation, stolen vehicle, CHRI | Keyword-anchored pattern + GLiNER + LLM |
| Org subtypes | hospital_name, bank_name, law_firm_name | GLiNER + LLM (+ safety-net pattern) |

**Policy groups:** `hipaa` (33), `pci_dss` (15), `general` (21), `law_enforcement` (23), `transportation` (4)  
(Sum > 103 because some entities belong to multiple policies)

**To add an entity:** add a YAML block and restart. Zero Python changes.

**`gliner_label` field:** optional per-entity override. When set, this string is sent to GLiNER instead of `display_name`. Use it when `display_name` is a formatted label (e.g. "Biometric — Facial Recognition / Photograph") and you want a clean NER-friendly phrase (e.g. "facial recognition or face scan data").

**Masking format tokens:**

| Token | Value |
|---|---|
| `{n}` | Sequential counter per entity type per document |
| `{label}` | Entity ID in uppercase |
| `{hash8}` | First 8 chars of SHA-256 of original value |
| `{last4}` | Last 4 characters of original value |
| `{fake_name}` | Faker-generated person name |
| `{fake_email}` | Faker-generated email address |
| `{fake_phone}` | Faker-generated phone number |
| `{fake_company}` | Faker-generated company name |

---

## Project Structure

```
pii_masker/
├── app.py                        FastAPI server (8 endpoints incl. SSE streaming + runtime model switch)
├── main.py                       CLI entry point
├── config.py                     Loads .env + entities_config.yaml → AppConfig
├── entities_config.yaml          All entity definitions — single source of truth
├── .env                          Models, API keys, tuning params
│
├── pipeline/
│   ├── orchestrator.py           Chains all layers; only filter: _trim_multiline_spans
│   ├── preprocessor.py           [1] Encoding fix, format detect, strip markup
│   ├── pattern_layer.py          [2a] Presidio + spaCy + custom regex
│   ├── ner_layer.py              [2b] GLiNER zero-shot NER, local CPU
│   ├── llm_layer.py              [3] validate_and_augment via OpenRouter
│   ├── span_merger.py            [4] Dedup + overlap resolution
│   └── masking_engine.py         [6] Apply masking strategies
│
├── prompts/
│   ├── llm_validate_augment_system.txt   LLM system prompt (validate + augment)
│   └── llm_validate_augment_user.txt     LLM user prompt ({candidates_json} + {text})
│
├── models/
│   └── schemas.py                DetectedSpan, MaskedSpan, PipelineOutput
│
├── strategies/
│   └── masking_strategies.py     redact, substitute, hash, partial_redact, encrypt
│
├── utils/
│   ├── logger.py                 Structured logging (never logs PII values)
│   └── text_utils.py             normalize_encoding, strip_html, strip_markdown,
│                                 normalize_whitespace, detect_language, chunking
│
├── benchmark.py                  Multi-model accuracy benchmark (no server needed)
└── test_accuracy.py              Endpoint accuracy test suite (hits /mask)
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Web UI |
| POST | `/mask` | `{"text": "…"}` → `{masked_text, spans, stats, warnings}` |
| POST | `/mask/stream` | Same as `/mask` but SSE — streams per-step progress events then `{"type":"complete","result":{…}}` |
| GET | `/health` | Server status, active model key, model names, entity count |
| GET | `/entities` | All enabled entities with strategies |
| GET | `/config/models` | List all selectable LLM backends + which is active |
| POST | `/config/model` | `{"model_key": "openrouter_7b"}` — hot-swap LLM at runtime, no restart |
| GET | `/docs` | FastAPI Swagger UI |

**`/mask` — LLM is mandatory:**  
If LLM validation fails (all retries exhausted, JSON unrecoverable), the request is rejected with `503`. No partial results are ever returned. Check `detail` in the 503 response for the specific model + reason.

**`/mask/stream` SSE event types:**
```
{"type":"progress","step":1,"name":"preprocessor",...}
{"type":"progress","step":2,"name":"pattern_ner",...}
{"type":"progress","step":3,"name":"llm_chunk","chunk":1,"total_chunks":2,...}
{"type":"progress","step":4,"name":"merge",...}
{"type":"progress","step":5,"name":"masking",...}
{"type":"complete","result":{...full MaskResponse...}}
{"type":"error","message":"..."}   ← only on exception
```

---

## No-Hardcoding Rule

| What | Lives in |
|---|---|
| Entity IDs, display names, descriptions | `entities_config.yaml` |
| GLiNER zero-shot labels | `gliner_label` (if set) or `display_name` in `entities_config.yaml` |
| LLM entity list with descriptions | Injected at runtime from `entities_config.yaml` |
| Masking strategies and format strings | `entities_config.yaml` |
| Per-entity confidence thresholds | `entities_config.yaml` (per entity) |
| Prompt text | `prompts/*.txt` |
| LLM model names | `MODEL_7B_NAME` / `MODEL_72B_NAME` / `MODEL_PRIVATE_NAME` in `.env` |
| LLM base URLs | `MODEL_7B_BASE_URL` / `MODEL_72B_BASE_URL` / `MODEL_PRIVATE_BASE_URL` in `.env` |
| LLM API keys | `MODEL_7B_API_KEY` / `MODEL_72B_API_KEY` / `MODEL_PRIVATE_API_KEY` in `.env` |
| Per-model timeouts, retries, max_tokens | `MODEL_XXX_TIMEOUT` / `MODEL_XXX_MAX_RETRIES` / `MODEL_XXX_MAX_TOKENS` in `.env` |
| Default active LLM on startup | `DEFAULT_MODEL` in `.env` |
| Local model names (GLiNER, spaCy) | `GLINER_MODEL_NAME` / `SPACY_MODEL_NAME` in `.env` |
| Default confidence threshold | `.env` |

Zero Python changes needed to: add an entity, change a masking format, swap a model, adjust a threshold, or change the active LLM backend.

---

## Production Readiness Assessment

### What is production-grade

| Area | Status | Notes |
|---|---|---|
| Entity coverage | 103 entities, 5 policy groups | All entities have ≥2 detection methods |
| Pattern quality | 84 of 103 entities have regex patterns | Word-boundary anchored, no IGNORECASE traps |
| GLiNER labels | All 103 have clean NER labels | 40 overridden via `gliner_label` for better zero-shot |
| LLM context | All 103 entities sent with descriptions | Model knows exactly what to look for |
| Overlap resolution | 4-level tiebreaker | confidence → priority → source → length |
| Async pipeline | Pattern + NER run in parallel | No blocking I/O in local layers |
| Chunking | Both GLiNER and LLM chunk large docs | No token limit failures |
| Structured logging | All steps log span counts, entity types | Never logs PII values |
| Config-driven | Zero hardcoding in pipeline code | All behaviour driven by YAML + .env |
| Graceful error handling | LLM retries (per-model), timeout, rate-limit backoff | Returns 503 on exhaustion — never returns partial results |
| JSON repair | Truncated LLM responses repaired before parse | Handles token-limit cut-off without losing detections |
| Runtime model switching | 3 LLM backends, hot-swappable via API or env | No restart needed to change model |
| LLM mandatory | LLM validation is required — no fallback | 503 returned if LLM unavailable; no silent degradation |

### Known limitations

| Limitation | Impact | Mitigation |
|---|---|---|
| LLM required for NER validation | If LLM unavailable or JSON unrecoverable, request fails with 503 | Ensure LLM endpoint is reachable and API key is valid before serving traffic |
| OCR / garbled text | Malformed tokens won't match patterns or NER | Pre-process with OCR correction before pipeline |
| Semantic entities need context | GLiNER may miss `race`, `religion` in ambiguous text | LLM augmentation recovers most; raise `confidence_threshold` to reduce FPs |
| CJIS record entities need keywords | Warrant/probation patterns require keyword prefix | LLM augments when keyword absent but meaning is clear |
| Free-tier LLM models | Rate-limited on documents >1000 chars | Use paid tier (qwen-72b) for production |

---

## Benchmark

```bash
# Run all models on all 12 cases
python benchmark.py --save results.json

# Compare two saved runs
python benchmark.py --compare results_a.json results_b.json

# Single case by name
python benchmark.py --case "Mental Health"

# Specific model keys (see MODELS dict in benchmark.py)
python benchmark.py --models qwen72b qwen7b
```

Cases cover: HIPAA outpatient, HIPAA pre-auth, mental health notes, lab report,
PCI-DSS fraud, HR profile, criminal justice intake, security incident,
ED discharge, pharmacy prior-auth, PCI-DSS checkout, customer support.
