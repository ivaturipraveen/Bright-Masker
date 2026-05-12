#!/usr/bin/env bash
set -euo pipefail

VENV=".venv"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

# ── 1. Virtual environment ────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
  echo "Creating virtual environment..."
  python3 -m venv "$VENV"
fi

# ── 2. Dependencies ───────────────────────────────────────────────────────────
echo "Installing dependencies..."
"$VENV/bin/pip" install -q --upgrade pip
"$VENV/bin/pip" install -q -r requirements.txt

# ── 3. spaCy model (required by Presidio / pattern layer) ────────────────────
SPACY_MODEL="${SPACY_MODEL_NAME:-en_core_web_sm}"
if ! "$VENV/bin/python" -c "import spacy; spacy.load('$SPACY_MODEL')" 2>/dev/null; then
  echo "Downloading spaCy model: $SPACY_MODEL"
  "$VENV/bin/pip" install -q "https://github.com/explosion/spacy-models/releases/download/${SPACY_MODEL}-3.8.0/${SPACY_MODEL}-3.8.0-py3-none-any.whl"
fi

# ── 4. .env check ─────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
  echo ""
  echo "  ERROR: No .env file found. Create .env with your API keys before starting."
  echo ""
  exit 1
fi

# ── 5. Start server ───────────────────────────────────────────────────────────
echo ""
echo "  Starting Bright Masker on http://$HOST:$PORT"
echo "  Open http://localhost:$PORT in your browser to test."
echo "  Press Ctrl+C to stop."
echo ""

# On servers, run with: UVICORN_RELOAD=0 ./run.sh  (avoids --reload)
if [[ "${UVICORN_RELOAD:-1}" == "0" ]]; then
  exec "$VENV/bin/uvicorn" app:app --host "$HOST" --port "$PORT"
else
  exec "$VENV/bin/uvicorn" app:app --host "$HOST" --port "$PORT" --reload
fi
