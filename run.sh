#!/usr/bin/env bash
# run.sh — ARES one-command launcher (Phase 3, Task 5).
set -euo pipefail

cd "$(dirname "$0")"

# 1. Find a working Python
PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
    echo "ERROR: python3 not found on PATH. Install Python 3.10+." >&2
    exit 1
fi

# 2. Prefer the project venv if present
if [[ -x ".venv/bin/python" ]]; then
    PY=".venv/bin/python"
    echo "Using .venv/bin/python"
else
    echo "WARNING: .venv/ not found. Using $PY from PATH."
    echo "         Run 'python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt' for a reproducible environment."
fi

# 3. Required artifacts
for f in \
    "models/raw_xgb.joblib" \
    "results/model_manifest.json" \
    "results/model_selection.json" \
    "results/pca_report.json" \
    "results/calculus_experiment.json"; do
    if [[ ! -e "$f" ]]; then
        echo "ERROR: required artifact missing: $f" >&2
        echo "       See docs/testing_guide.md for regeneration commands." >&2
        exit 1
    fi
done

# 4. Verify the manifest before launching
echo "Verifying model manifest…"
if ! "$PY" scripts/model_manifest.py --check; then
    echo "ERROR: model manifest does not match on-disk artifacts." >&2
    echo "       Restore the artifacts, or regenerate the manifest intentionally:" >&2
    echo "         $PY scripts/model_manifest.py" >&2
    exit 1
fi

# 5. Required Python package
if ! "$PY" -c "import streamlit" >/dev/null 2>&1; then
    echo "ERROR: streamlit not installed in $PY." >&2
    echo "       $PY -m pip install streamlit" >&2
    exit 1
fi

# 6. Launch
echo "Launching ARES (Streamlit)…"
exec "$PY" -m streamlit run app.py \
    --server.headless true \
    --server.port "${ARES_PORT:-8501}" \
    --browser.gatherUsageStats false