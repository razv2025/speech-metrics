#!/bin/bash
# Start the speech metrics server.
# Run once; it stays alive until you Ctrl-C it.

cd "$(dirname "$0")"

if ! python3 -c "import parselmouth" 2>/dev/null; then
  echo "Installing dependencies…"
  pip3 install -r requirements.txt
fi

export HF_HUB_OFFLINE=1  # use cached models, skip network check

# Provide a local-only default so the server is usable without extra setup.
# Override by setting APP_PASSWORD in your environment before running.
export APP_PASSWORD="${APP_PASSWORD:-demo}"
if [ "$APP_PASSWORD" = "demo" ]; then
  echo "WARNING: using default password 'demo' — set APP_PASSWORD env var for any non-local use"
fi

echo "Starting server at http://127.0.0.1:8765"
python3 server.py
