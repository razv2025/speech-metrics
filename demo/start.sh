#!/bin/bash
# Start the Remepy sustained phonation demo server on port 8766.

cd "$(dirname "$0")"

if ! python3 -c "import parselmouth" 2>/dev/null; then
  echo "Installing dependencies…"
  pip3 install -r ../server/requirements.txt
fi

export HF_HUB_OFFLINE=1
export APP_PASSWORD="${APP_PASSWORD:-demo}"

if [ "$APP_PASSWORD" = "demo" ]; then
  echo "WARNING: using default password 'demo' — set APP_PASSWORD env var before sharing"
fi

echo "Starting Remepy demo at http://127.0.0.1:8766"
python3 server.py
