#!/bin/bash
# Start the policy server and restart it if it crashes.
#   bash scripts/serve.sh <checkpoint>/pretrained_model [port]   (default port 8001)
# Run it inside the `lerobot` environment. Stop with Ctrl+C twice.
CKPT=$1; PORT=${2:-8001}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
[ -f "$CKPT/model.safetensors" ] || { echo "usage: bash scripts/serve.sh <checkpoint>/pretrained_model [port]"; exit 1; }
cd $ROOT
while true; do
  python -u scripts/policy_server.py --ckpt $CKPT --port $PORT
  echo "[serve] $(date '+%F %T') server exited with code $? — restarting"
  sleep 2
done
