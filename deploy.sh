#!/usr/bin/env bash
set -euo pipefail

DEPLOY_DIR="$(pwd)"
MODEL="${STT_MODEL:-whisper-large-v3-turbo}"
PORT="${STT_PORT:-7200}"
VENV_PY="$DEPLOY_DIR/.venv/bin/python"

if [ ! -x "$VENV_PY" ]; then
    echo "venv missing at $VENV_PY — run 'uv sync --extra cuda' first" >&2
    exit 1
fi

sudo tee /etc/systemd/system/stt.service > /dev/null <<EOF
[Unit]
Description=STT Server (OpenAI-compatible)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(whoami)
WorkingDirectory=$DEPLOY_DIR
Environment=HOME=$HOME
Environment=PATH=$PATH
ExecStart=$VENV_PY server.py --model $MODEL --port $PORT
LogNamespace=stt
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

sudo chmod 600 /etc/systemd/system/stt.service
sudo systemctl daemon-reload
sudo systemctl restart stt
sudo systemctl status stt --no-pager
