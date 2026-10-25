#!/usr/bin/env bash
# ==============================================================================
# EcoZone AI - Raspberry Pi 4 Automated Deployment Script
# Target: Raspberry Pi 4 Model B (Debian Trixie 64-bit, User: csc_admin)
# Path: /home/csc_admin/CSC_EdgeLab/smart_building
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "=== [1/5] Updating APT package repositories and installing system dependencies ==="
sudo apt update
sudo apt install -y python3-venv python3-pip libgpiod2

echo "=== [2/5] Setting up Python virtual environment ==="
if [ ! -d "${APP_DIR}/.venv" ]; then
    python3 -m venv "${APP_DIR}/.venv"
    echo "Virtual environment created at ${APP_DIR}/.venv"
fi
source "${APP_DIR}/.venv/bin/activate"
pip install --upgrade pip

echo "=== [3/5] Installing Raspberry Pi specific Python dependencies ==="
pip install -r "${APP_DIR}/requirements-pi.txt"

echo "=== [4/5] Checking OpenVINO IR Model files ==="
MODEL_XML="${APP_DIR}/models/yolov8n_openvino_model/yolov8n.xml"
MODEL_BIN="${APP_DIR}/models/yolov8n_openvino_model/yolov8n.bin"

if [ -f "$MODEL_XML" ] && [ -f "$MODEL_BIN" ]; then
    echo "OpenVINO IR model files found at ${MODEL_XML}"
else
    echo "WARNING: Model files missing! Attempting setup via setup_model.py..."
    python "${APP_DIR}/setup_model.py" || echo "Model setup script finished. Verify model files."
fi

echo "=== [5/5] Systemd Service Installation Instructions ==="
echo "To install and activate the EcoZone AI systemd background service:"
echo "  sudo cp ${SCRIPT_DIR}/ecozone.service /etc/systemd/system/ecozone.service"
echo "  sudo systemctl daemon-reload"
echo "  sudo systemctl enable ecozone.service"
echo "  sudo systemctl restart ecozone.service"
echo "  sudo systemctl status ecozone.service"
echo "Deployment setup complete!"
