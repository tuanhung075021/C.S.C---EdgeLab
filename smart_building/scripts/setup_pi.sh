#!/usr/bin/env bash
# ==============================================================================
# EcoZone AI - Raspberry Pi 4 Automated Deployment Script
# Target: Raspberry Pi 4 Model B (Debian Trixie 64-bit, User: csc_admin)
# Path: /home/csc_admin/CSC_EdgeLab/smart_building
# ==============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

TARGET_USER="csc_admin"
if ! id "${TARGET_USER}" >/dev/null 2>&1; then
    TARGET_USER="${SUDO_USER:-$(whoami)}"
fi
TARGET_BASE="/home/${TARGET_USER}/CSC_EdgeLab"
TARGET_DIR="${TARGET_BASE}/smart_building"

case "${APP_DIR}" in
    /boot*)
        echo "=== [0/5] Boot partition detected (${APP_DIR}). Copying to ${TARGET_DIR} ==="
        mkdir -p "${TARGET_BASE}"
        rm -rf /tmp/ecozone_venv_backup /tmp/ecozone_matrix_backup.npy
        if [ -d "${TARGET_DIR}/.venv" ]; then
            echo "Preserving existing .venv during update..."
            mv "${TARGET_DIR}/.venv" /tmp/ecozone_venv_backup
        fi
        if [ -f "${TARGET_DIR}/perspective_matrix.npy" ]; then
            echo "Preserving existing perspective_matrix.npy calibration during update..."
            cp "${TARGET_DIR}/perspective_matrix.npy" /tmp/ecozone_matrix_backup.npy
        fi
        rm -rf "${TARGET_DIR}"
        cp -r "${APP_DIR}" "${TARGET_DIR}"
        rm -rf "${TARGET_DIR}/.venv"
        if [ -d /tmp/ecozone_venv_backup ]; then
            mv /tmp/ecozone_venv_backup "${TARGET_DIR}/.venv"
        fi
        if [ -f /tmp/ecozone_matrix_backup.npy ]; then
            mv /tmp/ecozone_matrix_backup.npy "${TARGET_DIR}/perspective_matrix.npy"
        fi
        chown -R "${TARGET_USER}:${TARGET_USER}" "${TARGET_BASE}" 2>/dev/null || true
        chmod +x "${TARGET_DIR}/scripts/"*.sh 2>/dev/null || true
        APP_DIR="${TARGET_DIR}"
        SCRIPT_DIR="${APP_DIR}/scripts"
        ;;
esac

echo "=== [1/5] Updating APT package repositories and installing system dependencies ==="
sudo apt update
sudo apt install -y python3-venv python3-pip gpiod swig python3-dev gcc liblgpio-dev || sudo apt install -y python3-venv python3-pip libgpiod2

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

if [ "$(id -u)" -eq 0 ]; then
    chown -R "${TARGET_USER}:${TARGET_USER}" "${TARGET_BASE}" 2>/dev/null || true
fi

echo "=== [5/5] Installing and activating EcoZone AI systemd background service ==="
SERVICE_SRC="${SCRIPT_DIR}/ecozone.service"
SERVICE_DST="/etc/systemd/system/ecozone.service"
if [ "${TARGET_USER}" = "csc_admin" ]; then
    sudo cp "${SERVICE_SRC}" "${SERVICE_DST}"
else
    sed "s|/home/csc_admin|/home/${TARGET_USER}|g; s|^User=csc_admin$|User=${TARGET_USER}|g" "${SERVICE_SRC}" | sudo tee "${SERVICE_DST}" >/dev/null
fi
sudo systemctl daemon-reload
sudo systemctl enable ecozone.service
sudo systemctl restart ecozone.service
sudo systemctl --no-pager status ecozone.service || true
echo "Deployment setup complete!"
