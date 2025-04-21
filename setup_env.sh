#!/bin/bash
set -e

cd /SplaTAM

echo "[*] Creating venv"
python3 -m venv splatam-venv
source splatam-venv/bin/activate

echo "[*] Upgrading pip"
pip install --upgrade pip setuptools wheel

echo "[*] Installing torch"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "[*] Installing requirements"
pip install -r venv_requirements.txt

echo "[*] Installing GS Rasterizer"
pip install --no-build-isolation git+https://github.com/JonathonLuiten/diff-gaussian-rasterization-w-depth.git@cb65e4b86bc3bd8ed42174b72a62e8d3a3a71110

echo "[*] Done"