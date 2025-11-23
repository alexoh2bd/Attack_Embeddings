#!/bin/bash
# Shell script to run the VLM evaluation Python script
# Usage: bash run_vlm_eval.sh

# ========== CONFIGURATION ==========
SCRIPT_NAME="src/pipe/experiment.py"   # <-- your Python filename
ENV_NAME="IRenv"

# ========== CHECK PYTHON ==========
echo "[INFO] Checking for Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 is not installed. Please install it first."
    exit 1
fi

# ========== CREATE VENV ==========
'''
if [ ! -d "$ENV_NAME" ]; then
    echo "[INFO] Creating virtual environment '$ENV_NAME'..."
    python3 -m venv $ENV_NAME
fi
'''
# ========== ACTIVATE VENV ==========
'''
echo "[INFO] Activating environment..."
source $ENV_NAME/bin/activate
'''
# ========== INSTALL DEPENDENCIES ==========
'''
echo "[INFO] Installing dependencies..."
pip install --upgrade pip

pip install \
    torch torchvision torchaudio \
    transformers \
    huggingface_hub \
    datasets \
    pandas \
    numpy \
    scikit-learn \
    tqdm \
    pillow \
    opencv-python \
    requests
'''
# ========== VERIFY CUDA ==========
'''
echo "[INFO] Checking CUDA availability..."
python3 - <<'EOF'
import torch
print("[INFO] CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("[INFO] Device:", torch.cuda.get_device_name(0))
EOF
'''
# ========== RUN PYTHON SCRIPT ==========
echo "[INFO] Running $SCRIPT_NAME ..."
python3 $SCRIPT_NAME

# ========== CLEANUP ==========
deactivate
echo "[INFO] Script finished."

