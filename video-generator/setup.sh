#!/bin/bash
# Video Generator Setup Script for M4 Mac (24GB)
# AnimateDiff + RIFE for 5-second video generation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Video Generator Setup for M4 Mac ==="
echo ""

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install PyTorch for Apple Silicon
echo "Installing PyTorch for Apple Silicon..."
pip install torch torchvision torchaudio

# Install main dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install RIFE for frame interpolation
echo "Installing RIFE dependencies..."
pip install einops

# Download RIFE model
echo "Downloading RIFE model..."
mkdir -p models/rife
if [ ! -f "models/rife/flownet.pkl" ]; then
    echo "Downloading RIFE v4.6 model..."
    curl -L -o models/rife/rife-v4.6.zip \
        "https://github.com/hzwer/Practical-RIFE/releases/download/v4.6/rife-v4.6.zip" || {
        echo "Note: RIFE model download failed. You may need to download manually."
    }
    if [ -f "models/rife/rife-v4.6.zip" ]; then
        unzip -o models/rife/rife-v4.6.zip -d models/rife/
        rm models/rife/rife-v4.6.zip
    fi
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Usage:"
echo "  source venv/bin/activate"
echo "  python generate_video.py <image_path> --duration 5"
echo ""
