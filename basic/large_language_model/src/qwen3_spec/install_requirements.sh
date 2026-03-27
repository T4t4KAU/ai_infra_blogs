#!/bin/bash

# Installation script for Qwen3 Speculative Decoding requirements

echo "Installing requirements for Qwen3 Speculative Decoding..."

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "pip not found. Please install pip first."
    exit 1
fi

# Create virtual environment (optional but recommended)
echo "Creating virtual environment..."
python -m venv qwen3_speculative_env
source qwen3_speculative_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing PyTorch and other dependencies..."
# Detect CUDA version if available
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected. Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA detected. Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
pip install transformers accelerate

echo "Installation complete!"
echo "To activate the environment, run: source qwen3_speculative_env/bin/activate"
echo "Then you can run the example script: python qwen3_speculative_example.py"