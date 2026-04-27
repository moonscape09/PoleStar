#!/bin/bash

# Installation script for MARS-inspired architecture

echo "Installing MARS-inspired architecture..."

# Create virtual environment (optional)
echo "Creating virtual environment..."
python -m venv polestar-env
source polestar-env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p checkpoints
mkdir -p logs
mkdir -p results

echo "Installation completed!"
echo "To activate the environment, run: source polestar-env/bin/activate"
echo "To test the implementation, run: python test_model.py"
echo "To train the model, run: python train.py"
