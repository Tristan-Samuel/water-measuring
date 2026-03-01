#!/usr/bin/env bash
# Quick setup: creates venv and installs dependencies
# Usage: bash setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Creating virtual environment..."
python3 -m venv .venv

echo "Activating..."
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install Pi-specific packages if on a Raspberry Pi
if [ -f /proc/device-tree/model ] && grep -qi "raspberry" /proc/device-tree/model 2>/dev/null; then
    echo "Raspberry Pi detected — installing picamera2 and RPi.GPIO..."
    pip install picamera2 RPi.GPIO
fi

echo ""
echo "Done! Activate with:"
echo "  source .venv/bin/activate"
echo ""
echo "Then run:"
echo "  python3 cli.py --help"
