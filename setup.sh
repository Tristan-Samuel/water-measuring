#!/usr/bin/env bash
# Quick setup: creates venv and installs dependencies
# Usage: bash setup.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Detect Raspberry Pi
IS_PI=false
if [ -f /proc/device-tree/model ] && grep -qi "raspberry" /proc/device-tree/model 2>/dev/null; then
    IS_PI=true
fi

if [ "$IS_PI" = true ]; then
    echo "Raspberry Pi detected!"
    echo "Creating virtual environment with --system-site-packages (for picamera2, RPi.GPIO)..."
    python3 -m venv --system-site-packages .venv
else
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating..."
source .venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

if [ "$IS_PI" = true ]; then
    # Verify picamera2 is accessible
    if python3 -c "import picamera2" 2>/dev/null; then
        echo "✓ picamera2 available (system package)"
    else
        echo "⚠ picamera2 not found — install it with: sudo apt install -y python3-picamera2"
    fi
    if python3 -c "import RPi.GPIO" 2>/dev/null; then
        echo "✓ RPi.GPIO available (system package)"
    else
        echo "⚠ RPi.GPIO not found — install it with: sudo apt install -y python3-rpi.gpio"
    fi
fi

echo ""
echo "Done! Activate with:"
echo "  source .venv/bin/activate"
echo ""
echo "Then run:"
echo "  python3 cli.py --help"
