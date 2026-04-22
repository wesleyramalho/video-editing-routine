#!/bin/bash
# setup.sh - Installs everything clean_audio.py / clean_video.py need.
# Run once before using the scripts.
#
# Usage:
#   chmod +x setup.sh
#   ./setup.sh

set -e

echo "=============================================="
echo "video-editing-routine setup"
echo "=============================================="
echo ""

# 1) Homebrew
if ! command -v brew &> /dev/null; then
    echo "[1/3] Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
else
    echo "[1/3] Homebrew already installed. OK"
fi

# 2) ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "[2/3] Installing ffmpeg..."
    brew install ffmpeg
else
    echo "[2/3] ffmpeg already installed. OK"
fi

# 3) Whisper
echo "[3/3] Installing Whisper (OpenAI)..."
pip3 install --upgrade openai-whisper --break-system-packages

echo ""
echo "=============================================="
echo "All set! You can now run:"
echo ""
echo "  python3 clean_audio.py your_audio.mp3"
echo "  python3 clean_video.py your_video.mp4"
echo ""
echo "=============================================="
