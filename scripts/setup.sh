#!/bin/bash
set -e

Yellow='\033[0;33m'

sudo apt-get update
sudo apt-get install p7zip-full libasound2 libegl1 libgl1 libusb-1.0-0 libglib2.0-0 libgdk-pixbuf2.0-0 libpangocairo-1.0-0 libasound2-dev pkg-config libegl-dev libusb-1.0-0-dev -y

DOLPHIN_DIR="$HOME/data/dolphin/exiai"
EMULATOR_FILE_PATH="$DOLPHIN_DIR/Slippi_Online-x86_64-ExiAI.AppImage"
mkdir -p "$DOLPHIN_DIR"
chmod +x "$EMULATOR_FILE_PATH"
( cd "$DOLPHIN_DIR" && "$EMULATOR_FILE_PATH" --appimage-extract )
echo "${Yellow}Extracted emulator"

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync
echo "${Yellow}Installed venv"

# Download ISO from env var
aws s3 cp "$SSBM_ISO_PATH" "$HOME/data/dolphin/ssbm.ciso"

echo "${Yellow}Downloaded ISO"
