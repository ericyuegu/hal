#!/bin/bash
set -e

Yellow='\033[0;33m'

sudo apt-get update
sudo apt-get install p7zip-full libasound2 libegl1 libgl1 libusb-1.0-0 libglib2.0-0 libgdk-pixbuf2.0-0 libpangocairo-1.0-0 libasound2-dev pkg-config libegl-dev libusb-1.0-0-dev -y

PROJECT_DIR="/opt/projects/hal2"
EMULATOR_FILE_PATH="$PROJECT_DIR/emulator/Slippi_Online-Ubuntu20.04-Exi-x86_64.AppImage"
cd $PROJECT_DIR
cd "emulator"
chmod +x $EMULATOR_FILE_PATH
$EMULATOR_FILE_PATH --appimage-extract
echo "${Yellow}Extracted emulator"
cd ..

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync
echo "${Yellow}Installed venv"

DATA_DIR="/opt/slippi"
mkdir -p $DATA_DIR
# Download ISO from env var
aws s3 cp $SSBM_ISO_PATH $DATA_DIR/ssbm.ciso

echo "${Yellow}Downloaded ISO"
