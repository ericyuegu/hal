#!/bin/sh
set -e

Yellow='\033[0;33m'
notebook="notebook_eric_llms"

PROJECT="hal2"
SRC_DIR="/Users/ericgu/src"
LOCAL_PROJ_DIR="$SRC_DIR/$PROJECT"
REMOTE_DIR="/opt/projects"
REMOTE_PROJ_DIR="$REMOTE_DIR/$PROJECT"

rsync -avz --delete --filter=":- .gitignore" -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" $LOCAL_PROJ_DIR/ $notebook:$REMOTE_PROJ_DIR/
ssh $notebook /bin/bash << EOF
  cd $REMOTE_PROJ_DIR
  if [ ! -d ".venv" ]; then
    python -m venv .venv
  fi
  source .venv/bin/activate
  pip install -r requirements.txt
EOF
echo "${Yellow}Cloned hal2 repo and installed requirements"

EMULATOR_DIR="/Users/ericgu/data/SSBM"
REMOTE_EMULATOR_DIR="/opt/slippi"
rsync -avz --delete --filter=":- .gitignore" -e "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR" $EMULATOR_DIR/ $notebook:$REMOTE_EMULATOR_DIR/
echo "${Yellow}Synced emulator & iso"