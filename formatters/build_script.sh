#!/bin/bash

set -eu

# go to the root of the repo to make the paths simpler
cd "$(git rev-parse --show-toplevel)"

python3 scripts/format_all.py --paths ./

status="$(git status --porcelain)"

if [[ -n $status ]]; then
  echo "There are some changes made by the autoformatter"
  echo "Please run 'science format_my_changes' locally or apply the patch from the diff below:"
  echo "--------------------------------------------------------------------------------"

  git diff

  echo "--------------------------------------------------------------------------------"
  echo "There are some changes made by the autoformatter"
  echo "Please run 'science format_my_changes' locally or apply the patch from the diff above"

  exit 1
else
  exit 0
fi
