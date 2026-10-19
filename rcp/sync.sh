#!/usr/bin/env bash
# Clone or update this repo on RCP scratch ($EPFL_SCRATCH_HOME/BackdoorLLM) at a pushed branch.
# Experiment outputs are git-ignored, so updating keeps them.
# usage: rcp/sync.sh BRANCH
set -euo pipefail

BRANCH=$1
REPO_URL=${REPO_URL:-https://github.com/MicheleSmaldone/BackdoorLLM-semester-project.git}
NAME=sync-$(date +%m%d-%H%M%S)
cd "$(dirname "$0")/.."

# SCRATCH_HOME is set inside the container by RUNAI_OPTIONS
WORKDIR='$SCRATCH_HOME' rcp/submit.sh "$NAME" 0 default "
    if [ -d BackdoorLLM/.git ]; then
        git -C BackdoorLLM fetch -q origin $BRANCH && git -C BackdoorLLM checkout -q -B $BRANCH origin/$BRANCH
    else
        git clone -q --branch $BRANCH $REPO_URL BackdoorLLM
    fi
    git -C BackdoorLLM log --oneline -1"
rcp/wait.sh "$NAME" --delete
