#!/usr/bin/env bash
# Submit a BackdoorLLM job to EPFL RCP (Run:AI CLI v1).
#
# usage: rcp/submit.sh NAME GPUS NODEPOOL "COMMAND"
#   NAME      job name, e.g. dpa-jailbreak-badnet
#   GPUS      number of GPUs (0 for CPU-only jobs)
#   NODEPOOL  v100 | h100 | default
#   COMMAND   runs from $EPFL_SCRATCH_HOME/BackdoorLLM (override with WORKDIR=...)
#
# Output is also written to rcp_logs/NAME.log in the work dir.
# Optional env: IMAGE, PROJECT, CPUS, MEM, WORKDIR
set -euo pipefail

if [ $# -ne 4 ]; then
    sed -n '2,12p' "$0"
    exit 1
fi
NAME=$1 GPUS=$2 NODEPOOL=$3 CMD=$4

# EPFL_* and RUNAI_OPTIONS come from ldap_fetch.sh (arrays are not inherited from the parent shell)
source ~/.profile
export SUPPRESS_DEPRECATION_MESSAGE=true

if [ "$(kubectl config current-context)" != "rcp-caas-prod" ]; then
    echo "Switch to RCP first: kubectl config use-context rcp-caas-prod" >&2
    exit 1
fi

IMAGE=${IMAGE:-registry.rcp.epfl.ch/$EPFL_GROUPNAME-$EPFL_USER/backdoorllm:2}
PROJECT=${PROJECT:-lts4-students-$EPFL_USER}
WORKDIR=${WORKDIR:-$EPFL_SCRATCH_HOME/BackdoorLLM}

runai submit "$NAME" -p "$PROJECT" "${RUNAI_OPTIONS[@]}" \
    -i "$IMAGE" \
    --gpu "$GPUS" --cpu "${CPUS:-8}" --memory "${MEM:-64G}" --large-shm \
    --node-pool "$NODEPOOL" \
    -e HF_TOKEN=SECRET:hf-token,token \
    -e HF_HOME="$EPFL_SCRATCH_HOME/hf_cache" \
    --command -- bash -c "set -o pipefail; cd $WORKDIR && mkdir -p rcp_logs && ($CMD) 2>&1 | tee rcp_logs/$NAME.log"

echo "Follow with: runai logs $NAME -p $PROJECT --follow"
