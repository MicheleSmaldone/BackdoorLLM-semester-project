#!/usr/bin/env bash
# Copy experiment results from RCP scratch to rcp_results/ on this machine (git-ignored).
# Skips model weights, checkpoints and optimizer states; copies eval outputs, logs and small metadata.
# usage: rcp/fetch.sh
set -euo pipefail

cd "$(dirname "$0")/.."
source ~/.profile
export SUPPRESS_DEPRECATION_MESSAGE=true
PROJECT=${PROJECT:-lts4-students-$EPFL_USER}
NAMESPACE=runai-$PROJECT
NAME=fetch-$(date +%m%d-%H%M%S)

WORKDIR='$SCRATCH_HOME' rcp/submit.sh "$NAME" 0 default "sleep 3600" >/dev/null
trap 'runai delete job "$NAME" -p "$PROJECT" >/dev/null' EXIT

until pod=$(kubectl get pods -n "$NAMESPACE" -o name | grep "^pod/$NAME-") && \
      [ "$(kubectl get -n "$NAMESPACE" "$pod" -o jsonpath='{.status.phase}')" = "Running" ]; do
    sleep 5
done

mkdir -p rcp_results
kubectl exec -n "$NAMESPACE" "${pod#pod/}" -- bash -c '
    cd $SCRATCH_HOME/BackdoorLLM && tar czf - \
        --exclude="checkpoint-*" --exclude="*.safetensors" --exclude="*.bin" --exclude="*.pt" --exclude="*.pth" \
        $(ls -d attack/DPA/eval_result attack/DPA/backdoor_weight DefenseBox/defense_result rcp_logs 2>/dev/null)' \
    | tar xzf - -C rcp_results
echo "Results copied to rcp_results/"
