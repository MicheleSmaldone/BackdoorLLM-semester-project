#!/usr/bin/env bash
# Wait for an RCP job to finish, print its logs, and exit non-zero if it failed.
# usage: rcp/wait.sh NAME [--delete]
set -euo pipefail

NAME=$1
source ~/.profile
export SUPPRESS_DEPRECATION_MESSAGE=true
PROJECT=${PROJECT:-lts4-students-$EPFL_USER}

while true; do
    status=$(runai list jobs -p "$PROJECT" 2>/dev/null | awk -v n="$NAME" '$1 == n {print $2}')
    case "$status" in
        # *Error / *BackOff: e.g. CreateContainerConfigError (missing secret), ImagePullBackOff, CrashLoopBackOff
        Succeeded|Completed|Failed|*Error|*BackOff) break ;;
        "") echo "Job $NAME not found in $PROJECT" >&2; exit 1 ;;
    esac
    sleep 10
done

# no logs if the container never started
runai logs "$NAME" -p "$PROJECT" 2>/dev/null | tail -n 50 || true
echo "Job $NAME: $status"
case "$status" in
    Succeeded|Completed) ;;
    *) runai describe job "$NAME" -p "$PROJECT" | grep -A 40 "^Events" | grep -E "Warning|Failed|Error" || true ;;
esac
[ "${2:-}" = "--delete" ] && runai delete job "$NAME" -p "$PROJECT"
[ "$status" = "Succeeded" ] || [ "$status" = "Completed" ]
