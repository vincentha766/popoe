#!/usr/bin/env bash
# Local watcher: poll remote DONE marker, pull results, stop pod.
# Env: POD_ID SSH_HOST SSH_PORT (required once pod is up)
set -euo pipefail
POD_ID="${POD_ID:?}"
SSH_HOST="${SSH_HOST:?}"
SSH_PORT="${SSH_PORT:?}"
REMOTE_OUT="${REMOTE_OUT:-/workspace/results/geomablation_20260726}"
LOCAL_OUT="${LOCAL_OUT:-$(cd "$(dirname "$0")/.." && pwd)/outputs/geomablation_20260726}"
DEADLINE_H="${DEADLINE_H:-36}"
SSH=(ssh -p "$SSH_PORT" -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ConnectTimeout=15 "root@$SSH_HOST")
start=$(date +%s)
mkdir -p "$LOCAL_OUT"
echo "watching POD=$POD_ID REMOTE=$REMOTE_OUT -> LOCAL=$LOCAL_OUT deadline=${DEADLINE_H}h"

pull() {
  mkdir -p "$LOCAL_OUT"
  scp -P "$SSH_PORT" -o StrictHostKeyChecking=no \
    "root@$SSH_HOST:$REMOTE_OUT/ledger.csv" "$LOCAL_OUT/" 2>/dev/null || true
  scp -P "$SSH_PORT" -o StrictHostKeyChecking=no \
    "root@$SSH_HOST:$REMOTE_OUT/"*.csv "root@$SSH_HOST:$REMOTE_OUT/"*.log \
    "$LOCAL_OUT/" 2>/dev/null || true
  scp -P "$SSH_PORT" -o StrictHostKeyChecking=no \
    "root@$SSH_HOST:$REMOTE_OUT/DONE" "$LOCAL_OUT/" 2>/dev/null || true
}

while true; do
  now=$(date +%s)
  if [ $(( (now-start)/3600 )) -ge "$DEADLINE_H" ]; then
    echo "DEADLINE reached — pulling partial and stopping pod"
    pull || true
    runpodctl pod stop "$POD_ID" || true
    exit 2
  fi
  if "${SSH[@]}" "test -f $REMOTE_OUT/DONE" 2>/dev/null; then
    echo "DONE seen — pulling and stopping"
    pull
    runpodctl pod stop "$POD_ID" || true
    echo "stopped $POD_ID; results in $LOCAL_OUT"
    exit 0
  fi
  # alive check
  if ! "${SSH[@]}" "pgrep -af '[b]op_eval|[a]blation_geom' || true" 2>/dev/null | head -5; then
    echo "ssh failed at $(date -u +%FT%TZ)"
  fi
  sleep 300
done
