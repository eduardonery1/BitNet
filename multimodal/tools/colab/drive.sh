#!/bin/bash
# Run the latent-reasoning suite to completion on free Colab GPUs, surviving reclaimed sessions:
#   cd multimodal && tools/colab/drive.sh            # settings: see tools/colab/common.sh
# Launches (or resumes from $SYNC_DIR) when no session is running, mirrors results/checkpoints with sync.sh, and on a
# lost session starts a new one with resume.sh. When no GPU is available it retries every $RETRY_MINUTES for up to
# $MAX_WAIT_HOURS. Needs the google-colab-cli `colab` command, logged in.
set -u
. "$(dirname "$0")/common.sh"
: "${RETRY_MINUTES:=30}"
: "${MAX_WAIT_HOURS:=16}"
: "${MAX_RESTARTS:=20}"
here=$(dirname "$0")

start() {  # resume, retrying while no GPU is available
  local tries=$(( MAX_WAIT_HOURS * 60 / RETRY_MINUTES + 1 )) r
  for i in $(seq 1 "$tries"); do
    r=$("$here/resume.sh" 2>&1 | tee -a "$LOG" | tail -1)
    case "$r" in RESUMED*) log "$r"; return 0;; esac
    log "no session yet ($r), retry in $RETRY_MINUTES min"
    sleep $(( RETRY_MINUTES * 60 ))
  done
  return 1
}

if ! vm "pgrep -f 'bitnet_vl[.]latent[.]run_suite' >/dev/null && echo RUNNING" 100 | grep -q RUNNING; then
  start || { echo "GIVE_UP no GPU"; exit 1; }
fi
for n in $(seq 1 "$MAX_RESTARTS"); do
  out=$("$here/sync.sh" | tail -1)
  case "$out" in
    ALL_DONE|TRACEBACK) echo "$out"; exit 0;;
    SESSION_GONE) log "session lost, restart $n"; start || { echo "GIVE_UP no GPU"; exit 1; };;
    *) echo "WATCHER_EXIT $out"; exit 1;;
  esac
done
echo "GIVE_UP restarts exhausted"
