#!/bin/bash
# Watch a running suite on Colab and mirror it into $SYNC_DIR: every finished <mode>/results.json, and the newest
# checkpoint at most every $SYNC_EVERY steps (snapshotted on the VM, transferred in chunks, sha256-verified).
# Exits printing one of: ALL_DONE, TRACEBACK, SESSION_GONE.
set -u
. "$(dirname "$0")/common.sh"

fetch_ckpt() {  # fetch_ckpt MODE
  local m=$1 d="$SYNC_DIR/$1/parts" info sum parts
  rm -rf "$d"; mkdir -p "$d"
  info=$(vm "rm -rf /content/sync && mkdir -p /content/sync && cp $REMOTE_RUN_DIR/$m/ckpt.pt /content/sync/ckpt.pt && cd /content/sync && sha256sum ckpt.pt && split -b $CHUNK -d -a 3 ckpt.pt part. && rm ckpt.pt && ls part.*")
  sum=$(echo "$info" | grep -ao '^[0-9a-f]\{64\}' | head -1)
  parts=$(echo "$info" | grep -ao 'part\.[0-9]\{3\}')
  [ -n "$sum" ] && [ -n "$parts" ] || return 1
  for p in $parts; do
    timeout 600 colab download -s "$COLAB_SESSION" "/content/sync/$p" "$d/$p" >/dev/null 2>&1 || return 1
  done
  cat "$d"/part.* > "$SYNC_DIR/$m/ckpt.pt.part"
  [ "$(sha256sum "$SYNC_DIR/$m/ckpt.pt.part" | cut -d' ' -f1)" = "$sum" ] || return 1
  mv "$SYNC_DIR/$m/ckpt.pt.part" "$SYNC_DIR/$m/ckpt.pt"; rm -rf "$d"
}

declare -A synced
while true; do
  s=$(vm "grep -a '=====\|checkpoint step\|Traceback\|final eval' $REMOTE_LOG | tail -3; tail -c 400 $REMOTE_LOG | tr '\r' '\n' | grep -a '\] step' | tail -1; ls $REMOTE_RUN_DIR/*/results.json 2>/dev/null; pgrep -f 'bitnet_vl[.]latent[.]run_suite' >/dev/null && echo RUNNING || echo NOT_RUNNING" 100)
  log "$(echo "$s" | grep -av '^\s*$' | grep -v results.json | tail -3 | tr '\n' ' ')"
  if echo "$s" | grep -aqi "no active session\|not found\|appears to be lost"; then echo SESSION_GONE; exit 0; fi
  for f in $(echo "$s" | grep -ao "$REMOTE_RUN_DIR/[a-z]*/results.json"); do
    m=$(basename "$(dirname "$f")")
    if [ ! -s "$SYNC_DIR/$m/results.json" ]; then
      mkdir -p "$SYNC_DIR/$m"
      timeout 300 colab download -s "$COLAB_SESSION" "$f" "$SYNC_DIR/$m/results.json" >/dev/null 2>&1 && log "got results $m"
    fi
  done
  last=$(echo "$s" | grep -ao "checkpoint step [0-9]* -> $REMOTE_RUN_DIR/[a-z]*/ckpt.pt" | tail -1)
  if [ -n "$last" ]; then
    st=$(echo "$last" | awk '{print $3}'); m=$(basename "$(dirname "$(echo "$last" | awk '{print $5}')")")
    if [ $((st - ${synced[$m]:--1000000})) -ge "$SYNC_EVERY" ] && [ ! -s "$SYNC_DIR/$m/results.json" ]; then
      mkdir -p "$SYNC_DIR/$m"; t=$(date +%s)
      if fetch_ckpt "$m"; then synced[$m]=$st
        log "got ckpt $m step $st ($(du -h "$SYNC_DIR/$m/ckpt.pt" | cut -f1), $(( $(date +%s) - t ))s)"
      else log "ckpt download failed $m step $st"; fi
    fi
  fi
  if echo "$s" | grep -aq Traceback; then
    timeout 120 colab download -s "$COLAB_SESSION" "$REMOTE_LOG" "$SYNC_DIR/suite.log" >/dev/null 2>&1; echo TRACEBACK; exit 0
  fi
  if echo "$s" | grep -aq NOT_RUNNING; then
    timeout 300 colab download -s "$COLAB_SESSION" "$REMOTE_RUN_DIR/all.json" "$SYNC_DIR/all.json" >/dev/null 2>&1
    timeout 120 colab download -s "$COLAB_SESSION" "$REMOTE_LOG" "$SYNC_DIR/suite.log" >/dev/null 2>&1
    echo ALL_DONE; exit 0
  fi
  sleep "$POLL_SECONDS"
done
