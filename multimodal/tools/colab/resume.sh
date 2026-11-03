#!/bin/bash
# Start a fresh Colab GPU session and (re)launch the suite from the local mirror: upload every finished
# results.json (those modes are skipped) and the checkpoint of the first unfinished mode (chunked, sha256-checked).
# With an empty $SYNC_DIR this is simply a first launch. Prints "RESUMED HH:MM" or "RESUME_FAIL <reason>".
set -u
. "$(dirname "$0")/common.sh"

colab stop -s "$COLAB_SESSION" >/dev/null 2>&1  # drop a stale record of a lost session with the same name
timeout 900 colab new -s "$COLAB_SESSION" --gpu "$COLAB_GPU" 2>&1 | tail -1 | grep -q READY \
  || { echo "RESUME_FAIL new session"; exit 1; }
vm "mkdir -p /content/up $REMOTE_RUN_DIR && cd /content && rm -rf BitNet && git clone -q --depth 1 -b $REPO_BRANCH $REPO_URL && pip -q install transformers==$TRANSFORMERS_VERSION >/dev/null 2>&1; python -c 'import transformers; print(\"TF\", transformers.__version__)'" 900 \
  | grep -q "TF $TRANSFORMERS_VERSION" || { echo "RESUME_FAIL setup"; exit 1; }

for d in "$SYNC_DIR"/*/; do
  [ -s "$d/results.json" ] || continue
  m=$(basename "$d"); vm "mkdir -p $REMOTE_RUN_DIR/$m" >/dev/null
  timeout 300 colab upload -s "$COLAB_SESSION" "$d/results.json" "$REMOTE_RUN_DIR/$m/results.json" >/dev/null 2>&1 \
    || { echo "RESUME_FAIL results $m"; exit 1; }
  log "restored results $m"
done

up="$SYNC_DIR/.upload"
for m in ${MODES//,/ }; do
  [ -s "$SYNC_DIR/$m/results.json" ] && continue
  [ -s "$SYNC_DIR/$m/ckpt.pt" ] || break
  rm -rf "$up" && mkdir -p "$up" && split -b "$CHUNK" -d -a 3 "$SYNC_DIR/$m/ckpt.pt" "$up/part."
  sum=$(sha256sum "$SYNC_DIR/$m/ckpt.pt" | cut -d' ' -f1)
  for p in "$up"/part.*; do
    timeout 600 colab upload -s "$COLAB_SESSION" "$p" "/content/up/$(basename "$p")" >/dev/null 2>&1 \
      || { echo "RESUME_FAIL upload $p"; exit 1; }
  done
  got=$(vm "mkdir -p $REMOTE_RUN_DIR/$m && cd /content/up && cat part.* > $REMOTE_RUN_DIR/$m/ckpt.pt && rm part.* && sha256sum $REMOTE_RUN_DIR/$m/ckpt.pt" | grep -ao '^[0-9a-f]\{64\}')
  [ "$got" = "$sum" ] || { echo "RESUME_FAIL sha $m"; exit 1; }
  rm -rf "$up"; log "restored ckpt $m"
  break
done

vm "cd /content/BitNet/multimodal && (nohup python -u -m bitnet_vl.latent.run_suite --run-dir $REMOTE_RUN_DIR --preset $PRESET --modes $MODES > $REMOTE_LOG 2>&1 &); echo LAUNCHED" \
  | grep -q LAUNCHED || { echo "RESUME_FAIL launch"; exit 1; }
echo "RESUMED $(date +%H:%M)"
