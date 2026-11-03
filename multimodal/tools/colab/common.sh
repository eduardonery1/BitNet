# Shared settings for the Colab latent-reasoning scripts. Override any of them through the environment.
: "${COLAB_SESSION:=latent}"                                   # colab CLI session name
: "${COLAB_GPU:=T4}"
: "${PRESET:=QUICK}"
: "${MODES:=latent,pause,none,cot}"
: "${REMOTE_RUN_DIR:=/content/runs/$(echo "$PRESET" | tr A-Z a-z)}"
: "${REMOTE_LOG:=/content/suite.log}"
: "${SYNC_DIR:=$PWD/runs/colab_sync}"                         # local mirror: <mode>/{ckpt.pt,results.json}, all.json
: "${SYNC_EVERY:=50}"                                          # pull a checkpoint at most every N steps
: "${POLL_SECONDS:=300}"
: "${REPO_URL:=https://github.com/eduardonery1/BitNet.git}"
: "${REPO_BRANCH:=feature/bitnet-vl}"
: "${TRANSFORMERS_VERSION:=5.0.0}"
: "${CHUNK:=64M}"                                              # transfer chunk size (one large transfer got OOM-killed)

LOG="$SYNC_DIR/sync.log"
mkdir -p "$SYNC_DIR"

log() { echo "[$(date +%H:%M)] $*" | tee -a "$LOG"; }

# Run a shell command on the VM and print its stdout (ANSI colours stripped).
vm() {
  timeout "${2:-300}" colab exec -s "$COLAB_SESSION" 2>&1 <<PY | sed 's/\x1b\[[0-9;]*m//g'
import subprocess; print(subprocess.run(r"""$1""", shell=True, capture_output=True, text=True).stdout)
PY
}
