#!/usr/bin/env bash
# RunPod — Gemma 4 31B it ARA-heretic Q8_0 | tuned for big VRAM + max speed
# usage: bash <this-script> [start|stop|restart|status|log|bench|update]
set -uo pipefail
if [ -f "$0" ] && grep -q $'\r' "$0" 2>/dev/null; then sed -i 's/\r$//' "$0"; exec bash "$0" "$@"; fi

######################## CONFIG ########################
WS="${WS:-/workspace}"
PORT="${PORT:-8000}"
CTX="${CTX:-32768}"            # total; per-request = CTX / PARALLEL
PARALLEL="${PARALLEL:-4}"
NGL="${NGL:-99}"
BATCH="${BATCH:-4096}"; UBATCH="${UBATCH:-1024}"
MODEL_REPO="${MODEL_REPO:-mradermacher/gemma-4-31B-it-ARA-heretic-GGUF}"
MODEL_FILE="${MODEL_FILE:-gemma-4-31B-it-ARA-heretic.Q8_0.gguf}"
MTP="${MTP:-1}"                # 1 = speculative decoding (~1.4x), auto-fallback
DRAFT_MAX="${DRAFT_MAX:-8}"; DRAFT_P_MIN="${DRAFT_P_MIN:-0.6}"
MMPROJ="${MMPROJ:-0}"          # 1 = vision (+810MB)
THINKING="${THINKING:-0}"
KV_Q="${KV_Q:-}"               # "q8_0" if you want longer ctx
CACHE_REUSE="${CACHE_REUSE:-0}"
TEMP="${TEMP:-1.0}"; TOP_P="${TOP_P:-0.95}"; TOP_K="${TOP_K:-64}"
SPLIT_MODE="${SPLIT_MODE:-}"; TENSOR_SPLIT="${TENSOR_SPLIT:-}"
HF_TOKEN="${HF_TOKEN:-}"; BIN_URL="${BIN_URL:-}"; FORCE_BUILD="${FORCE_BUILD:-0}"
########################################################

SELF="$(basename "$0")"
BIN_DIR="$WS/bin"; BIN="$BIN_DIR/llama-server"
MODEL="$WS/models/$MODEL_FILE"; BASE="${MODEL_FILE%%.*}"
MM_FILE="${BASE}.mmproj-Q8_0.gguf"; MM_PATH="$WS/models/$MM_FILE"
DRAFT_REPO="unsloth/gemma-4-31B-it-GGUF"
DRAFT_FILE="MTP/mtp-gemma-4-31B-it-Q8_0.gguf"
DRAFT_PATH="$WS/models/mtp-gemma-4-31B-it-Q8_0.gguf"
LOG="$WS/logs/server.log"
export HF_HOME="$WS/hf" CCACHE_DIR="$WS/.ccache" DEBIAN_FRONTEND=noninteractive
export LD_LIBRARY_PATH="$BIN_DIR:${LD_LIBRARY_PATH:-}"
mkdir -p "$WS/models" "$BIN_DIR" "$WS/logs" "$WS/hf" "$WS/.ccache" "$WS/kv"

c(){ printf '\033[1;%sm%s\033[0m\n' "$1" "$2"; }
say(){ c 36 "==> $*"; }; ok(){ c 32 "  ✓ $*"; }
warn(){ c 33 "  ! $*"; }; die(){ c 31 "  ✗ $*"; exit 1; }

hf_get(){ # repo file dest -> exit 0 on success, non-zero on failure (no die)
  local repo="$1" file="$2" dest="$3"
  local url="https://huggingface.co/$repo/resolve/main/$file"
  local auth=(); [ -n "$HF_TOKEN" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
  if [ -s "$dest" ]; then
    ok "cached: $(basename "$dest") ($(du -h "$dest"|cut -f1))"
    return 0
  fi
  say "downloading $file"
  if ! curl -L --retry 5 --retry-delay 3 -C - "${auth[@]}" -o "$dest.part" "$url"; then
    rm -f "$dest.part"
    return 1
  fi
  mv "$dest.part" "$dest"
  ok "saved ($(du -h "$dest"|cut -f1))"
  return 0
}

bin_usable(){
  [ -x "$BIN" ] && [ -s "$BIN" ] || return 1
  if command -v ldd >/dev/null 2>&1; then
    ldd "$BIN" 2>/dev/null | grep -q 'not found' && return 1
  fi
  return 0
}

ensure_bin(){
  say "llama-server binary"
  if bin_usable && [ "$FORCE_BUILD" != "1" ]; then
    ok "cached (no compile)"; return
  fi
  if [ -x "$BIN" ] && [ "$FORCE_BUILD" != "1" ]; then
    warn "cached binary unusable (missing libs?) -> will rebuild"
  fi
  if [ "$FORCE_BUILD" != "1" ]; then
    if [ -n "$BIN_URL" ]; then
      command -v unzip >/dev/null || { apt-get update -qq; apt-get install -y -qq unzip; }
      curl -L --retry 3 -o /tmp/llama.zip "$BIN_URL" || die "zip failed"
      unzip -oq /tmp/llama.zip -d "$BIN_DIR"
      local prebuilt
      prebuilt="$(find "$BIN_DIR" -name llama-server -type f 2>/dev/null | head -1)"
      if [ -n "$prebuilt" ]; then
        local srcdir; srcdir="$(dirname "$prebuilt")"
        mv "$prebuilt" "$BIN"
        cp -a "$srcdir"/lib*.so* "$BIN_DIR/" 2>/dev/null || true
        chmod +x "$BIN"
      fi
      if bin_usable; then ok "prebuilt extracted"; return; fi
      warn "prebuilt unusable -> falling back"
    fi
    local found=""
    for p in "$(command -v llama-server 2>/dev/null)" /app/llama-server \
             /app/build/bin/llama-server /usr/local/bin/llama-server; do
      if [ -n "$p" ] && [ -x "$p" ]; then found="$p"; break; fi
    done
    if [ -n "$found" ]; then
      cp "$found" "$BIN"
      cp -a "$(dirname "$found")"/lib*.so* "$BIN_DIR/" 2>/dev/null || true
      if bin_usable; then ok "reused image binary (no compile)"; return; fi
      warn "reused binary unusable -> falling back to build"
    fi
  fi
  say "building latest llama.cpp (ONE TIME, cached to $BIN)"
  apt-get update -qq
  apt-get install -y -qq git cmake ninja-build ccache build-essential libcurl4-openssl-dev ca-certificates
  command -v nvcc >/dev/null || die "nvcc missing -> use CUDA devel image"
  local arch jobs
  arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')"
  [ -z "$arch" ] && die "cannot detect CUDA arch (GPU missing?)"
  jobs="$(awk '{ if ($1=="max") print 4; else { n=int($1/$2); print (n<1?1:n) } }' /sys/fs/cgroup/cpu.max 2>/dev/null)"
  [ -z "${jobs:-}" ] && jobs=4
  ok "CUDA_ARCH=$arch JOBS=$jobs"
  rm -rf /tmp/llama.cpp /tmp/lbuild
  git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp || die "clone failed"
  cmake -S /tmp/llama.cpp -B /tmp/lbuild -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$arch" -DGGML_CUDA_FA_ALL_QUANTS=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=ON \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache || die "configure failed"
  cmake --build /tmp/lbuild --target llama-server -j"$jobs" || die "build failed"
  cp /tmp/lbuild/bin/llama-server "$BIN"
  cp -a /tmp/lbuild/bin/lib*.so* "$BIN_DIR/" 2>/dev/null || true
  bin_usable || die "built binary still unusable (ldd check failed)"
  ok "built & cached"
}

ensure_models(){
  hf_get "$MODEL_REPO" "$MODEL_FILE" "$MODEL" || die "model download failed: $MODEL_FILE"
  if [ "$MMPROJ" = "1" ]; then
    hf_get "$MODEL_REPO" "$MM_FILE" "$MM_PATH" || die "mmproj download failed: $MM_FILE"
  fi
  if [ "$MTP" = "1" ]; then
    if ! hf_get "$DRAFT_REPO" "$DRAFT_FILE" "$DRAFT_PATH"; then
      warn "draft model unavailable -> MTP disabled"
      MTP=0
    fi
  fi
}

A=(); D=()
build_args(){
  local help
  help="$("$BIN" --help 2>&1)"
  [ -z "$help" ] && die "binary unusable: $BIN --help produced no output"
  has(){ printf '%s' "$help" | grep -q -- "$1"; }
  A=(-m "$MODEL" -ngl "$NGL" -c "$CTX" --host 0.0.0.0 --port "$PORT"
     --temp "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K")
  if has '--jinja'; then A+=(--jinja); else warn "no --jinja -> old build, run: bash $SELF update"; fi
  if has '--flash-attn'; then
    local fl; fl="$(printf '%s' "$help" | grep -- '--flash-attn' | head -1)"
    case "$fl" in *auto*) A+=(-fa on);; *) A+=(-fa);; esac
  fi
  has '--batch-size'    && A+=(-b "$BATCH")
  has '--ubatch-size'   && A+=(-ub "$UBATCH")
  has '--parallel'      && A+=(--parallel "$PARALLEL")
  has '--cont-batching' && A+=(--cont-batching)
  has '--slot-save-path' && A+=(--slot-save-path "$WS/kv")
  [ "$CACHE_REUSE" != "0" ] && has '--cache-reuse' && A+=(--cache-reuse "$CACHE_REUSE")
  [ -n "$KV_Q" ] && has '--cache-type-k' && A+=(--cache-type-k "$KV_Q" --cache-type-v "$KV_Q")
  [ -n "$SPLIT_MODE" ] && has '--split-mode' && A+=(--split-mode "$SPLIT_MODE")
  [ -n "$TENSOR_SPLIT" ] && has '--tensor-split' && A+=(--tensor-split "$TENSOR_SPLIT")
  if [ "$MMPROJ" = "1" ]; then
    if has '--mmproj'; then A+=(--mmproj "$MM_PATH"); ok "vision ON"
    else warn "binary has no --mmproj -> vision disabled"; fi
  fi
  if [ "$THINKING" = "1" ]; then
    has '--chat-template-kwargs' && A+=(--chat-template-kwargs '{"enable_thinking":true}')
    has '--reasoning-format' && A+=(--reasoning-format auto)
    ok "thinking ON"
  fi
  D=()
  if [ "$MTP" = "1" ]; then
    if has '--model-draft'; then
      D=(--model-draft "$DRAFT_PATH")
      has '--gpu-layers-draft' && D+=(-ngld 99)
      has '--draft-max'   && D+=(--draft-max "$DRAFT_MAX")
      has '--draft-p-min' && D+=(--draft-p-min "$DRAFT_P_MIN")
      ok "MTP speculative ON (draft-max $DRAFT_MAX)"
    else
      warn "binary has no --model-draft -> MTP disabled"
    fi
  fi
}

stop_server(){
  pkill -x llama-server 2>/dev/null
  pkill -f "$BIN" 2>/dev/null
  sleep 2
}

launch(){
  nohup "$BIN" "$@" > "$LOG" 2>&1 &
  local i
  for i in $(seq 1 300); do
    curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    pgrep -x llama-server >/dev/null 2>&1 || pgrep -f "$BIN" >/dev/null 2>&1 || return 1
    sleep 2
  done
  return 1
}

start_server(){
  say "launch Q8_0 | ctx $CTX / $PARALLEL slots | ubatch $UBATCH"
  stop_server
  build_args
  if [ ${#D[@]} -gt 0 ]; then
    if launch "${A[@]}" "${D[@]}"; then ok "READY (MTP)"; return; fi
    warn "MTP failed -> fallback to normal decoding"; tail -n 12 "$LOG" 2>/dev/null
    stop_server
  fi
  if ! launch "${A[@]}"; then
    tail -n 40 "$LOG" 2>/dev/null
    grep -qi "unknown model architecture" "$LOG" 2>/dev/null && die "binary too old for gemma4 -> bash $SELF update"
    die "server crashed (see log above)"
  fi
  ok "READY"
}

info(){
  echo; c 32 "──────────────────────────────────────────────"
  c 32 " API : http://127.0.0.1:$PORT/"
  [ -n "${RUNPOD_POD_ID:-}" ] && c 32 " URL : https://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net/"
  c 32 "──────────────────────────────────────────────"
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
  cat <<EOF

  bash $SELF bench | log | status | stop | restart | update

  curl http://127.0.0.1:$PORT/v1/chat/completions \\
    -H 'Content-Type: application/json' \\
    -d '{"messages":[{"role":"user","content":"สวัสดี ทดสอบหน่อย"}],"max_tokens":200}'
EOF
}

bench(){
  say "benchmark (200 tokens)"
  curl -s "http://127.0.0.1:$PORT/completion" -H 'Content-Type: application/json' \
    -d '{"prompt":"Explain how a GPU renders a frame in detail.","n_predict":200,"stream":false,"cache_prompt":false}' \
  | python3 -c '
import sys,json
d=json.load(sys.stdin); t=d.get("timings",{})
print("  prompt      : %.1f t/s" % t.get("prompt_per_second",0))
print("  generation  : %.1f t/s" % t.get("predicted_per_second",0))
print("  tokens out  : %s" % d.get("tokens_predicted","-"))
' 2>/dev/null || die "bench failed"
}

case "${1:-start}" in
  start)
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || die "no GPU"
    ensure_bin
    ensure_models
    start_server; info ;;
  update)
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader >/dev/null 2>&1 \
      || die "no GPU (needed to detect CUDA arch)"
    rm -rf "$BIN_DIR"; mkdir -p "$BIN_DIR"
    FORCE_BUILD=1 ensure_bin
    ok "updated -> bash $SELF restart" ;;
  stop)
    stop_server && ok stopped || ok "not running" ;;
  restart)
    ensure_bin
    ensure_models
    start_server; info ;;
  status)
    pgrep -x llama-server -a 2>/dev/null || echo "  not running"
    curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 \
      && ok "health OK" || echo "  health: down"
    nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader 2>/dev/null || true ;;
  log|logs)
    if [ -f "$LOG" ]; then tail -n 60 -f "$LOG"; else echo "  no log yet: $LOG"; fi ;;
  bench)   bench ;;
  *) echo "usage: bash $SELF [start|stop|restart|status|log|bench|update]" ;;
esac
