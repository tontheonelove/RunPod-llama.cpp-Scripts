#!/usr/bin/env bash
# Standalone — RunPod | Qwen3.8 27B Ultimate Heretic 2.1ARA NEO Q6_K (22.1 GB)
# https://huggingface.co/DavidAU/Qwen3.8-27B-Ultimate-Heretic-Uncensored-Turbo-Brilliance-50x-AGI-NEO-GGUF
# usage: bash run-q6.sh [start|stop|restart|status|log|bench|update]
set -uo pipefail
if [ -f "$0" ] && grep -q $'\r' "$0" 2>/dev/null; then sed -i 's/\r$//' "$0"; exec bash "$0" "$@"; fi

######################## CONFIG ########################
WS="${WS:-/workspace}"
PORT="${PORT:-8000}"
CTX="${CTX:-131072}"           # total; per-request = CTX / PARALLEL -> 32k/slot (card suggests >=24k)
PARALLEL="${PARALLEL:-4}"      # 4 slots; MTP gains need PARALLEL=1
NGL="${NGL:-99}"
BATCH="${BATCH:-4096}"; UBATCH="${UBATCH:-1024}"
MODEL_REPO="${MODEL_REPO:-DavidAU/Qwen3.8-27B-Ultimate-Heretic-Uncensored-Turbo-Brilliance-50x-AGI-NEO-GGUF}"
MODEL_FILE="${MODEL_FILE:-Qwen3.8-27B-Ultimate-Heretic-2.1ARA-Uncn-Turbo-Brilli-50x-AGI-NEO-Q6_K.gguf}"
MTP="${MTP:-0}"                # 1 = MTP speculative decoding (best with PARALLEL=1), auto-fallback
DRAFT_MAX="${DRAFT_MAX:-3}"; DRAFT_P_MIN="${DRAFT_P_MIN:-0}"
DRAFT_REPO="${DRAFT_REPO:-unsloth/Qwen3.8-27B-GGUF}"        # sidecar MTP head, 1.37 GB (non-MTP quant only)
DRAFT_FILE="${DRAFT_FILE:-MTP/mtp-Qwen3.8-27B-Q4_0.gguf}"
MMPROJ="${MMPROJ:-0}"          # 1 = vision (+928MB)
MM_FILE="${MM_FILE:-mmproj-F16.gguf}"
THINKING="${THINKING:-1}"      # Qwen3.8 hybrid thinking; 0 = enable_thinking:false
EFFORT="${EFFORT:-}"           # "" = model default (xhigh) | medium | low
KV_Q="${KV_Q:-q8_0}"           # q8_0 = ~half KV VRAM; "" = f16
CACHE_REUSE="${CACHE_REUSE:-0}"  # model card: no caching suggested
TEMP="${TEMP:-1.0}"; TOP_P="${TOP_P:-0.95}"; TOP_K="${TOP_K:-64}"; MIN_P="${MIN_P:-0.05}"
SPLIT_MODE="${SPLIT_MODE:-}"; TENSOR_SPLIT="${TENSOR_SPLIT:-}"
HF_TOKEN="${HF_TOKEN:-}"; BIN_URL="${BIN_URL:-}"; FORCE_BUILD="${FORCE_BUILD:-0}"
########################################################

BIN_DIR="$WS/bin"; BIN="$BIN_DIR/llama-server"
MODEL="$WS/models/$MODEL_FILE"
MM_PATH="$WS/models/$MM_FILE"
DRAFT_PATH="$WS/models/$(basename "$DRAFT_FILE")"
LOG="$WS/logs/server.log"
export HF_HOME="$WS/hf" CCACHE_DIR="$WS/.ccache" DEBIAN_FRONTEND=noninteractive
export LD_LIBRARY_PATH="$BIN_DIR:${LD_LIBRARY_PATH:-}"
mkdir -p "$WS/models" "$BIN_DIR" "$WS/logs" "$WS/hf" "$WS/.ccache" "$WS/kv"

c(){ printf '\033[1;%sm%s\033[0m\n' "$1" "$2"; }
say(){ c 36 "==> $*"; }; ok(){ c 32 "  ✓ $*"; }
warn(){ c 33 "  ! $*"; }; die(){ c 31 "  ✗ $*"; exit 1; }

hf_get(){ # repo file dest
  local url="https://huggingface.co/$1/resolve/main/$2"
  local auth=(); [ -n "$HF_TOKEN" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
  [ -s "$3" ] && { ok "cached: $(basename "$3") ($(du -h "$3"|cut -f1))"; return; }
  say "downloading $2"
  curl -L --retry 5 --retry-delay 3 -C - "${auth[@]}" -o "$3.part" "$url" || die "download failed: $2"
  mv "$3.part" "$3"; ok "saved ($(du -h "$3"|cut -f1))"
}

ensure_bin(){
  say "llama-server binary"
  if [ -x "$BIN" ] && [ "$FORCE_BUILD" != "1" ]; then ok "cached (no compile)"; return; fi
  if [ "$FORCE_BUILD" != "1" ]; then
    if [ -n "$BIN_URL" ]; then
      command -v unzip >/dev/null || { apt-get update -qq; apt-get install -y -qq unzip; }
      curl -L --retry 3 -o /tmp/llama.zip "$BIN_URL" || die "zip failed"
      unzip -oq /tmp/llama.zip -d "$BIN_DIR"
      find "$BIN_DIR" -name llama-server -exec mv {} "$BIN" \; 2>/dev/null
      [ -x "$BIN" ] && { ok "prebuilt extracted"; return; }
    fi
    local found=""
    for p in "$(command -v llama-server 2>/dev/null)" /app/llama-server /app/build/bin/llama-server /usr/local/bin/llama-server; do
      if [ -n "$p" ] && [ -x "$p" ]; then found="$p"; break; fi
    done
    if [ -n "$found" ]; then
      cp "$found" "$BIN"; cp -a "$(dirname "$found")"/lib*.so* "$BIN_DIR/" 2>/dev/null
      ok "reused image binary (no compile)"; return
    fi
  fi
  say "building latest llama.cpp (ONE TIME, cached to $BIN)"
  apt-get update -qq
  apt-get install -y -qq git cmake ninja-build ccache build-essential libcurl4-openssl-dev ca-certificates
  command -v nvcc >/dev/null || die "nvcc missing -> use CUDA devel image"
  local arch jobs
  arch="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')"
  jobs="$(awk '{ if ($1=="max") print 4; else { n=int($1/$2); print (n<1?1:n) } }' /sys/fs/cgroup/cpu.max 2>/dev/null)"
  [ -z "${jobs:-}" ] && jobs=4
  ok "CUDA_ARCH=$arch JOBS=$jobs"
  rm -rf /tmp/llama.cpp /tmp/lbuild
  git clone --depth 1 https://github.com/ggml-org/llama.cpp /tmp/llama.cpp || die "clone failed"
  cmake -S /tmp/llama.cpp -B /tmp/lbuild -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$arch" -DGGML_CUDA_FA_ALL_QUANTS=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_CUDA_COMPILER_LAUNCHER=ccache || die "configure failed"
  cmake --build /tmp/lbuild --target llama-server -j"$jobs" || die "build failed"
  cp /tmp/lbuild/bin/llama-server "$BIN"; ok "built & cached"
}

A=(); D=()
build_args(){
  local help; help="$("$BIN" --help 2>&1)"
  has(){ printf '%s' "$help" | grep -q -- "$1"; }
  A=(-m "$MODEL" -ngl "$NGL" -c "$CTX" --host 0.0.0.0 --port "$PORT"
     --temp "$TEMP" --top-p "$TOP_P" --top-k "$TOP_K")
  has '--min-p' && A+=(--min-p "$MIN_P")
  has '--jinja' && A+=(--jinja) || warn "no --jinja -> old build, run: bash run-q6.sh update"
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
  [ "$MMPROJ" = "1" ] && has '--mmproj' && A+=(--mmproj "$MM_PATH") && ok "vision ON ($MM_FILE)"
  if [ "$THINKING" = "1" ]; then
    local kw='{"enable_thinking":true'
    [ -n "$EFFORT" ] && kw="$kw,\"reasoning_effort\":\"$EFFORT\""
    kw="$kw}"
    has '--chat-template-kwargs' && A+=(--chat-template-kwargs "$kw")
    has '--reasoning-format' && A+=(--reasoning-format auto)
    ok "thinking ON (effort: ${EFFORT:-xhigh [model default]})"
  else
    has '--chat-template-kwargs' && A+=(--chat-template-kwargs '{"enable_thinking":false}')
    ok "thinking OFF (instruct)"
  fi
  D=()
  if [ "$MTP" = "1" ]; then
    if has '--spec-type'; then
      D=(--spec-type draft-mtp)
      case "$MODEL_FILE" in
        *-MTP-*) ok "MTP: head embedded in quant (no sidecar)" ;;
        *) D+=(--spec-draft-model "$DRAFT_PATH")
           has '--spec-draft-ngl' && D+=(-ngld 99) ;;
      esac
      has '--spec-draft-n-max' && D+=(--spec-draft-n-max "$DRAFT_MAX")
      has '--spec-draft-p-min' && D+=(--spec-draft-p-min "$DRAFT_P_MIN")
      ok "MTP speculative ON (n-max $DRAFT_MAX)"
      [ "$PARALLEL" != "1" ] && warn "MTP gains fade with parallel slots -> use PARALLEL=1 for max speed"
    else
      warn "binary has no --spec-type -> MTP skipped (run: bash run-q6.sh update)"
    fi
  fi
}

launch(){
  nohup "$BIN" "$@" > "$LOG" 2>&1 &
  local i
  for i in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && return 0
    pgrep -f llama-server >/dev/null || return 1
    sleep 2
  done
  return 1
}

start_server(){
  say "launch Qwen3.8-27B Heretic NEO Q6_K | port $PORT | ctx $CTX / $PARALLEL slots"
  pkill -f llama-server 2>/dev/null; sleep 2
  build_args
  if [ ${#D[@]} -gt 0 ]; then
    launch "${A[@]}" "${D[@]}" && { ok "READY (MTP)"; return; }
    warn "MTP failed -> fallback to normal decoding"; tail -n 12 "$LOG"
    pkill -f llama-server 2>/dev/null; sleep 2
  fi
  if ! launch "${A[@]}"; then
    tail -n 40 "$LOG"
    grep -qi "unknown model architecture" "$LOG" && die "binary too old for qwen3.8 (qwen35 arch) -> bash run-q6.sh update"
    die "server crashed (see log above)"
  fi
  ok "READY"
}

info(){
  echo; c 32 "──────────────────────────────────────────────"
  c 32 " API : http://127.0.0.1:$PORT/v1"
  [ -n "${RUNPOD_POD_ID:-}" ] && c 32 " URL : https://${RUNPOD_POD_ID}-${PORT}.proxy.runpod.net/v1"
  c 32 "──────────────────────────────────────────────"
  nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
  cat <<EOF

  bash run-q6.sh bench | log | status | stop | restart | update

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
print("  prompt     : %.1f t/s" % t.get("prompt_per_second",0))
print("  generation : %.1f t/s" % t.get("predicted_per_second",0))
' 2>/dev/null || die "bench failed"
}

case "${1:-start}" in
  start)
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || die "no GPU"
    ensure_bin
    hf_get "$MODEL_REPO" "$MODEL_FILE" "$MODEL"
    [ "$MMPROJ" = "1" ] && hf_get "$MODEL_REPO" "$MM_FILE" "$MM_PATH"
    if [ "$MTP" = "1" ]; then
      case "$MODEL_FILE" in
        *-MTP-*) ok "MTP quant: no sidecar download needed" ;;
        *) hf_get "$DRAFT_REPO" "$DRAFT_FILE" "$DRAFT_PATH" ;;
      esac
    fi
    start_server; info ;;
  update)  rm -rf "$BIN_DIR"; mkdir -p "$BIN_DIR"; FORCE_BUILD=1 ensure_bin; ok "updated -> bash run-q6.sh restart" ;;
  stop)    pkill -f llama-server && ok stopped || ok "not running" ;;
  restart) pkill -f llama-server 2>/dev/null; sleep 2; ensure_bin; start_server; info ;;
  status)  pgrep -af llama-server || echo "  not running"
           curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && ok "health OK" || echo "  health: down"
           nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader ;;
  log|logs) tail -n 60 -f "$LOG" ;;
  bench)   bench ;;
  *) echo "usage: bash run-q6.sh [start|stop|restart|status|log|bench|update]" ;;
esac
