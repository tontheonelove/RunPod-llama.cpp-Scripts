#!/usr/bin/env bash
# =============================================================================
# Inkling-2-Mini-Uncensored-AD-Q4_K.sh  (v2 - hardened)
#
# Single-file deployer สำหรับ:
#   Model   : AtomicChat/Inkling-2-Mini-Uncensored-GGUF
#             (AD-Q4_K ~16.5GB + mmproj F16 ~0.64GB)
#   Runtime : llama.cpp tag b12600+ (จำเป็น — รองรับสถาปัตยกรรม inkling2_loop)
#   Target  : RunPod GPU VRAM >= 24GB (ทดสอบบน A40/48GB, L40/48GB)
#   Port    : 8080 (Web UI + OpenAI-compatible API)
#
# Usage:
#   chmod +x Inkling-2-Mini-Uncensored-AD-Q4_K.sh
#   ./Inkling-2-Mini-Uncensored-AD-Q4_K.sh
#
# Env overrides:
#   HF_TOKEN=hf_xxx         # สำหรับ private repo / เลี่ยง rate limit
#   FORCE_DEPS=1            # ติดตั้ง system dependencies ใหม่ (ไม่ผูกกับ rebuild)
#   FORCE_REBUILD=1         # build llama.cpp ใหม่
#   FORCE_REDOWNLOAD=1      # ดาวน์โหลดโมเดลใหม่
#   SKIP_START=1            # เตรียมทุกอย่าง แต่ไม่เปิด server
#   PORT=8080               # เปลี่ยนพอร์ต
#   CTX=32768               # เปลี่ยน context length
#   WORKDIR=/workspace/inkling
#
# Advanced:
#   KV_K=f16 KV_V=f16       # override KV cache type (default: auto ตาม VRAM)
#   BUILD_JOBS_MAX=16       # cap parallel build jobs (กัน OOM ตอน compile CUDA)
#   ALIAS=inkling-2-mini    # model alias ที่ API ตอบ
#   NO_MMAP=1               # load โมเดลลง RAM/VRAM ทันที (ไม่ใช้ mmap)
#   METRICS=1               # เปิด /metrics endpoint
#   EXTRA_ARGS="--foo bar"  # ส่ง flag เพิ่มเติมให้ llama-server
# =============================================================================
set -euo pipefail

# ---------- Config ----------
WORKDIR="${WORKDIR:-/workspace/inkling}"
LLAMA_DIR="${WORKDIR}/llama.cpp"
MODEL_DIR="${WORKDIR}/models"
BIN="${LLAMA_DIR}/build/bin/llama-server"
LLAMA_TAG="${LLAMA_TAG:-b12600}"
REPO_ID="AtomicChat/Inkling-2-Mini-Uncensored-GGUF"
MODEL_FILE="Inkling-2-Mini-Uncensored-AD-Q4_K.gguf"
MMPROJ_FILE="mmproj-Inkling-2-Mini-Uncensored-F16.gguf"
HOST="0.0.0.0"
PORT="${PORT:-8080}"
CTX="${CTX:-32768}"
ALIAS="${ALIAS:-inkling-2-mini-uncensored}"
BUILD_JOBS_MAX="${BUILD_JOBS_MAX:-16}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# ---------- Helpers ----------
log()  { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }
warn() { printf "\n\033[1;33m[WARN] %s\033[0m\n" "$*"; }
die()  { printf "\n\033[1;31m[ERROR] %s\033[0m\n" "$*" >&2; exit 1; }

# ---------- Environment detection ----------
detect_cuda_arch() {
    local cap
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
          | head -n1 | tr -d ' ')

    if [[ -z "$cap" || "$cap" == "N/A" ]]; then
        warn "อ่าน compute_cap ไม่ได้ (driver เก่า?) — fallback เป็น sm_86"
        echo "86"
    else
        echo "${cap/./}"
    fi
}

detect_vram_gb() {
    local mb
    mb=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
         | head -n1 | tr -d ' ')
    if [[ -z "$mb" || "$mb" == "N/A" ]]; then
        echo "0"
    else
        echo "$(( mb / 1024 ))"
    fi
}

# ---------- Steps ----------
check_env() {
    log "ตรวจสอบสภาพแวดล้อม"
    [ "$(id -u)" -eq 0 ] || warn "ไม่ได้รันเป็น root — การติดตั้ง system package อาจล้มเหลว"

    command -v nvidia-smi &>/dev/null || die "ไม่พบ nvidia-smi — GPU/RunPod ไม่พร้อม"
    nvidia-smi &>/dev/null || die "nvidia-smi ทำงานไม่สำเร็จ — ตรวจสอบ GPU driver"

    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

    export PATH="/usr/local/cuda/bin:${PATH}"
    command -v nvcc &>/dev/null || die "ไม่พบ nvcc — image นี้ควรมี CUDA toolkit อยู่แล้ว"
    nvcc --version | tail -n1
}

install_deps() {
    local marker="${WORKDIR}/.deps_installed"
    if [[ -f "$marker" && "${FORCE_DEPS:-0}" != "1" ]]; then
        log "พบ marker: system dependencies ติดตั้งแล้ว — ข้าม"
    else
        log "ติดตั้ง system dependencies"
        export DEBIAN_FRONTEND=noninteractive
        apt-get update
        apt-get install -y --no-install-recommends \
            build-essential cmake git curl ca-certificates \
            libcurl4-openssl-dev libssl-dev pkg-config ninja-build \
            python3 python3-pip
        touch "$marker"
    fi

    log "ตรวจสอบ huggingface_hub (Python API)"
    python3 -c "import huggingface_hub" 2>/dev/null && return 0

    # Ubuntu 24.04 / Debian 12+ บังคับ PEP 668 → pip ตรง ๆ จะ fail
    # ลองแบบปกติก่อน, ถ้า fail ให้ใช้ --break-system-packages
    if pip install -q --upgrade huggingface_hub 2>/dev/null; then
        return 0
    fi
    warn "pip install ปกติล้มเหลว (PEP 668?) — ลอง --break-system-packages"
    pip install -q --upgrade --break-system-packages huggingface_hub \
        || die "ติดตั้ง huggingface_hub ไม่สำเร็จ"
}

build_llama() {
    local arch jobs
    arch=$(detect_cuda_arch)
    jobs="$(nproc)"
    if (( jobs > BUILD_JOBS_MAX )); then
        jobs="$BUILD_JOBS_MAX"
    fi

    if [[ -x "$BIN" && "${FORCE_REBUILD:-0}" != "1" ]]; then
        log "พบ llama-server แล้ว — ข้ามการ build (ใช้ FORCE_REBUILD=1 เพื่อบังคับ)"
        "$BIN" --version 2>&1 | head -n1 || true
        return
    fi

    log "Clone llama.cpp (tag ${LLAMA_TAG})"
    if [[ ! -d "$LLAMA_DIR/.git" ]]; then
        git clone https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR"
    fi
    cd "$LLAMA_DIR"
    git fetch --tags
    git checkout -f "$LLAMA_TAG"

    log "Build llama.cpp ด้วย CUDA (sm_${arch}, -j${jobs})"
    cmake -B build \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${arch}" \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j"$jobs" --target llama-server

    [[ -x "$BIN" ]] || die "Build ล้มเหลว — ไม่พบ $BIN"
    log "Build สำเร็จ: $BIN"
}

download_model() {
    mkdir -p "$MODEL_DIR"

    if [[ "${FORCE_REDOWNLOAD:-0}" != "1" \
          && -s "${MODEL_DIR}/${MODEL_FILE}" \
          && -s "${MODEL_DIR}/${MMPROJ_FILE}" ]]; then
        log "พบโมเดลทั้งสองไฟล์แล้ว — ข้ามการดาวน์โหลด"
    else
        log "ดาวน์โหลดโมเดลจาก Hugging Face"
        export MODEL_DIR REPO_ID MODEL_FILE MMPROJ_FILE HF_TOKEN="${HF_TOKEN:-}"
        python3 - <<'PY'
import os
from huggingface_hub import hf_hub_download

repo_id   = os.environ["REPO_ID"]
local_dir = os.environ["MODEL_DIR"]
files     = [os.environ["MODEL_FILE"], os.environ["MMPROJ_FILE"]]
token     = os.environ.get("HF_TOKEN") or None

os.makedirs(local_dir, exist_ok=True)
for f in files:
    print(f"==> {f}")
    path = hf_hub_download(repo_id=repo_id, filename=f,
                           local_dir=local_dir, token=token)
    print(f"    {path}  ({os.path.getsize(path)/1e9:.2f} GB)")
PY
    fi

    [[ -s "${MODEL_DIR}/${MODEL_FILE}"  ]] || die "ไม่พบ ${MODEL_FILE}"
    [[ -s "${MODEL_DIR}/${MMPROJ_FILE}" ]] || die "ไม่พบ ${MMPROJ_FILE}"
}

check_port() {
    # set -e + if-block → ถ้า python exit 1 (พอร์ตว่าง) จะไม่ kill script
    if python3 - "$PORT" <<'PY'
import socket, sys
p = int(sys.argv[1])
s = socket.socket()
s.settimeout(0.5)
busy = s.connect_ex(("127.0.0.1", p)) == 0
s.close()
sys.exit(0 if busy else 1)
PY
    then
        die "พอร์ต ${PORT} ถูกใช้งานอยู่แล้ว"
    fi
    log "พอร์ต ${PORT} ว่าง — พร้อมใช้งาน"
}

start_server() {
    if [[ "${SKIP_START:-0}" == "1" ]]; then
        log "SKIP_START=1 — เตรียมไฟล์เสร็จแล้ว ไม่เปิด server"
        echo "    Binary : $BIN"
        echo "    Model  : ${MODEL_DIR}/${MODEL_FILE}"
        echo "    mmproj : ${MODEL_DIR}/${MMPROJ_FILE}"
        return
    fi
    check_port

    # ---- Auto เลือก KV cache type ตาม VRAM ----
    local vram kv_k kv_v
    vram=$(detect_vram_gb)
    if (( vram >= 40 )); then
        kv_k="${KV_K:-f16}"
        kv_v="${KV_V:-f16}"
    else
        # 24-39GB — ใช้ quantized KV เพื่อให้ context 32k พอดี
        kv_k="${KV_K:-q4_0}"
        kv_v="${KV_V:-q4_0}"
    fi

    # ---- Build argument array (ปลอดภัยกับ EXTRA_ARGS ที่มีช่องว่าง) ----
    local args=(
        -m "${MODEL_DIR}/${MODEL_FILE}"
        --mmproj "${MODEL_DIR}/${MMPROJ_FILE}"
        --host "$HOST"
        --port "$PORT"
        --alias "$ALIAS"
        --jinja
        -c "$CTX"
        -ngl 99
        --parallel 1
        --flash-attn on
        --cache-type-k "$kv_k"
        --cache-type-v "$kv_v"
    )
    [[ "${NO_MMAP:-0}"  == "1" ]] && args+=(--no-mmap)
    [[ "${METRICS:-0}"  == "1" ]] && args+=(--metrics)
    if [[ -n "$EXTRA_ARGS" ]]; then
        # shellcheck disable=SC2206
        args+=($EXTRA_ARGS)
    fi

    log "เปิด llama-server ที่ ${HOST}:${PORT}"
    echo "    VRAM   : ${vram} GB detected"
    echo "    KV     : k=${kv_k}  v=${kv_v}"
    echo "    Alias  : ${ALIAS}"
    echo "    Web UI : http://${HOST}:${PORT}/"
    echo "    API    : http://${HOST}:${PORT}/v1"
    echo "    Health : http://${HOST}:${PORT}/health"
    echo "    (กด Ctrl+C เพื่อหยุด)"
    echo

    exec "$BIN" "${args[@]}"
}

# ---------- Main ----------
main() {
    mkdir -p "$WORKDIR" "$MODEL_DIR"

    log "Inkling-2-Mini-Uncensored — single-file deployer (v2)"
    check_env
    install_deps
    build_llama
    download_model
    start_server
}

main "$@"
