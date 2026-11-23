#!/usr/bin/env bash
# =============================================================================
# Inkling-2-Mini-Uncensored-AD-Q4_K.sh
#
# Single-file deployer สำหรับ:
#   Model   : AtomicChat/Inkling-2-Mini-Uncensored-GGUF
#             (AD-Q4_K ~16.5GB + mmproj F16 ~0.64GB)
#   Runtime : llama.cpp tag b12600+ (จำเป็น — รองรับสถาปัตยกรรม inkling2_loop)
#   Target  : RunPod GPU VRAM >= 24GB (ทดสอบบน A40/48GB)
#   Port    : 8080 (Web UI + OpenAI-compatible API)
#
# Usage:
#   chmod +x Inkling-2-Mini-Uncensored-AD-Q4_K.sh
#   ./Inkling-2-Mini-Uncensored-AD-Q4_K.sh
#
# Env overrides:
#   HF_TOKEN=hf_xxx         # สำหรับ private repo / เลี่ยง rate limit
#   FORCE_REBUILD=1         # build llama.cpp ใหม่
#   FORCE_REDOWNLOAD=1      # ดาวน์โหลดโมเดลใหม่
#   SKIP_START=1            # เตรียมทุกอย่าง แต่ไม่เปิด server
#   PORT=8080               # เปลี่ยนพอร์ต
#   CTX=32768               # เปลี่ยน context length
#   WORKDIR=/workspace/inkling
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

# ---------- Helpers ----------
log()  { printf "\n\033[1;36m==> %s\033[0m\n" "$*"; }
warn() { printf "\n\033[1;33m[WARN] %s\033[0m\n" "$*"; }
die()  { printf "\n\033[1;31m[ERROR] %s\033[0m\n" "$*" >&2; exit 1; }

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
    if [[ -f "$marker" && "${FORCE_REBUILD:-0}" != "1" ]]; then
        log "พบ marker: system dependencies ติดตั้งแล้ว — ข้าม"
    else
        log "ติดตั้ง system dependencies"
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -y
        apt-get install -y --no-install-recommends \
            build-essential cmake git curl ca-certificates \
            libcurl4-openssl-dev libssl-dev pkg-config ninja-build \
            python3 python3-pip
        touch "$marker"
    fi

    log "ตรวจสอบ huggingface_hub (Python API)"
    python3 -c "import huggingface_hub" 2>/dev/null \
        || pip install -q --upgrade huggingface_hub
}

detect_cuda_arch() {
    local cap
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d ' ')
    if [[ -z "$cap" || "$cap" == "N/A" ]]; then
        echo "86"
    else
        echo "${cap/./}"
    fi
}

build_llama() {
    local arch
    arch=$(detect_cuda_arch)

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

    log "Build llama.cpp ด้วย CUDA (sm_${arch})"
    cmake -B build \
        -DGGML_CUDA=ON \
        -DGGML_NATIVE=OFF \
        -DCMAKE_CUDA_ARCHITECTURES="${arch}" \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build build --config Release -j"$(nproc)" --target llama-server

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
    python3 - "${PORT}" <<'PY' && die "พอร์ต ${PORT} ถูกใช้งานอยู่แล้ว"
import socket, sys
p = int(sys.argv[1])
s = socket.socket()
s.settimeout(0.5)
ok = s.connect_ex(("127.0.0.1", p)) == 0
s.close()
sys.exit(0 if ok else 1)
PY
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

    log "เปิด llama-server ที่ ${HOST}:${PORT} (ctx=${CTX})"
    echo "    Web UI : http://${HOST}:${PORT}/"
    echo "    API    : http://${HOST}:${PORT}/v1"
    echo "    Health : http://${HOST}:${PORT}/health"
    echo "    (กด Ctrl+C เพื่อหยุด)"
    echo

    exec "$BIN" \
        -m "${MODEL_DIR}/${MODEL_FILE}" \
        --mmproj "${MODEL_DIR}/${MMPROJ_FILE}" \
        --host "$HOST" \
        --port "$PORT" \
        --jinja \
        -c "$CTX" \
        -ngl 99 \
        --parallel 1 \
        --flash-attn on \
        --cache-type-k q4_0 \
        --cache-type-v q4_0
}

# ---------- Main ----------
main() {
    mkdir -p "$WORKDIR" "$MODEL_DIR"

    log "Inkling-2-Mini-Uncensored — single-file deployer"
    check_env
    install_deps
    build_llama
    download_model
    start_server
}

main "$@"
