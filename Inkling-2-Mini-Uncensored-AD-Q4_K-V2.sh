#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# RunPod Setup: Inkling-2-Mini-Uncensored AD-Q4_K + llama.cpp v1.1.0
# GPU: A40 (48GB) / L40 / L40S (48GB) / H100 (80GB) — VRAM >= 24GB
# ============================================================

MODEL_REPO="AtomicChat/Inkling-2-Mini-Uncensored-GGUF"
MODEL_FILE="Inkling-2-Mini-Uncensored-AD-Q4_K.gguf"
MODEL_DIR="/root/models"
BUILD_DIR="/root/llama.cpp"
LLAMA_VERSION="v1.1.0"
CUDA_ARCH="86;89;90"          # A40=86, L40/L40S=89, H100=90 (แก้ให้ตรง GPU เพื่อ build เร็วขึ้น)
CTX_SIZE="${CTX_SIZE:-32768}"
N_GPU_LAYERS="${N_GPU_LAYERS:-999}"
PORT="${PORT:-8080}"

echo "=== [1/5] Install dependencies ==="
apt-get update -qq
apt-get install -y -qq git wget build-essential >/dev/null

if ! command -v cmake &>/dev/null; then
    pip install -q cmake
fi

echo "=== [2/5] Clone llama.cpp ${LLAMA_VERSION} ==="
rm -rf "${BUILD_DIR}"
git clone --depth 1 --branch "${LLAMA_VERSION}" \
    https://github.com/ggml-org/llama.cpp "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "=== [3/5] Build llama.cpp with CUDA (arch: ${CUDA_ARCH}) ==="
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
    -DLLAMA_CURL=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release \
    --target llama-server llama-cli \
    -j"$(nproc)"

echo "=== [4/5] Download model (16.53 GB) ==="
mkdir -p "${MODEL_DIR}"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"

if [ ! -f "${MODEL_PATH}" ]; then
    wget -q --show-progress -O "${MODEL_PATH}" \
        "https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}"
else
    echo "Model already exists: ${MODEL_PATH}"
fi
ls -lh "${MODEL_PATH}"

echo "=== [5/5] Launch llama-server ==="
export CUDA_SCALE_LAUNCH_QUEUES=4x

exec "${BUILD_DIR}/build/bin/llama-server" \
    -m "${MODEL_PATH}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    -ngl "${N_GPU_LAYERS}" \
    -c "${CTX_SIZE}" \
    --flash-attn on \
    --jinja \
    -b 2048 \
    -ub 512 \
    --threads "$(nproc)" \
    --load-mode none
