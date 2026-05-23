#!/bin/bash
# omnivoice-fast — OmniVoice TTS server launcher.
#
#   ./serve.sh                          # blocked streaming (default)
#   BLOCKED_STREAMING=false ./serve.sh  # full-clip optimized
#   BASELINE=true ./serve.sh            # upstream vllm-omni baseline
#
# Listens on 0.0.0.0:8091. BASELINE expects upstream cloned at
# ../vllm-omni-upstream with its own venv (see README 'Setup').

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${BASELINE:-false}" = "true" ]; then
    UPSTREAM_DIR="${SCRIPT_DIR}/../vllm-omni-upstream"
    if [ ! -d "${UPSTREAM_DIR}" ]; then
        echo "ERROR: ${UPSTREAM_DIR} not found. See README 'Setup'." >&2
        exit 1
    fi
    source "${UPSTREAM_DIR}/.venv/bin/activate"
    cd "${UPSTREAM_DIR}"
    echo "==> baseline (upstream vllm-omni)"
    exec vllm serve k2-fsa/OmniVoice \
        --host 0.0.0.0 \
        --port 8091 \
        --trust-remote-code \
        --omni
fi

source "${SCRIPT_DIR}/.venv/bin/activate"

# Tuned defaults verified on H100. Each variable can be overridden from the
# environment.
export VLLM_OMNI_OMNIVOICE_NUM_STEP="${VLLM_OMNI_OMNIVOICE_NUM_STEP:-32}"
export VLLM_OMNI_OMNIVOICE_STREAM_HOLDBACK_FRAMES="${VLLM_OMNI_OMNIVOICE_STREAM_HOLDBACK_FRAMES:-2}"

export VLLM_OMNI_OMNIVOICE_OPT="${VLLM_OMNI_OMNIVOICE_OPT:-1}"
export VLLM_OMNI_OMNIVOICE_COMPILE_MODE="${VLLM_OMNI_OMNIVOICE_COMPILE_MODE:-default}"
export VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE="${VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE:-bf16}"

export VLLM_OMNI_DIFFUSION_CONCURRENT="${VLLM_OMNI_DIFFUSION_CONCURRENT:-1}"
export VLLM_OMNI_DIFFUSION_BATCH_SIZE="${VLLM_OMNI_DIFFUSION_BATCH_SIZE:-16}"
export VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS="${VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS:-10}"
export VLLM_OMNI_DIFFUSION_BATCH_STRATEGY="${VLLM_OMNI_DIFFUSION_BATCH_STRATEGY:-duration_bucket}"
export VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS="${VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS:-128}"

export VLLM_OMNI_OMNIVOICE_VOICE_MAP="${SCRIPT_DIR}/voices.json"
export VLLM_OMNI_OMNIVOICE_DEFAULT_VOICE="${VLLM_OMNI_OMNIVOICE_DEFAULT_VOICE:-default}"

if [ "${BLOCKED_STREAMING:-true}" = "true" ]; then
    echo "==> blocked streaming (this fork)"
    export VLLM_OMNI_OMNIVOICE_BLOCK_SIZE="${VLLM_OMNI_OMNIVOICE_BLOCK_SIZE:-32}"
    export VLLM_OMNI_OMNIVOICE_FIRST_BLOCK_NUM_STEP="${VLLM_OMNI_OMNIVOICE_FIRST_BLOCK_NUM_STEP:-8}"
else
    echo "==> full-clip optimized (this fork)"
    export VLLM_OMNI_OMNIVOICE_BLOCK_SIZE=0
    export VLLM_OMNI_OMNIVOICE_FIRST_BLOCK_NUM_STEP=32
fi

cd "${SCRIPT_DIR}"
exec vllm serve k2-fsa/OmniVoice \
    --host 0.0.0.0 \
    --port 8091 \
    --trust-remote-code \
    --omni \
    --step-execution
