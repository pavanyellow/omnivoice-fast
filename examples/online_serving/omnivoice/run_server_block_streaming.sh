#!/bin/bash
# Launch OmniVoice with block-wise audio streaming enabled.
#
# This keeps requests on the diffusion step scheduler so multiple active
# /v1/audio/speech stream=True requests can share batched generator forwards.

set -euo pipefail

MODEL="${MODEL:-k2-fsa/OmniVoice}"
PORT="${PORT:-8091}"
VLLM_BIN="${VLLM_BIN:-vllm}"

export VLLM_OMNI_DIFFUSION_CONCURRENT="${VLLM_OMNI_DIFFUSION_CONCURRENT:-1}"
export VLLM_OMNI_DIFFUSION_BATCH_SIZE="${VLLM_OMNI_DIFFUSION_BATCH_SIZE:-16}"
export VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS="${VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS:-10}"
export VLLM_OMNI_DIFFUSION_BATCH_STRATEGY="${VLLM_OMNI_DIFFUSION_BATCH_STRATEGY:-duration_bucket}"
export VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS="${VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS:-128}"

export VLLM_OMNI_OMNIVOICE_OPT="${VLLM_OMNI_OMNIVOICE_OPT:-1}"
export VLLM_OMNI_OMNIVOICE_COMPILE_MODE="${VLLM_OMNI_OMNIVOICE_COMPILE_MODE:-default}"
export VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE="${VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE:-bf16}"
export VLLM_OMNI_OMNIVOICE_NUM_STEP="${VLLM_OMNI_OMNIVOICE_NUM_STEP:-32}"
export VLLM_OMNI_OMNIVOICE_BLOCK_SIZE="${VLLM_OMNI_OMNIVOICE_BLOCK_SIZE:-32}"
export VLLM_OMNI_OMNIVOICE_STREAM_HOLDBACK_FRAMES="${VLLM_OMNI_OMNIVOICE_STREAM_HOLDBACK_FRAMES:-2}"
export VLLM_OMNI_OMNIVOICE_VOICE_MAP="${VLLM_OMNI_OMNIVOICE_VOICE_MAP:-}"
export VLLM_OMNI_OMNIVOICE_DEFAULT_VOICE="${VLLM_OMNI_OMNIVOICE_DEFAULT_VOICE:-}"

echo "Starting block-streaming OmniVoice server"
echo "  model: $MODEL"
echo "  port: $PORT"
echo "  vllm bin: $VLLM_BIN"
echo "  diffusion batch size: $VLLM_OMNI_DIFFUSION_BATCH_SIZE"
echo "  diffusion batch wait ms: $VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS"
echo "  diffusion batch strategy: $VLLM_OMNI_DIFFUSION_BATCH_STRATEGY"
echo "  omnivoice num_step: $VLLM_OMNI_OMNIVOICE_NUM_STEP"
echo "  omnivoice block_size: $VLLM_OMNI_OMNIVOICE_BLOCK_SIZE"
echo "  stream holdback frames: $VLLM_OMNI_OMNIVOICE_STREAM_HOLDBACK_FRAMES"
if [[ -n "$VLLM_OMNI_OMNIVOICE_VOICE_MAP" ]]; then
    echo "  voice map: $VLLM_OMNI_OMNIVOICE_VOICE_MAP"
fi
if [[ -n "$VLLM_OMNI_OMNIVOICE_DEFAULT_VOICE" ]]; then
    echo "  default voice: $VLLM_OMNI_OMNIVOICE_DEFAULT_VOICE"
fi

"$VLLM_BIN" serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --omni \
    --step-execution
