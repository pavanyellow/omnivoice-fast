#!/bin/bash
# Launch OmniVoice with the serving settings used by the H100 batching tests.
#
# Defaults are intentionally env-overridable so the scheduler strategy can be
# swept without editing this file.

set -euo pipefail

MODEL="${MODEL:-k2-fsa/OmniVoice}"
PORT="${PORT:-8091}"
VLLM_BIN="${VLLM_BIN:-vllm}"

export VLLM_OMNI_DIFFUSION_CONCURRENT="${VLLM_OMNI_DIFFUSION_CONCURRENT:-1}"
export VLLM_OMNI_DIFFUSION_BATCH_SIZE="${VLLM_OMNI_DIFFUSION_BATCH_SIZE:-32}"
export VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS="${VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS:-0}"
export VLLM_OMNI_DIFFUSION_BATCH_STRATEGY="${VLLM_OMNI_DIFFUSION_BATCH_STRATEGY:-duration_bucket}"
export VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS="${VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS:-128}"
export VLLM_OMNI_OMNIVOICE_OPT="${VLLM_OMNI_OMNIVOICE_OPT:-1}"
export VLLM_OMNI_OMNIVOICE_COMPILE_MODE="${VLLM_OMNI_OMNIVOICE_COMPILE_MODE:-default}"
export VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE="${VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE:-bf16}"
export VLLM_OMNI_OMNIVOICE_NUM_STEP="${VLLM_OMNI_OMNIVOICE_NUM_STEP:-16}"

echo "Starting optimized OmniVoice server"
echo "  model: $MODEL"
echo "  port: $PORT"
echo "  vllm bin: $VLLM_BIN"
echo "  diffusion concurrent: $VLLM_OMNI_DIFFUSION_CONCURRENT"
echo "  diffusion batch size: $VLLM_OMNI_DIFFUSION_BATCH_SIZE"
echo "  diffusion batch wait ms: $VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS"
echo "  diffusion batch strategy: $VLLM_OMNI_DIFFUSION_BATCH_STRATEGY"
echo "  diffusion duration bucket tokens: $VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS"
echo "  omnivoice opt: $VLLM_OMNI_OMNIVOICE_OPT"
echo "  omnivoice compile mode: $VLLM_OMNI_OMNIVOICE_COMPILE_MODE"
echo "  omnivoice generator dtype: $VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE"
echo "  omnivoice num_step: $VLLM_OMNI_OMNIVOICE_NUM_STEP"

"$VLLM_BIN" serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --omni
