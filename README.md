# omnivoice-fast

Companion code to *[Why Is Diffusion TTS Slow? A Roofline Analysis of
OmniVoice Inference](#)*. Three OmniVoice serving configurations referenced
in the paper.

## Results (H100 80GB, single client, N=10)

| Mode | TTFA p50 | vs. baseline |
|---|---|---|
| baseline (upstream `vllm-omni`) | **647 ms** | 1.0× |
| full-clip optimized             | **383 ms** | 1.7× |
| blocked streaming               | **114 ms** | 5.7× |

Reproduce with `python benchmark.py`.

## About OmniVoice

[`k2-fsa/OmniVoice`](https://huggingface.co/k2-fsa/OmniVoice) is a
zero-shot multilingual TTS model (600+ languages) built on a diffusion
language model architecture, with voice cloning and voice design. The
HuggingFace model card links to the paper, source repo, demo Space, and
Colab.

## Run a mode

```bash
./serve.sh                              # blocked streaming  (default)
BLOCKED_STREAMING=false ./serve.sh      # full-clip optimized
BASELINE=true ./serve.sh                # upstream baseline
```

Server listens on `http://localhost:8091`.

## Call the server

```bash
# Non-streaming WAV
curl http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"model":"k2-fsa/OmniVoice","input":"Hello world","response_format":"wav"}' \
    -o hello.wav

# Streaming PCM
curl http://localhost:8091/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"model":"k2-fsa/OmniVoice","input":"Hello world","response_format":"pcm","stream":true}' \
    --output - | aplay -f S16_LE -r 24000 -c 1
```

## Setup

Needs CUDA, Python 3.12, `uv`.

```bash
git clone https://github.com/pavanyellow/omnivoice-fast.git
cd omnivoice-fast
uv venv --python 3.12
uv pip install vllm==0.19.0
uv pip install -e .
uv pip install "transformers>=5.3,<6"
```

For the baseline mode, also clone upstream as a sibling:

```bash
cd ..
git clone https://github.com/vllm-project/vllm-omni.git vllm-omni-upstream
cd vllm-omni-upstream
uv venv --python 3.12
uv pip install vllm==0.21.0
uv pip install -e .
uv pip install "transformers>=5.3,<6"
```

Apache-2.0 — inherits from
[`vllm-project/vllm-omni`](https://github.com/vllm-project/vllm-omni).
