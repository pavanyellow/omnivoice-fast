"""Modal deployment for OmniVoice block-wise streaming.

Usage:
    modal deploy examples/online_serving/omnivoice/modal_app.py
    modal run examples/online_serving/omnivoice/modal_app.py::smoke
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import time
import urllib.error
import urllib.request
import wave

import modal


APP_NAME = "omnivoice-vllm"
MODEL_NAME = "k2-fsa/OmniVoice"
VLLM_PORT = 8091
MINUTES = 60

REPO_URL = os.environ.get("VLLM_OMNI_REPO_URL", "https://github.com/vllm-project/vllm-omni.git")
REPO_BRANCH = os.environ.get("VLLM_OMNI_REPO_BRANCH", "a100-blockwise-streaming")
REPO_COMMIT = os.environ.get("VLLM_OMNI_REPO_COMMIT", "73df326e8029e8b254f6fc92008a09e23f0abcf1")
REPO_ARCHIVE_URL = os.environ.get(
    "VLLM_OMNI_REPO_ARCHIVE_URL",
    f"https://api.github.com/repos/vllm-project/vllm-omni/tarball/{REPO_COMMIT}",
)

GPU = os.environ.get("MODAL_GPU", "H100")
MAX_CONCURRENT_INPUTS = int(os.environ.get("MODAL_MAX_CONCURRENT_INPUTS", "16"))
TARGET_CONCURRENT_INPUTS = int(os.environ.get("MODAL_TARGET_CONCURRENT_INPUTS", str(MAX_CONCURRENT_INPUTS)))
MIN_CONTAINERS = int(os.environ.get("MODAL_MIN_CONTAINERS", "0"))
MAX_CONTAINERS = int(os.environ.get("MODAL_MAX_CONTAINERS", "2"))
SCALEDOWN_WINDOW = int(os.environ.get("MODAL_SCALEDOWN_WINDOW", str(15 * MINUTES)))

SERVER_ENV = {
    "HF_HOME": "/root/.cache/huggingface",
    "HF_XET_HIGH_PERFORMANCE": "1",
    "VLLM_CACHE_ROOT": "/root/.cache/vllm",
    "VLLM_OMNI_OMNIVOICE_FIRST_BLOCK_NUM_STEP": "8",
    "VLLM_OMNI_OMNIVOICE_NUM_STEP": "32",
    "VLLM_OMNI_OMNIVOICE_BLOCK_SIZE": "32",
    "VLLM_OMNI_DIFFUSION_CONCURRENT": "1",
    "VLLM_OMNI_DIFFUSION_BATCH_SIZE": str(MAX_CONCURRENT_INPUTS),
    "VLLM_OMNI_DIFFUSION_BATCH_WAIT_MS": "10",
    "VLLM_OMNI_DIFFUSION_BATCH_STRATEGY": "duration_bucket",
    "VLLM_OMNI_DIFFUSION_DURATION_BUCKET_TOKENS": "128",
    "VLLM_OMNI_OMNIVOICE_GENERATOR_DTYPE": "bf16",
    "VLLM_OMNI_OMNIVOICE_OPT": "1",
    "VLLM_OMNI_OMNIVOICE_COMPILE_MODE": "default",
    "VLLM_OMNI_OMNIVOICE_STREAM_HOLDBACK_FRAMES": "2",
}

DIFFUSION_ENGINE_PATCH = (
    "python3 -c \"from pathlib import Path; "
    "p = Path('/app/vllm-omni/vllm_omni/diffusion/diffusion_engine.py'); "
    "text = p.read_text(); "
    "old = '        if runner_output is not None and runner_output.result is not None:\\n"
    "            return runner_output.result\\n\\n"
    "        return DiffusionOutput(error=missing_result_error)\\n'; "
    "new = '        if runner_output is not None:\\n"
    "            per_request = runner_output.per_request_results or {}\\n"
    "            if sched_req_id in per_request:\\n"
    "                return per_request[sched_req_id]\\n"
    "            if runner_output.result is not None:\\n"
    "                return runner_output.result\\n\\n"
    "        return DiffusionOutput(error=missing_result_error)\\n'; "
    "assert old in text, 'expected DiffusionEngine finalizer block was not found'; "
    "p.write_text(text.replace(old, new, 1))\""
)

DIFFUSION_AUDIO_DICT_PATCH = (
    "python3 -c \"from pathlib import Path; "
    "p = Path('/app/vllm-omni/vllm_omni/diffusion/diffusion_engine.py'); "
    "text = p.read_text(); "
    "old = '            model_audio_sample_rate = outputs.get(\\\"audio_sample_rate\\\")\\n'; "
    "new = '            model_audio_sample_rate = outputs.get(\\\"audio_sample_rate\\\", outputs.get(\\\"sr\\\"))\\n'; "
    "assert old in text, 'expected audio sample-rate extraction was not found'; "
    "text = text.replace(old, new, 1); "
    "old = '                request_audio_payload = outputs[0] if len(outputs) == 1 else outputs\\n'; "
    "new = '                request_audio_payload = (\\n"
    "                    audio_payload\\n"
    "                    if audio_payload is not None\\n"
    "                    else (outputs[0] if len(outputs) == 1 else outputs)\\n"
    "                )\\n"
    "                multimodal_output = {\\\"audio\\\": request_audio_payload}\\n"
    "                if model_audio_sample_rate is not None:\\n"
    "                    multimodal_output[\\\"sr\\\"] = model_audio_sample_rate\\n"
    "                    multimodal_output[\\\"audio_sample_rate\\\"] = model_audio_sample_rate\\n'; "
    "assert old in text, 'expected audio payload block was not found'; "
    "text = text.replace(old, new, 1); "
    "old = '                        multimodal_output={\\\"audio\\\": request_audio_payload},\\n'; "
    "new = '                        multimodal_output=multimodal_output,\\n'; "
    "assert old in text, 'expected audio multimodal output block was not found'; "
    "p.write_text(text.replace(old, new, 1))\""
)

SERVING_SPEECH_PATCH = (
    "python3 -c \"from pathlib import Path; "
    "p = Path('/app/vllm-omni/vllm_omni/entrypoints/openai/serving_speech.py'); "
    "text = p.read_text(); "
    "old = '                prompt[\\\"_stream_audio_blocks\\\"] = True\\n\\n"
    "                generator = self._diffusion_engine.generate(\\n'; "
    "new = '                prompt[\\\"_stream_audio_blocks\\\"] = True\\n"
    "                prompt[\\\"_stream_output_enabled\\\"] = True\\n\\n"
    "                generator = self._diffusion_engine.generate(\\n'; "
    "assert old in text, 'expected pure diffusion stream block was not found'; "
    "p.write_text(text.replace(old, new, 1))\""
)


def _load_local_env() -> dict[str, str]:
    env_path = pathlib.Path(".env.local")
    if not env_path.exists():
        return {}

    values: dict[str, str] = {}
    for line in env_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _hf_secrets() -> list[modal.Secret]:
    """Use an HF token only when the caller supplied one."""
    secret_name = os.environ.get("MODAL_HF_SECRET_NAME")
    if secret_name:
        return [modal.Secret.from_name(secret_name)]

    local_env = _load_local_env()
    hf_token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or local_env.get("HF_TOKEN")
        or local_env.get("HUGGINGFACE_TOKEN")
    )
    if hf_token:
        return [
            modal.Secret.from_dict(
                {
                    "HF_TOKEN": hf_token,
                    "HUGGING_FACE_HUB_TOKEN": hf_token,
                }
            )
        ]
    return []


def _github_secrets() -> list[modal.Secret]:
    local_env = _load_local_env()
    token = (
        os.environ.get("GITHUB_TOKEN")
        or os.environ.get("GH_TOKEN")
        or local_env.get("GITHUB_TOKEN")
        or local_env.get("GH_TOKEN")
    )
    if not token:
        try:
            result = subprocess.run(
                ["gh", "auth", "token"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            result = None
        if result is not None and result.returncode == 0:
            token = result.stdout.strip()

    if not token:
        return []
    return [modal.Secret.from_dict({"GITHUB_TOKEN": token})]


image = (
    modal.Image.from_registry("vllm/vllm-openai:v0.19.0")
    .entrypoint([])
    .apt_install("curl", "ffmpeg", "git", "libsndfile1")
    .run_commands("ln -sf /usr/bin/python3 /usr/local/bin/python")
    .uv_pip_install("httpx>=0.27.0", "transformers>=5.3.0")
    .run_commands(
        "mkdir -p /app/vllm-omni",
        (
            f"if [ -n \"${{GITHUB_TOKEN:-}}\" ]; then "
            f"curl -fsSL -H \"Authorization: Bearer ${{GITHUB_TOKEN}}\" {REPO_ARCHIVE_URL} "
            "-o /tmp/vllm-omni.tar.gz; "
        f"else curl -fsSL {REPO_ARCHIVE_URL} -o /tmp/vllm-omni.tar.gz; fi"
        ),
        "tar -xzf /tmp/vllm-omni.tar.gz -C /app/vllm-omni --strip-components=1",
        "test -f /app/vllm-omni/setup.py",
        DIFFUSION_ENGINE_PATCH,
        DIFFUSION_AUDIO_DICT_PATCH,
        SERVING_SPEECH_PATCH,
        (
            "cd /app/vllm-omni && "
            "VLLM_OMNI_TARGET_DEVICE=cuda "
            "VLLM_OMNI_VERSION_OVERRIDE=0.0.0+modal "
            "uv pip install --python \"$(python3 -c 'import sys; print(sys.executable)')\" "
            "--no-cache-dir ."
        ),
        (
            "uv pip install --python \"$(python3 -c 'import sys; print(sys.executable)')\" "
            "--no-cache-dir 'transformers>=5.3.0'"
        ),
        (
            "python3 -c \"import importlib.metadata as m, re; "
            "version = m.version('transformers'); "
            "match = re.match(r'^(\\\\d+)\\\\.(\\\\d+)\\\\.(\\\\d+)', version); "
            "assert match and tuple(map(int, match.groups())) >= (5, 3, 0), version; "
            "print('preflight transformers', version)\""
        ),
        secrets=_github_secrets(),
    )
    .env(SERVER_ENV)
)

hf_cache_vol = modal.Volume.from_name("omnivoice-huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("omnivoice-vllm-cache", create_if_missing=True)

app = modal.App(APP_NAME)


def _wait_for_health(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
                if response.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001 - report the final startup failure.
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"OmniVoice health check did not pass within {timeout_s}s: {last_error}")


def _post_json(base_url: str, payload: dict, timeout_s: float = 300.0) -> bytes:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/v1/audio/speech",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:500]}") from exc


def _warmup(base_url: str) -> None:
    prompts = [
        "Short warmup.",
        "This is a medium length warmup sentence for the OmniVoice service.",
        "Long warmup sample. " * 40,
    ]
    for index, prompt in enumerate(prompts, start=1):
        start = time.perf_counter()
        audio = _post_json(
            base_url,
            {
                "model": MODEL_NAME,
                "input": prompt,
                "voice": "default",
                "response_format": "wav",
            },
            timeout_s=300.0,
        )
        elapsed = time.perf_counter() - start
        print(f"warmup {index}/{len(prompts)}: {len(audio)} bytes in {elapsed:.3f}s", flush=True)


def _runtime_preflight() -> None:
    import importlib.metadata as metadata
    import re
    import subprocess

    import torch
    from vllm_omni.diffusion.models.omnivoice.pipeline_omnivoice import OmniVoicePipeline

    version = metadata.version("transformers")
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", version)
    if not match or tuple(map(int, match.groups())) < (5, 3, 0):
        raise RuntimeError(f"transformers>=5.3.0 required, found {version}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in the Modal runtime")

    print(f"runtime preflight transformers {version}", flush=True)
    print(f"runtime preflight cuda {torch.cuda.get_device_name(0)}", flush=True)
    print(f"runtime preflight omnivoice pipeline {OmniVoicePipeline.__name__}", flush=True)

    subprocess.check_call(
        ["vllm", "serve", "--omni", "--step-execution", "--help"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
    print("runtime preflight vllm serve accepted --omni --step-execution", flush=True)


@app.function(
    image=image,
    gpu=GPU,
    secrets=_hf_secrets(),
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    timeout=30 * MINUTES,
    startup_timeout=30 * MINUTES,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
)
@modal.concurrent(max_inputs=MAX_CONCURRENT_INPUTS, target_inputs=TARGET_CONCURRENT_INPUTS)
@modal.web_server(port=VLLM_PORT, startup_timeout=30 * MINUTES)
def serve() -> None:
    import subprocess

    _runtime_preflight()

    cmd = [
        "vllm",
        "serve",
        MODEL_NAME,
        "--host",
        "0.0.0.0",
        "--port",
        str(VLLM_PORT),
        "--trust-remote-code",
        "--omni",
        "--step-execution",
    ]
    print("Starting OmniVoice:", " ".join(cmd), flush=True)
    subprocess.Popen(cmd)

    base_url = f"http://127.0.0.1:{VLLM_PORT}"
    _wait_for_health(base_url, timeout_s=20 * MINUTES)
    print("OmniVoice health check passed; running warmup.", flush=True)
    _warmup(base_url)
    print("OmniVoice warmup complete; server is ready.", flush=True)


@app.local_entrypoint()
def smoke(
    api_base: str | None = None,
    text: str = "Hello, how are you? This is OmniVoice running on Modal.",
    output: str = "outputs/modal_omnivoice_smoke.wav",
    stream: bool = True,
) -> None:
    """Run a health check and synthesize one WAV through Modal."""
    if api_base is None:
        api_base = serve.get_web_url()

    api_base = api_base.rstrip("/")
    print(f"Testing {api_base}")
    _wait_for_health(api_base, timeout_s=30 * MINUTES)

    payload = {
        "model": MODEL_NAME,
        "input": text,
        "voice": "default",
        "response_format": "wav",
        "stream": stream,
    }
    request = urllib.request.Request(
        f"{api_base}/v1/audio/speech",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    first_byte_s: float | None = None
    total_bytes = 0
    chunk_count = 0
    try:
        with urllib.request.urlopen(request, timeout=300) as response, output_path.open("wb") as f:
            while True:
                chunk = response.read(8 * 1024)
                if not chunk:
                    break
                if first_byte_s is None:
                    first_byte_s = time.perf_counter() - start
                f.write(chunk)
                total_bytes += len(chunk)
                chunk_count += 1
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body[:500]}") from exc

    total_s = time.perf_counter() - start
    with wave.open(str(output_path), "rb") as wav:
        frames = wav.getnframes()
        channels = wav.getnchannels()
        rate = wav.getframerate()
        sample_width = wav.getsampwidth()

    duration_s = frames / float(rate)
    if stream and output_path.stat().st_size >= 44:
        audio_bytes = output_path.stat().st_size - 44
        duration_s = audio_bytes / float(rate * channels * sample_width)

    first = first_byte_s if first_byte_s is not None else total_s
    print(
        "smoke ok: "
        f"{output_path} bytes={total_bytes} chunks={chunk_count} "
        f"first_byte={first:.3f}s total={total_s:.3f}s "
        f"audio={duration_s:.3f}s channels={channels} rate={rate}",
        flush=True,
    )
