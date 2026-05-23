"""Reproduce the omnivoice-fast TTFA table.

Launches each of the three modes in turn, measures time-to-first-audio on a
fixed ~10 s prompt, prints the comparison table. Run from the repo root:

    python benchmark.py
"""
from __future__ import annotations

import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parent
SERVE = REPO / "serve.sh"
LOG = REPO / "server.log"

PROMPT = (
    "Gollum carefully leads Frodo and Sam towards the heavily defended "
    "Black Gate, and recommends another route to them. Frodo and Sam "
    "are captured by Faramir's Rangers."
)
BASE = "http://localhost:8091"
MODEL = "k2-fsa/OmniVoice"
SR = 24000
SW = 2  # int16
N = 10
WARM = 2

MODES = [
    # (label, env-var dict for serve.sh, stream-the-response?)
    ("baseline (upstream vllm-omni)", {"BASELINE": "true"}, False),
    ("full-clip optimized",           {"BLOCKED_STREAMING": "false"}, False),
    ("blocked streaming",             {"BLOCKED_STREAMING": "true"},  True),
]


def kill_any_server() -> None:
    subprocess.run(
        ["pkill", "-9", "-f", "[/]vllm serve"],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(2)


def boot(env_overrides: dict[str, str], timeout: int = 240) -> subprocess.Popen:
    kill_any_server()
    LOG.write_text("")
    env = {**os.environ, **env_overrides}
    proc = subprocess.Popen(
        [str(SERVE)], env=env, stdout=LOG.open("w"), stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.stderr.write(LOG.read_text()[-2000:])
            raise RuntimeError(f"serve.sh exited early (rc={proc.returncode})")
        log = LOG.read_text()
        if "Application startup complete" in log:
            return proc
        time.sleep(2)
    proc.send_signal(signal.SIGKILL)
    raise TimeoutError("server did not come up in time")


def measure_once(client: httpx.Client, stream: bool) -> tuple[float, float]:
    """Return (ttfa_ms, audio_seconds)."""
    if stream:
        t0 = time.perf_counter()
        ttfa = None
        nbytes = 0
        with client.stream(
            "POST", f"{BASE}/v1/audio/speech",
            json={"model": MODEL, "input": PROMPT,
                  "response_format": "pcm", "stream": True},
        ) as r:
            r.raise_for_status()
            for chunk in r.iter_raw():
                if not chunk:
                    continue
                if ttfa is None:
                    ttfa = time.perf_counter() - t0
                nbytes += len(chunk)
        return ttfa * 1000.0, nbytes / SW / SR
    t0 = time.perf_counter()
    r = client.post(
        f"{BASE}/v1/audio/speech",
        json={"model": MODEL, "input": PROMPT, "response_format": "wav"},
    )
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1000.0, max(0, (len(r.content) - 44) / SW / SR)


def bench_mode(label: str, env_overrides: dict[str, str], stream: bool) -> dict:
    print(f"\n=== {label} ===")
    print(f"    booting with {env_overrides} …", flush=True)
    proc = boot(env_overrides)
    print(f"    ready (pid={proc.pid})", flush=True)
    with httpx.Client(timeout=180) as client:
        for _ in range(WARM):
            measure_once(client, stream)
        runs = [measure_once(client, stream) for _ in range(N)]
    ttfa = sorted(r[0] for r in runs)
    audio = statistics.mean(r[1] for r in runs)
    kill_any_server()
    p50 = statistics.median(ttfa)
    print(f"    TTFA p50 = {p50:.1f} ms   (audio mean = {audio:.2f} s, N={N})")
    return {
        "label": label,
        "ttfa_ms_p50": p50,
        "ttfa_ms_min": ttfa[0],
        "ttfa_ms_max": ttfa[-1],
        "audio_s_mean": audio,
        "n": N,
    }


def main() -> None:
    if not SERVE.exists():
        sys.exit(f"serve.sh not found at {SERVE}")
    try:
        results = [bench_mode(*m) for m in MODES]
    finally:
        kill_any_server()

    base = results[0]["ttfa_ms_p50"]

    print("\n=== Results ===\n")
    print(f"{'Mode':<32} {'TTFA p50':>10} {'Audio':>8} {'vs. baseline':>13}")
    print("-" * 67)
    for r in results:
        print(
            f"{r['label']:<32} "
            f"{r['ttfa_ms_p50']:>7.0f} ms "
            f"{r['audio_s_mean']:>5.2f} s "
            f"{base / r['ttfa_ms_p50']:>11.2f}×"
        )

    out = REPO / "benchmark_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
