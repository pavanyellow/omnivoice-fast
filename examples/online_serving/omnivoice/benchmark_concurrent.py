"""Concurrent variant of benchmark_online.py.

Submits the same TrySalient/tts-test-set length-distribution sample to
/v1/audio/speech, but with N in flight at once. Sweeps a list of concurrency
levels and prints per-request latency stats alongside server-side throughput
(audio_sec produced per wall-clock second). With --stream, it requests
block-wise streaming audio and reports TTFA plus stream RTF.

Usage:
    python benchmark_concurrent.py --concurrencies 1,4,8,16
    python benchmark_concurrent.py --stream --concurrencies 1,2,4,8,16,32
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import soundfile as sf
from huggingface_hub import hf_hub_download


def select_length_distribution_sample(df, length_col, sample_count):
    df = df.sort_values([length_col, "dataset_index"]).reset_index(drop=True)
    if sample_count >= len(df):
        return df.copy().reset_index(drop=True)
    positions = np.linspace(0, len(df) - 1, sample_count).round().astype(int)
    positions = list(dict.fromkeys(int(p) for p in positions))
    if len(positions) < sample_count:
        used = set(positions)
        for pos in range(len(df)):
            if pos not in used:
                positions.append(pos)
                used.add(pos)
                if len(positions) == sample_count:
                    break
    return df.iloc[positions].copy().reset_index(drop=True)


def _decode_wav(content: bytes) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(io.BytesIO(content), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


async def synth_one(client, api_url, headers, model, text, language, sem, stream):
    payload = {
        "model": model,
        "input": text,
        "voice": "default",
        "response_format": "wav",
    }
    if stream:
        payload["stream"] = True
    if language:
        payload["language"] = language
    async with sem:
        t0 = time.perf_counter()
        if stream:
            first_sec = None
            chunk_count = 0
            chunks: list[bytes] = []
            async with client.stream(
                "POST", api_url, json=payload, headers=headers,
            ) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    text_body = body.decode("utf-8", errors="replace")
                    raise RuntimeError(f"HTTP {resp.status_code}: {text_body[:300]}")
                async for chunk in resp.aiter_bytes():
                    if not chunk:
                        continue
                    if first_sec is None:
                        first_sec = time.perf_counter() - t0
                    chunks.append(chunk)
                    chunk_count += 1
            wall = time.perf_counter() - t0
            content = b"".join(chunks)
            if first_sec is None:
                first_sec = wall
        else:
            resp = await client.post(api_url, json=payload, headers=headers)
            wall = time.perf_counter() - t0
            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
            content = resp.content
            first_sec = None
            chunk_count = 1

    audio, sr = _decode_wav(content)
    return {
        "audio": audio,
        "sr": sr,
        "wall_sec": wall,
        "ttfa_sec": first_sec,
        "chunk_count": chunk_count,
        "byte_count": len(content),
    }


async def run_concurrency(
    client, api_url, headers, model, sample, language, concurrency,
    jitter_ms_min, jitter_ms_max, rng, stream,
):
    sem = asyncio.Semaphore(concurrency)

    async def worker(idx, row):
        text = str(row["sentence"])
        try:
            output = await synth_one(
                client, api_url, headers, model, text, language, sem, stream,
            )
            audio = output["audio"]
            sr = output["sr"]
            audio_sec = len(audio) / sr
            return {
                "sample_order": idx,
                "dataset_index": int(row["dataset_index"]),
                "char_len": int(row["char_len"]),
                "wall_sec": float(output["wall_sec"]),
                "ttfa_sec": output["ttfa_sec"],
                "audio_sec": audio_sec,
                "rtf": (
                    float(output["wall_sec"] / audio_sec)
                    if audio_sec > 0 else float("inf")
                ),
                "chunk_count": int(output["chunk_count"]),
                "byte_count": int(output["byte_count"]),
                "status": "ok",
            }
        except Exception as exc:
            return {
                "sample_order": idx,
                "dataset_index": int(row["dataset_index"]),
                "char_len": int(row["char_len"]),
                "wall_sec": 0.0,
                "ttfa_sec": None,
                "audio_sec": 0.0,
                "rtf": float("inf"),
                "chunk_count": 0,
                "byte_count": 0,
                "status": "error",
                "error": str(exc),
            }

    # Stagger task creation with realistic inter-arrival jitter so the
    # server sees a sequence of requests rather than a thundering herd.
    # The semaphore still caps the in-flight concurrency.
    t_start = time.perf_counter()
    tasks = []
    for i, (_, row) in enumerate(sample.iterrows()):
        tasks.append(asyncio.create_task(worker(i, row)))
        if i < len(sample) - 1:
            delay_ms = rng.uniform(jitter_ms_min, jitter_ms_max)
            await asyncio.sleep(delay_ms / 1000.0)
    results = await asyncio.gather(*tasks)
    total_wall = time.perf_counter() - t_start
    return results, total_wall


def summarize(results, total_wall, concurrency, min_audio, max_audio):
    df = pd.DataFrame(results)
    in_range = df[
        (df["status"] == "ok")
        & (df["audio_sec"] >= min_audio)
        & (df["audio_sec"] <= max_audio)
    ].copy()
    n = len(in_range)
    ttfa = (
        in_range["ttfa_sec"].dropna()
        if n and "ttfa_sec" in in_range else pd.Series(dtype=float)
    )
    return {
        "concurrency": concurrency,
        "n_in_range": n,
        "errors": int((df["status"] == "error").sum()),
        "total_wall_sec": float(total_wall),
        "total_audio_sec": float(in_range["audio_sec"].sum()) if n else 0.0,
        "aggregate_rtf": (
            float(total_wall / in_range["audio_sec"].sum())
            if n and in_range["audio_sec"].sum() > 0 else None
        ),
        "throughput_audio_per_wall": (
            float(in_range["audio_sec"].sum() / total_wall) if n else 0.0
        ),
        "req_per_sec": float(len(df) / total_wall) if total_wall > 0 else 0.0,
        "ttfa_avg": float(ttfa.mean()) if len(ttfa) else None,
        "ttfa_p50": float(ttfa.median()) if len(ttfa) else None,
        "ttfa_p95": float(ttfa.quantile(0.95)) if len(ttfa) else None,
        "ttfa_max": float(ttfa.max()) if len(ttfa) else None,
        "wall_avg": float(in_range["wall_sec"].mean()) if n else None,
        "wall_p50": float(in_range["wall_sec"].median()) if n else None,
        "wall_p95": float(in_range["wall_sec"].quantile(0.95)) if n else None,
        "wall_max": float(in_range["wall_sec"].max()) if n else None,
        "rtf_mean": float(in_range["rtf"].mean()) if n else None,
        "rtf_p50": float(in_range["rtf"].median()) if n else None,
        "rtf_p95": float(in_range["rtf"].quantile(0.95)) if n else None,
        "rtf_max": float(in_range["rtf"].max()) if n else None,
        "chunk_count_avg": float(in_range["chunk_count"].mean()) if n else None,
    }


async def main_async(args):
    csv_path = hf_hub_download(
        repo_id=args.dataset, repo_type="dataset", filename=args.dataset_file,
    )
    df = pd.read_csv(csv_path).reset_index(names="dataset_index")
    df["char_len"] = df["char_len"].astype(int)
    sample = select_length_distribution_sample(
        df, "char_len", min(args.candidate_count, len(df)),
    )
    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    concurrencies = [int(c.strip()) for c in args.concurrencies.split(",")]
    limits = httpx.Limits(
        max_connections=max(concurrencies) * 2,
        max_keepalive_connections=max(concurrencies) * 2,
    )

    summaries = []
    all_results = []
    async with httpx.AsyncClient(timeout=args.http_timeout, limits=limits) as client:
        if args.warmup_iters > 0:
            sem = asyncio.Semaphore(1)
            logging.info(
                "Warmup %d iters (length-spread from sample)",
                args.warmup_iters,
            )
            # Warmup with prompts spread evenly across the sample's
            # length distribution. Sample is sorted by char_len ascending,
            # so np.linspace gives us short, medium, and long prompts in
            # one pass. This populates the cudagraph cache (when
            # COMPILE_MODE=reduce-overhead) for every length bucket the
            # timed sweep will hit, so the first sweep does not pay
            # cold-capture cost.
            n_sample = len(sample)
            warm_idx = np.linspace(
                0, n_sample - 1, args.warmup_iters,
            ).round().astype(int)
            for idx in warm_idx:
                row = sample.iloc[int(idx)]
                await synth_one(
                    client, api_url, headers, args.model,
                    str(row["sentence"]), args.language, sem, args.stream,
                )

        rng = random.Random(args.jitter_seed)
        for c in concurrencies:
            logging.info(
                "Running concurrency=%d (%d candidates, jitter=%d-%dms)",
                c, len(sample), args.jitter_ms_min, args.jitter_ms_max,
            )
            results, total_wall = await run_concurrency(
                client, api_url, headers, args.model, sample, args.language, c,
                args.jitter_ms_min, args.jitter_ms_max, rng, args.stream,
            )
            summary = summarize(
                results, total_wall, c, args.min_audio_sec, args.max_audio_sec,
            )
            summaries.append(summary)
            for result in results:
                result["concurrency"] = c
                all_results.append(result)
            logging.info(
                "c=%d  n=%d  total_wall=%.2fs  audio/wall=%.2f  req/s=%.2f  "
                "ttfa_p50=%.3fs  ttfa_p95=%.3fs  rtf_mean=%.3f  rtf_p95=%.3f",
                c, summary["n_in_range"], summary["total_wall_sec"],
                summary["throughput_audio_per_wall"], summary["req_per_sec"],
                summary["ttfa_p50"] or 0, summary["ttfa_p95"] or 0,
                summary["rtf_mean"] or 0, summary["rtf_p95"] or 0,
            )
    return summaries, all_results


def main():
    parser = argparse.ArgumentParser(
        description="Concurrent OmniVoice TTFA / throughput benchmark."
    )
    parser.add_argument("--api-base", default="http://localhost:8091")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--dataset", default="TrySalient/tts-test-set")
    parser.add_argument("--dataset_file", default="test_set.csv")
    parser.add_argument("--candidate_count", type=int, default=200)
    parser.add_argument("--min_audio_sec", type=float, default=2.0)
    parser.add_argument("--max_audio_sec", type=float, default=25.0)
    parser.add_argument("--warmup_iters", type=int, default=3)
    parser.add_argument("--language", default=None)
    parser.add_argument(
        "--stream", action="store_true",
        help="Request stream=True and measure time to first audio chunk",
    )
    parser.add_argument(
        "--concurrencies", default="1,4,8,16",
        help="Comma-separated concurrency levels to sweep",
    )
    parser.add_argument(
        "--jitter_ms_min", type=float, default=10.0,
        help="Minimum inter-request delay (ms)",
    )
    parser.add_argument(
        "--jitter_ms_max", type=float, default=50.0,
        help="Maximum inter-request delay (ms)",
    )
    parser.add_argument("--jitter_seed", type=int, default=1234)
    parser.add_argument("--http_timeout", type=float, default=600.0)
    parser.add_argument("--out_dir", default=None)
    args = parser.parse_args()

    fmt = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO, force=True)

    if args.out_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = Path("outputs") / f"vllm_omni_concurrent_{ts}"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries, all_results = asyncio.run(main_async(args))
    df = pd.DataFrame(summaries)
    df.to_csv(out_dir / "concurrency_sweep.csv", index=False)
    pd.DataFrame(all_results).to_csv(out_dir / "concurrency_runs.csv", index=False)
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summaries, f, indent=2)

    print("\nCONCURRENCY_SWEEP")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
