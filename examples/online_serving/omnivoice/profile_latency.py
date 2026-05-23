#!/usr/bin/env python3
"""HTTP streaming latency profiler for OmniVoice.

The original script targets an Orpheus-specific `/v1/audio/speech/stream`
endpoint that returns raw PCM. This copy targets the OpenAI-compatible
`/v1/audio/speech` endpoint with `stream=true` and defaults to
`response_format=pcm`, preserving the original byte-count audio-duration math.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import struct
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np


SAMPLE_RATE = 24000
BITS_PER_SAMPLE = 16
CHANNELS = 1

DEFAULT_TEXTS = [
    "Hello, this is a short streaming latency test.",
    "The quick brown fox jumps over the lazy dog while the clock ticks steadily.",
    "Please read these numbers clearly: one, two, three, four, five, six, seven, eight.",
    "Streaming text to speech should deliver the first audio block quickly while the rest continues to generate.",
    (
        "Although the request is longer than usual, the benchmark should still "
        "measure time to first audio, total generation time, and real time factor."
    ),
]

_DURATION_LADDER_WORDS = (
    "Streaming speech begins with a quiet greeting before the sentence grows "
    "into a careful benchmark passage. The listener should hear clean pacing, "
    "stable pronunciation, and natural pauses while the server measures first "
    "audio, chunk cadence, and sustained delivery. Each longer request adds "
    "more descriptive detail about concurrent users, mixed duration batches, "
    "padding behavior, and the way shorter utterances can wait beside longer "
    "ones during diffusion generation. This paragraph keeps the vocabulary "
    "plain so differences in latency mostly come from length rather than "
    "unusual symbols, abbreviations, or pronunciation traps. The final part "
    "continues with extra clauses to create long audio without changing the "
    "style, allowing the benchmark to cover brief, medium, and near thirty "
    "second requests in one run."
).split()

_DURATION_LADDER_SHORT_TEXTS = [
    "Hi.",
    "Hello.",
    "A short test.",
    "This is a short streaming test.",
]

_DURATION_LADDER_WORD_COUNTS = [
    7, 9, 11, 13, 15, 17, 19, 21,
    23, 25, 27, 29, 31, 33, 35, 37,
    39, 41, 43, 45, 47, 49, 51, 53,
    55, 59, 64, 69,
]


@dataclass
class TTSMetrics:
    connection_time: float
    ttfa: Optional[float]
    streaming_duration: Optional[float]
    total_latency: float
    audio_duration: float
    chunk_count: int
    audio_data: bytes
    bytes_received: int
    success: bool
    text: str
    error: Optional[str] = None


def generate_wav_header(
    sample_rate: int = SAMPLE_RATE,
    bits_per_sample: int = BITS_PER_SAMPLE,
    channels: int = CHANNELS,
    data_size: int = 0,
) -> bytes:
    """Generate a WAV header for PCM audio data."""
    bytes_per_sample = bits_per_sample // 8
    block_align = bytes_per_sample * channels
    byte_rate = sample_rate * block_align
    file_size = 36 + data_size

    header = bytearray()
    header.extend(b"RIFF")
    header.extend(struct.pack("<I", file_size))
    header.extend(b"WAVE")
    header.extend(b"fmt ")
    header.extend(struct.pack("<I", 16))
    header.extend(struct.pack("<H", 1))
    header.extend(struct.pack("<H", channels))
    header.extend(struct.pack("<I", sample_rate))
    header.extend(struct.pack("<I", byte_rate))
    header.extend(struct.pack("<H", block_align))
    header.extend(struct.pack("<H", bits_per_sample))
    header.extend(b"data")
    header.extend(struct.pack("<I", data_size))
    return bytes(header)


def duration_ladder_texts() -> list[str]:
    texts = list(_DURATION_LADDER_SHORT_TEXTS)
    for count in _DURATION_LADDER_WORD_COUNTS:
        words = _DURATION_LADDER_WORDS[:count]
        texts.append(" ".join(words))
    return texts


def load_texts(path: str | None, override_text: str | None, duration_ladder: bool) -> list[str]:
    if override_text:
        return [override_text]
    if duration_ladder:
        return duration_ladder_texts()

    if path and Path(path).exists():
        texts: list[str] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    text = data.get("text", data.get("sentence", ""))
                except json.JSONDecodeError:
                    text = line
                if text:
                    texts.append(str(text))
        if texts:
            return texts

    return DEFAULT_TEXTS


def pcm_audio_duration(audio_data: bytes, response_format: str) -> float:
    if response_format == "wav" and audio_data.startswith(b"RIFF") and len(audio_data) >= 44:
        audio_data = audio_data[44:]
    bytes_per_second = SAMPLE_RATE * CHANNELS * (BITS_PER_SAMPLE // 8)
    return len(audio_data) / bytes_per_second if bytes_per_second > 0 else 0.0


async def tts_request(
    session: aiohttp.ClientSession,
    *,
    api_base: str,
    api_key: str,
    model: str,
    text: str,
    language: str | None,
    response_format: str,
    chunk_size: int,
    http_timeout: float,
) -> TTSMetrics:
    """Make a single streaming TTS request."""
    start_time = time.perf_counter()
    first_chunk_time = None
    last_chunk_time = None
    chunk_count = 0

    payload = {
        "model": model,
        "input": text,
        "voice": "default",
        "response_format": response_format,
        "stream": True,
    }
    if language:
        payload["language"] = language

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        timeout = aiohttp.ClientTimeout(total=http_timeout)
        async with session.post(
            f"{api_base.rstrip('/')}/v1/audio/speech",
            json=payload,
            headers=headers,
            timeout=timeout,
        ) as response:
            response_start = time.perf_counter()
            connection_time = response_start - start_time

            if response.status != 200:
                error_text = await response.text()
                return TTSMetrics(
                    connection_time=connection_time,
                    ttfa=None,
                    streaming_duration=None,
                    total_latency=time.perf_counter() - start_time,
                    audio_duration=0.0,
                    chunk_count=0,
                    audio_data=b"",
                    bytes_received=0,
                    success=False,
                    text=text,
                    error=f"HTTP {response.status}: {error_text[:500]}",
                )

            bytes_received = 0
            audio_chunks: list[bytes] = []
            async for chunk in response.content.iter_chunked(chunk_size):
                if not chunk:
                    continue
                now = time.perf_counter()
                if first_chunk_time is None:
                    first_chunk_time = now
                last_chunk_time = now
                chunk_count += 1
                audio_chunks.append(chunk)
                bytes_received += len(chunk)

            end_time = time.perf_counter()
            if bytes_received == 0:
                return TTSMetrics(
                    connection_time=connection_time,
                    ttfa=None,
                    streaming_duration=None,
                    total_latency=end_time - start_time,
                    audio_duration=0.0,
                    chunk_count=0,
                    audio_data=b"",
                    bytes_received=0,
                    success=False,
                    text=text,
                    error="No audio data received",
                )

            audio_data = b"".join(audio_chunks)
            ttfa = first_chunk_time - start_time if first_chunk_time is not None else None
            streaming_duration = (
                last_chunk_time - first_chunk_time
                if last_chunk_time is not None and first_chunk_time is not None
                else None
            )

            return TTSMetrics(
                connection_time=connection_time,
                ttfa=ttfa,
                streaming_duration=streaming_duration,
                total_latency=end_time - start_time,
                audio_duration=pcm_audio_duration(audio_data, response_format),
                chunk_count=chunk_count,
                audio_data=audio_data,
                bytes_received=bytes_received,
                success=True,
                text=text,
            )

    except asyncio.TimeoutError:
        return TTSMetrics(
            connection_time=0.0,
            ttfa=None,
            streaming_duration=None,
            total_latency=time.perf_counter() - start_time,
            audio_duration=0.0,
            chunk_count=0,
            audio_data=b"",
            bytes_received=0,
            success=False,
            text=text,
            error="Request timeout",
        )
    except Exception as exc:
        return TTSMetrics(
            connection_time=0.0,
            ttfa=None,
            streaming_duration=None,
            total_latency=time.perf_counter() - start_time,
            audio_duration=0.0,
            chunk_count=0,
            audio_data=b"",
            bytes_received=0,
            success=False,
            text=text,
            error=f"Request failed: {exc}",
        )


async def run_concurrent_requests(args: argparse.Namespace, text: str) -> tuple[list[TTSMetrics], float]:
    """Run one concurrent batch and return per-request metrics plus batch wall."""
    return await run_concurrent_texts(args, [text] * args.concurrency)


async def run_concurrent_texts(
    args: argparse.Namespace,
    request_texts: list[str],
) -> tuple[list[TTSMetrics], float]:
    """Run one concurrent batch with explicit per-request texts."""
    connector = aiohttp.TCPConnector(limit=max(args.concurrency * 2, 32))
    async with aiohttp.ClientSession(connector=connector) as session:
        t0 = time.perf_counter()
        tasks = [
            tts_request(
                session,
                api_base=args.api_base,
                api_key=args.api_key,
                model=args.model,
                text=request_text,
                language=args.language,
                response_format=args.response_format,
                chunk_size=args.chunk_size,
                http_timeout=args.http_timeout,
            )
            for request_text in request_texts
        ]
        results = await asyncio.gather(*tasks)
        batch_wall = time.perf_counter() - t0
    return results, batch_wall


def select_request_texts(
    texts: list[str],
    *,
    concurrency: int,
    run_index: int,
    mixed_texts: bool,
) -> list[str]:
    if not mixed_texts:
        return [random.choice(texts)] * concurrency
    if not texts:
        raise ValueError("No texts available")
    if concurrency == 1:
        return [texts[(run_index - 1) % len(texts)]]
    if len(texts) >= concurrency:
        selected = []
        for i in range(concurrency):
            bucket_start = i * len(texts) // concurrency
            bucket_end = (i + 1) * len(texts) // concurrency
            bucket = texts[bucket_start:bucket_end]
            offset = (run_index - 1) % len(bucket)
            if i % 2 == 1:
                offset = len(bucket) - 1 - offset
            selected.append(bucket[offset])
        return selected
    return [texts[(run_index - 1 + i) % len(texts)] for i in range(concurrency)]


def save_audio(audio_data: bytes, filename: Path, response_format: str) -> None:
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, "wb") as f:
        if response_format == "pcm":
            f.write(generate_wav_header(data_size=len(audio_data)))
        f.write(audio_data)


def p(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile))


def print_metrics(results: list[TTSMetrics], concurrency_level: int, batch_wall: float) -> None:
    print(f"\n=== Concurrency Level: {concurrency_level} ===")

    successful_results = [r for r in results if r.success]
    failed_count = len(results) - len(successful_results)
    if failed_count > 0:
        print(f"Failed requests: {failed_count}/{len(results)}")
        for i, result in enumerate(results):
            if not result.success:
                print(f"  Request {i + 1}: {result.error}")

    if not successful_results:
        print("No successful requests to analyze.")
        return

    ttfa_values = [r.ttfa for r in successful_results if r.ttfa is not None]
    total_latencies = [r.total_latency for r in successful_results]
    audio_durations = [r.audio_duration for r in successful_results]
    chunk_counts = [r.chunk_count for r in successful_results]
    streaming_durations = [
        r.streaming_duration for r in successful_results
        if r.streaming_duration is not None
    ]
    rtf_values = [
        total_lat / audio_dur
        for total_lat, audio_dur in zip(total_latencies, audio_durations)
        if audio_dur > 0
    ]
    stream_rtf_values = [
        (r.total_latency - r.ttfa) / r.audio_duration
        for r in successful_results
        if r.ttfa is not None and r.audio_duration > 0
    ]

    print(f"\nSuccessful requests: {len(successful_results)}/{len(results)}")
    print(f"Batch Wall Time: {batch_wall:.3f}s")

    if ttfa_values:
        print("\nTTFA (request start to first audio byte):")
        print(f"  Mean: {np.mean(ttfa_values):.3f}s")
        print(f"  Median: {np.median(ttfa_values):.3f}s")
        print(f"  Min: {np.min(ttfa_values):.3f}s")
        print(f"  Max: {np.max(ttfa_values):.3f}s")
        print(f"  StdDev: {np.std(ttfa_values):.3f}s")
        if len(ttfa_values) >= 2:
            print(f"  P95: {p(ttfa_values, 95):.3f}s")
            print(f"  P99: {p(ttfa_values, 99):.3f}s")

    print("\nTotal Generation Time:")
    print(f"  Mean: {np.mean(total_latencies):.3f}s")
    print(f"  Median: {np.median(total_latencies):.3f}s")
    print(f"  Min: {np.min(total_latencies):.3f}s")
    print(f"  Max: {np.max(total_latencies):.3f}s")

    if streaming_durations:
        print("\nStreaming Duration (after first byte):")
        print(f"  Mean: {np.mean(streaming_durations):.3f}s")

    print("\nAudio Characteristics:")
    print(f"  Sample Rate: {SAMPLE_RATE}Hz, {BITS_PER_SAMPLE}-bit, {CHANNELS} channel(s)")
    print(f"  Average Audio Duration: {np.mean(audio_durations):.2f}s")
    print(f"  Average Chunks: {np.mean(chunk_counts):.1f}")

    if rtf_values:
        print("\nReal-time Factor (total latency / audio):")
        print(f"  Mean: {np.mean(rtf_values):.3f}")
        print(f"  Median: {np.median(rtf_values):.3f}")
        print(f"  Min: {np.min(rtf_values):.3f}")
        print(f"  Max: {np.max(rtf_values):.3f}")
        if len(rtf_values) >= 2:
            print(f"  P95: {p(rtf_values, 95):.3f}")

    if stream_rtf_values:
        print("\nStream RTF ((final byte - first byte) / audio):")
        print(f"  Median: {np.median(stream_rtf_values):.3f}")
        print(f"  P95: {p(stream_rtf_values, 95):.3f}")


def aggregate_metrics(all_results: list[list[TTSMetrics]], batch_walls: list[float]) -> dict:
    all_successful: list[TTSMetrics] = []
    total_requests = 0
    for run_results in all_results:
        total_requests += len(run_results)
        all_successful.extend([r for r in run_results if r.success])

    if not all_successful:
        return {
            "success_count": 0,
            "total_requests": total_requests,
            "success_rate": 0.0,
        }

    ttfa_values = [r.ttfa for r in all_successful if r.ttfa is not None]
    total_latencies = [r.total_latency for r in all_successful]
    audio_durations = [r.audio_duration for r in all_successful]
    rtf_values = [
        latency / duration
        for latency, duration in zip(total_latencies, audio_durations)
        if duration > 0
    ]
    stream_rtf_values = [
        (r.total_latency - r.ttfa) / r.audio_duration
        for r in all_successful
        if r.ttfa is not None and r.audio_duration > 0
    ]
    total_audio = float(sum(audio_durations))
    active_wall = float(sum(batch_walls))

    return {
        "success_count": len(all_successful),
        "total_requests": total_requests,
        "success_rate": len(all_successful) / total_requests if total_requests else 0.0,
        "batch_wall_sum_sec": active_wall,
        "total_audio_sec": total_audio,
        "aggregate_rtf": active_wall / total_audio if total_audio > 0 else None,
        "audio_per_wall": total_audio / active_wall if active_wall > 0 else None,
        "ttfa_mean": float(np.mean(ttfa_values)) if ttfa_values else None,
        "ttfa_p50": p(ttfa_values, 50) if ttfa_values else None,
        "ttfa_p95": p(ttfa_values, 95) if ttfa_values else None,
        "ttfa_p99": p(ttfa_values, 99) if ttfa_values else None,
        "latency_mean": float(np.mean(total_latencies)),
        "latency_p50": p(total_latencies, 50),
        "latency_p95": p(total_latencies, 95),
        "rtf_mean": float(np.mean(rtf_values)) if rtf_values else None,
        "rtf_p50": p(rtf_values, 50) if rtf_values else None,
        "rtf_p95": p(rtf_values, 95) if rtf_values else None,
        "rtf_max": float(np.max(rtf_values)) if rtf_values else None,
        "stream_rtf_mean": float(np.mean(stream_rtf_values)) if stream_rtf_values else None,
        "stream_rtf_p50": p(stream_rtf_values, 50) if stream_rtf_values else None,
        "stream_rtf_p95": p(stream_rtf_values, 95) if stream_rtf_values else None,
        "stream_rtf_max": float(np.max(stream_rtf_values)) if stream_rtf_values else None,
    }


def write_csv(path: Path, all_results: list[list[TTSMetrics]]) -> None:
    rows = []
    for run_index, run_results in enumerate(all_results, start=1):
        for request_index, result in enumerate(run_results, start=1):
            rtf = (
                result.total_latency / result.audio_duration
                if result.success and result.audio_duration > 0
                else None
            )
            stream_rtf = (
                (result.total_latency - result.ttfa) / result.audio_duration
                if (
                    result.success
                    and result.ttfa is not None
                    and result.audio_duration > 0
                )
                else None
            )
            rows.append({
                "run": run_index,
                "request_index": request_index,
                "success": result.success,
                "error": result.error,
                "ttfa_sec": result.ttfa,
                "connection_time_sec": result.connection_time,
                "streaming_duration_sec": result.streaming_duration,
                "total_latency_sec": result.total_latency,
                "audio_duration_sec": result.audio_duration,
                "chunk_count": result.chunk_count,
                "bytes_received": result.bytes_received,
                "rtf": rtf,
                "stream_rtf": stream_rtf,
                "text": result.text,
            })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OmniVoice HTTP streaming latency profiler.")
    parser.add_argument("--api-base", default="http://localhost:8091")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--language", default=None)
    parser.add_argument("--response-format", choices=["pcm", "wav"], default="pcm")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--sleep-between-runs", type=float, default=1.0)
    parser.add_argument("--http-timeout", type=float, default=600.0)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--text", default=None)
    parser.add_argument("--text-file", default="hallucination_eval/alphanumeric.jsonl")
    parser.add_argument(
        "--duration-ladder",
        action="store_true",
        help="Use built-in short-to-long texts instead of --text-file.",
    )
    parser.add_argument(
        "--mixed-texts",
        action="store_true",
        help="Use different texts within each concurrent run.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default="outputs/orpheus_profile_latency")
    parser.add_argument("--save-audio", dest="save_audio", action="store_true", default=True)
    parser.add_argument("--no-save-audio", dest="save_audio", action="store_false")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    random.seed(args.seed)

    texts = load_texts(args.text_file, args.text, args.duration_ladder)
    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.output_dir) / f"benchmark_{session_timestamp}"
    session_dir.mkdir(parents=True, exist_ok=True)

    print("TTS HTTP Streaming Latency Profiler")
    print(f"Server: {args.api_base}")
    print("\nConfiguration:")
    print(f"  Endpoint: {args.api_base.rstrip('/')}/v1/audio/speech")
    print(f"  Model: {args.model}")
    print(f"  Response Format: {args.response_format}")
    print(f"  Concurrency Level: {args.concurrency}")
    print(f"  Number of Runs: {args.num_runs}")
    print(f"  Sleep Between Runs: {args.sleep_between_runs}s")
    print(f"  Loaded Texts: {len(texts)}")
    print(f"  Output Directory: {session_dir}")

    all_results: list[list[TTSMetrics]] = []
    batch_walls: list[float] = []
    start_time = time.perf_counter()

    for run_num in range(1, args.num_runs + 1):
        print(f"\n{'=' * 60}")
        print(f"RUN {run_num}/{args.num_runs} - Concurrency: {args.concurrency}")
        print(f"{'=' * 60}")

        request_texts = select_request_texts(
            texts,
            concurrency=args.concurrency,
            run_index=run_num,
            mixed_texts=args.mixed_texts,
        )
        if args.mixed_texts:
            print(
                "Using mixed texts: "
                f"{len(set(request_texts))} unique across {len(request_texts)} requests"
            )
        else:
            text = request_texts[0]
            print(f"Using text: {text[:80]}..." if len(text) > 80 else f"Using text: {text}")

        results, batch_wall = asyncio.run(run_concurrent_texts(args, request_texts))
        all_results.append(results)
        batch_walls.append(batch_wall)
        print_metrics(results, args.concurrency, batch_wall)

        saved_count = 0
        if args.save_audio:
            for i, result in enumerate(results):
                if result.success and result.audio_data:
                    filename = session_dir / f"run{run_num:02d}_req{i + 1}_c{args.concurrency}.wav"
                    save_audio(result.audio_data, filename, args.response_format)
                    saved_count += 1
            print(f"\nSaved {saved_count} audio files for run {run_num}")

        if run_num < args.num_runs and args.sleep_between_runs > 0:
            print(f"\nSleeping {args.sleep_between_runs}s before next run...")
            time.sleep(args.sleep_between_runs)

    summary = aggregate_metrics(all_results, batch_walls)
    print(f"\n{'=' * 80}")
    print(f"AGGREGATE RESULTS - Concurrency: {args.concurrency}, Runs: {args.num_runs}")
    print(f"{'=' * 80}")
    print(json.dumps(summary, indent=2))

    write_csv(session_dir / "runs.csv", all_results)
    with open(session_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    end_time = time.perf_counter()
    print(f"\nTotal time taken: {end_time - start_time:.2f} seconds")
    print(f"Results saved to: {session_dir}")


if __name__ == "__main__":
    main()
