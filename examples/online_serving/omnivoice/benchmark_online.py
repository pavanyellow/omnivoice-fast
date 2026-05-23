"""Benchmark vLLM-Omni's OmniVoice online server on TrySalient/tts-test-set.

HTTP variant of examples/offline_inference/omnivoice/benchmark.py — measures the
production /v1/audio/speech path. Note: the online endpoint currently only
supports auto-voice (no voice cloning), so this is not apples-to-apples with
the standalone OmniVoice voice-clone benchmark; it's the right table to read
for a pure server-side TTFA / RTF view of vllm-omni.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
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


def synth_one(client, api_url, headers, model, text, language):
    payload = {
        "model": model,
        "input": text,
        "voice": "default",
        "response_format": "wav",
    }
    if language:
        payload["language"] = language
    resp = client.post(api_url, json=payload, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:300]}")
    audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def main():
    parser = argparse.ArgumentParser(
        description="vLLM-Omni OmniVoice online TTFA benchmark."
    )
    parser.add_argument("--api-base", default="http://localhost:8091")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument("--dataset", default="TrySalient/tts-test-set")
    parser.add_argument("--dataset_file", default="test_set.csv")
    parser.add_argument("--sample_count", type=int, default=50)
    parser.add_argument("--candidate_count", type=int, default=200)
    parser.add_argument("--min_audio_sec", type=float, default=2.0)
    parser.add_argument("--max_audio_sec", type=float, default=25.0)
    parser.add_argument("--warmup_iters", type=int, default=3)
    parser.add_argument("--language", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--save_wavs", action="store_true")
    parser.add_argument("--http_timeout", type=float, default=300.0)
    args = parser.parse_args()

    fmt = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO, force=True)

    if args.out_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = Path("outputs") / f"vllm_omni_online_{ts}"
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_wavs:
        (out_dir / "wavs").mkdir(parents=True, exist_ok=True)

    csv_path = hf_hub_download(
        repo_id=args.dataset, repo_type="dataset", filename=args.dataset_file,
    )
    df = pd.read_csv(csv_path).reset_index(names="dataset_index")
    df["char_len"] = df["char_len"].astype(int)
    candidate_count = min(args.candidate_count, len(df))
    sample = select_length_distribution_sample(df, "char_len", candidate_count)

    api_url = f"{args.api_base}/v1/audio/speech"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {args.api_key}",
    }

    logging.info("Connecting to %s", api_url)
    with httpx.Client(timeout=args.http_timeout) as client:
        logging.info("Warming up %d iters", args.warmup_iters)
        warmup_walls = []
        for i in range(args.warmup_iters):
            t0 = time.perf_counter()
            synth_one(
                client, api_url, headers, args.model,
                f"Warm up call number {i + 1}.", args.language,
            )
            warmup_walls.append(time.perf_counter() - t0)
        logging.info("Warmup walls (s): %s", [round(x, 3) for x in warmup_walls])

        rows = []
        in_range = 0
        for i, row in sample.iterrows():
            text = str(row["sentence"])
            t0 = time.perf_counter()
            try:
                audio, sr = synth_one(
                    client, api_url, headers, args.model, text, args.language,
                )
                wall = time.perf_counter() - t0
                audio_sec = len(audio) / sr
                rtf = wall / audio_sec if audio_sec > 0 else float("inf")
                status = "in_range"
                if audio_sec < args.min_audio_sec or audio_sec > args.max_audio_sec:
                    status = "skipped_duration"
                wav_path = ""
                if args.save_wavs and status == "in_range":
                    wav_path = str(
                        out_dir / "wavs"
                        / f"row{int(row['dataset_index']):04d}_len{int(row['char_len'])}.wav"
                    )
                    sf.write(wav_path, audio, sr)
            except Exception:
                wall = time.perf_counter() - t0
                audio_sec = 0.0
                rtf = float("inf")
                status = "error"
                wav_path = ""
                logging.exception("Synthesis failed for row %s", row["dataset_index"])

            rows.append({
                "sample_order": int(i),
                "dataset_index": int(row["dataset_index"]),
                "char_len": int(row["char_len"]),
                "bucket": row.get("bucket", ""),
                "client": row.get("client", ""),
                "sentence": text,
                "wall_sec": wall,
                "audio_sec": audio_sec,
                "rtf": rtf,
                "status": status,
                "wav_path": wav_path,
            })
            if status == "in_range":
                in_range += 1
            logging.info(
                "Run %d/%d in_range=%d row=%d chars=%d wall=%.4fs "
                "audio=%.4fs rtf=%.5f status=%s",
                i + 1, len(sample), in_range,
                int(row["dataset_index"]), int(row["char_len"]),
                wall, audio_sec, rtf, status,
            )

    res = pd.DataFrame(rows)
    in_range_df = res[res["status"] == "in_range"].sort_values(
        ["audio_sec", "dataset_index"]
    )
    if len(in_range_df) > args.sample_count:
        positions = np.linspace(
            0, len(in_range_df) - 1, args.sample_count
        ).round().astype(int)
        positions = list(dict.fromkeys(int(p) for p in positions))
        if len(positions) < args.sample_count:
            used = set(positions)
            for pos in range(len(in_range_df)):
                if pos not in used:
                    positions.append(pos)
                    used.add(pos)
                    if len(positions) == args.sample_count:
                        break
        keep_indices = set(in_range_df.iloc[positions].index)
        res.loc[
            in_range_df.index.difference(keep_indices), "status"
        ] = "skipped_selection"
        res.loc[list(keep_indices), "status"] = "success"
    else:
        res.loc[in_range_df.index, "status"] = "success"

    ok = res[res["status"] == "success"].copy()
    if not ok.empty:
        ok["audio_bin"] = pd.qcut(ok["audio_sec"], q=5, duplicates="drop")
        ok["text_bin"] = pd.qcut(ok["char_len"], q=5, duplicates="drop")
        text_summary = ok.groupby("text_bin", observed=True).agg(
            n=("wall_sec", "size"),
            char_min=("char_len", "min"),
            char_max=("char_len", "max"),
            avg_audio_sec=("audio_sec", "mean"),
            avg_wall_sec=("wall_sec", "mean"),
            p95_wall_sec=("wall_sec", lambda s: s.quantile(0.95)),
            avg_rtf=("rtf", "mean"),
        ).reset_index()
        audio_summary = ok.groupby("audio_bin", observed=True).agg(
            n=("wall_sec", "size"),
            audio_min=("audio_sec", "min"),
            audio_max=("audio_sec", "max"),
            avg_char_len=("char_len", "mean"),
            avg_wall_sec=("wall_sec", "mean"),
            p95_wall_sec=("wall_sec", lambda s: s.quantile(0.95)),
            avg_rtf=("rtf", "mean"),
        ).reset_index()
        text_summary.to_csv(out_dir / "summary_by_text_len.csv", index=False)
        audio_summary.to_csv(out_dir / "summary_by_audio_len.csv", index=False)
    else:
        text_summary = audio_summary = pd.DataFrame()

    overall = {
        "n": int(len(ok)),
        "errors": int((res["status"] == "error").sum()),
        "skipped_duration": int((res["status"] == "skipped_duration").sum()),
        "skipped_selection": int((res["status"] == "skipped_selection").sum()),
        "total_wall_sec": float(ok["wall_sec"].sum()) if len(ok) else 0.0,
        "total_audio_sec": float(ok["audio_sec"].sum()) if len(ok) else 0.0,
        "aggregate_rtf": (
            float(ok["wall_sec"].sum() / ok["audio_sec"].sum())
            if len(ok) and ok["audio_sec"].sum() > 0 else None
        ),
        "wall_avg": float(ok["wall_sec"].mean()) if len(ok) else None,
        "wall_p50": float(ok["wall_sec"].median()) if len(ok) else None,
        "wall_p95": float(ok["wall_sec"].quantile(0.95)) if len(ok) else None,
        "wall_max": float(ok["wall_sec"].max()) if len(ok) else None,
        "audio_avg": float(ok["audio_sec"].mean()) if len(ok) else None,
        "audio_max": float(ok["audio_sec"].max()) if len(ok) else None,
        "rtf_p50": float(ok["rtf"].median()) if len(ok) else None,
        "rtf_p95": float(ok["rtf"].quantile(0.95)) if len(ok) else None,
    }
    print(json.dumps({"event": "overall", **overall}))

    metadata = {
        "args": vars(args),
        "warmup_walls": warmup_walls,
        "overall": overall,
        "rows": rows,
    }
    res.to_csv(out_dir / "runs.csv", index=False)
    with (out_dir / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    if not text_summary.empty:
        print("SUMMARY_BY_TEXT_LEN")
        print(text_summary.to_string(index=False))
    if not audio_summary.empty:
        print("SUMMARY_BY_AUDIO_LEN")
        print(audio_summary.to_string(index=False))


if __name__ == "__main__":
    main()
