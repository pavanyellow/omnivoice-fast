"""Benchmark vLLM-Omni's OmniVoice serving on TrySalient/tts-test-set.

Mirrors examples/benchmark_optimized.py in the standalone OmniVoice repo so
the two stacks can be compared on the same input distribution and metric
definitions (wall_sec, audio_sec, rtf, p95). The vLLM-Omni stack replaces the
Python OmniVoice library; engine-internal optimizations (torch.compile, SDPA
backend, prefix caching) are applied internally by vllm-omni and not
configured here.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from vllm.multimodal.media.audio import load_audio

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


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


def build_prompt(text, ref_audio_signal, ref_sr, ref_text, language):
    prompt = {"prompt": text}
    multi_modal_data = {}
    mm_processor_kwargs = {}
    if ref_audio_signal is not None:
        multi_modal_data["audio"] = (ref_audio_signal, ref_sr)
        mm_processor_kwargs["ref_text"] = ref_text or ""
        mm_processor_kwargs["sample_rate"] = ref_sr
    if language:
        mm_processor_kwargs["lang"] = language
    if multi_modal_data:
        prompt["multi_modal_data"] = multi_modal_data
    if mm_processor_kwargs:
        prompt["mm_processor_kwargs"] = mm_processor_kwargs
    return prompt


def synth_one(omni, text, ref_audio_signal, ref_sr, ref_text, language):
    prompt = build_prompt(text, ref_audio_signal, ref_sr, ref_text, language)
    outputs = list(
        omni.generate(prompt, sampling_params_list=[OmniDiffusionSamplingParams()])
    )
    if not outputs:
        raise RuntimeError("No outputs returned from vllm-omni")
    ro = outputs[0].request_output
    if ro is None:
        raise RuntimeError("No request_output on omni output")
    mm = getattr(ro, "multimodal_output", None)
    if not mm and ro.outputs:
        mm = getattr(ro.outputs[0], "multimodal_output", None)
    if not mm or "audio" not in mm:
        raise RuntimeError("No audio in multimodal output")
    audio = mm["audio"]
    sr = mm.get("sr", 24000)
    if not isinstance(audio, np.ndarray):
        audio = audio.cpu().numpy().squeeze()
    return audio.astype(np.float32), int(sr)


def main():
    parser = argparse.ArgumentParser(
        description="vLLM-Omni OmniVoice TTFA benchmark."
    )
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument(
        "--stage-config",
        default="vllm_omni/model_executor/stage_configs/omnivoice.yaml",
    )
    parser.add_argument("--dataset", default="TrySalient/tts-test-set")
    parser.add_argument("--dataset_file", default="test_set.csv")
    parser.add_argument("--ref_audio", default="ref.wav")
    parser.add_argument("--ref_text", required=True)
    parser.add_argument("--sample_count", type=int, default=50)
    parser.add_argument("--candidate_count", type=int, default=200)
    parser.add_argument("--min_audio_sec", type=float, default=2.0)
    parser.add_argument("--max_audio_sec", type=float, default=25.0)
    parser.add_argument("--warmup_iters", type=int, default=3)
    parser.add_argument("--language", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--save_wavs", action="store_true")
    args = parser.parse_args()

    fmt = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=fmt, level=logging.INFO, force=True)

    if args.out_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = Path("outputs") / f"vllm_omni_{ts}"
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

    logging.info("Loading vllm-omni Omni engine for %s", args.model)
    t0 = time.perf_counter()
    omni = Omni(
        model=args.model,
        stage_configs_path=args.stage_config,
        log_stats=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    load_sec = time.perf_counter() - t0
    logging.info("Loaded in %.2fs", load_sec)

    logging.info("Loading reference audio %s", args.ref_audio)
    ref_audio_signal, ref_sr = load_audio(args.ref_audio, sr=None)
    ref_audio_signal = ref_audio_signal.astype(np.float32)

    logging.info("Warming up %d iters", args.warmup_iters)
    warmup_walls = []
    for i in range(args.warmup_iters):
        t0 = time.perf_counter()
        synth_one(
            omni,
            f"Warm up call number {i + 1}.",
            ref_audio_signal,
            ref_sr,
            args.ref_text,
            args.language,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        warmup_walls.append(time.perf_counter() - t0)
    logging.info("Warmup walls (s): %s", [round(x, 3) for x in warmup_walls])

    rows = []
    in_range = 0
    for i, row in sample.iterrows():
        text = str(row["sentence"])
        t0 = time.perf_counter()
        try:
            audio, sr = synth_one(
                omni, text, ref_audio_signal, ref_sr,
                args.ref_text, args.language,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            wall = time.perf_counter() - t0
            audio_sec = len(audio) / sr
            rtf = wall / audio_sec if audio_sec > 0 else float("inf")
            status = "in_range"
            if audio_sec < args.min_audio_sec or audio_sec > args.max_audio_sec:
                status = "skipped_duration"
            wav_path = ""
            if args.save_wavs and status == "in_range":
                wav_path = str(
                    out_dir
                    / "wavs"
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
        "load_sec": load_sec,
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

    omni.close()


if __name__ == "__main__":
    main()
