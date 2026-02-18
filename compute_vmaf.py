"""Compute VMAF for each encoded segment against the source reference."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config
import utils

log = utils.setup_logging("compute_vmaf")


def compute_vmaf_for_pair(encoded_path: Path, reference_path: Path) -> dict | None:
    """Compute VMAF between encoded and reference segment.

    Returns dict with video_name, segment_name, crf, vmaf.
    """
    video_name = encoded_path.parent.name
    # Parse CRF from filename: {video}_{seg_id}_crf{value}.mp4
    stem = encoded_path.stem  # e.g. "myvideo_0001_crf27"
    parts = stem.rsplit("_crf", 1)
    if len(parts) != 2:
        log.error(f"Cannot parse CRF from filename: {encoded_path.name}")
        return None
    segment_name = parts[0]  # e.g. "myvideo_0001"
    crf = int(parts[1])

    # VMAF output file
    vmaf_out_dir = config.VMAF_DIR / video_name
    vmaf_out_dir.mkdir(parents=True, exist_ok=True)
    vmaf_json = vmaf_out_dir / f"{encoded_path.stem}_vmaf.json"

    if not vmaf_json.exists():
        cmd = [
            "ffmpeg", "-y",
            "-i", str(encoded_path),
            "-i", str(reference_path),
            "-lavfi", f"libvmaf=n_threads=4:log_fmt=json:log_path={vmaf_json}",
            "-f", "null", "-",
        ]
        result = utils.run_cmd(cmd, timeout=600)
        if result.returncode != 0:
            log.error(f"VMAF failed: {encoded_path.name}: {result.stderr[-300:]}")
            return None

    # Parse VMAF score
    try:
        with open(vmaf_json) as f:
            vmaf_data = json.load(f)
        vmaf_score = vmaf_data["pooled_metrics"]["vmaf"]["mean"]
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        log.error(f"Cannot parse VMAF from {vmaf_json}: {e}")
        return None

    return {
        "video_name": video_name,
        "segment_name": segment_name,
        "crf": crf,
        "vmaf": vmaf_score,
    }


def main():
    parser = argparse.ArgumentParser(description="Compute VMAF for encoded segments")
    parser.add_argument("--workers", type=int, default=config.MAX_WORKERS)
    parser.add_argument("--video", type=str, default=None,
                        help="Process a single video name only")
    args = parser.parse_args()

    utils.ensure_dirs()

    # Collect encoded segments and match with references
    if args.video:
        enc_dirs = [config.ENCODED_DIR / args.video]
    else:
        enc_dirs = sorted(
            d for d in config.ENCODED_DIR.iterdir() if d.is_dir()
        )

    jobs = []
    for enc_dir in enc_dirs:
        video_name = enc_dir.name
        ref_dir = config.SEGMENTS_SOURCE_DIR / video_name
        if not ref_dir.exists():
            log.warning(f"Reference dir missing for {video_name}, skipping")
            continue

        for encoded_path in sorted(enc_dir.glob("*.mp4")):
            # Derive reference path: strip _crfXX from filename
            stem = encoded_path.stem
            parts = stem.rsplit("_crf", 1)
            if len(parts) != 2:
                continue
            ref_name = parts[0] + ".mp4"
            ref_path = ref_dir / ref_name
            if not ref_path.exists():
                log.warning(f"Reference not found: {ref_path}")
                continue
            jobs.append((encoded_path, ref_path))

    if not jobs:
        log.error("No encoded segments found. Run encode_segments.py first.")
        return

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(compute_vmaf_for_pair, enc, ref): enc
            for enc, ref in jobs
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="VMAF", unit="seg"):
            try:
                r = future.result()
                if r is not None:
                    results.append(r)
            except Exception as e:
                enc = futures[future]
                log.error(f"Exception for {enc.name}: {e}")

    df = pd.DataFrame(results)
    out_path = config.DATASET_DIR / "vmaf_scores.csv"
    df.to_csv(out_path, index=False)
    log.info(f"Saved {len(df)} VMAF scores to {out_path}")


if __name__ == "__main__":
    main()
