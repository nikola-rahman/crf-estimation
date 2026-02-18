"""Extract features from low-resolution segments using ffprobe and siti-tools."""

import argparse
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

import config
import utils

log = utils.setup_logging("extract_features")


def extract_ffprobe_frames(segment_path: Path) -> list[dict]:
    """Run ffprobe with signalstats, entropy, scdet filters and return per-frame data."""
    cmd = [
        "ffprobe", "-f", "lavfi",
        "-i", f"movie={segment_path},entropy,scdet,signalstats",
        "-show_frames", "-select_streams", "v",
        "-of", "json",
    ]
    result = utils.run_cmd(cmd, timeout=120)
    if result.returncode != 0:
        log.error(f"ffprobe failed for {segment_path.name}: {result.stderr[-300:]}")
        return []

    data = json.loads(result.stdout)
    return data.get("frames", [])


def parse_ffprobe_features(frames: list[dict]) -> dict[str, list[float]]:
    """Parse per-frame features from ffprobe output.

    Returns dict with keys: Y, U, V, SAT, HUE, SY, SU, SV, scene_score
    Each value is a list of floats (one per frame).
    """
    features = {
        "Y": [], "U": [], "V": [],
        "SAT": [], "HUE": [],
        "SY": [], "SU": [], "SV": [],
        "scene_score": [],
    }

    for frame in frames:
        tags = frame.get("tags", {})
        # signalstats features
        features["Y"].append(float(tags.get("lavfi.signalstats.YAVG", 0)))
        features["U"].append(float(tags.get("lavfi.signalstats.UAVG", 0)))
        features["V"].append(float(tags.get("lavfi.signalstats.VAVG", 0)))
        features["SAT"].append(float(tags.get("lavfi.signalstats.SATAVG", 0)))
        features["HUE"].append(float(tags.get("lavfi.signalstats.HUEAVG", 0)))
        # entropy features (normalized)
        features["SY"].append(float(tags.get("lavfi.entropy.entropy.normal.Y", 0)))
        features["SU"].append(float(tags.get("lavfi.entropy.entropy.normal.U", 0)))
        features["SV"].append(float(tags.get("lavfi.entropy.entropy.normal.V", 0)))
        # scene change detection score
        features["scene_score"].append(float(tags.get("lavfi.scd.score", 0)))

    return features


def extract_ti(segment_path: Path) -> list[float]:
    """Extract Temporal Information (TI) per frame.

    TI = std(frame_n - frame_{n-1}) per ITU-T P.910.
    SI is excluded per paper (doesn't correlate across resolutions).
    Computed directly from raw grayscale frames to avoid siti-tools stride bugs.
    """
    # Get resolution
    probe = utils.run_cmd([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0", str(segment_path),
    ], timeout=10)
    if probe.returncode != 0:
        log.error(f"ffprobe failed for {segment_path.name}")
        return []
    w, h = [int(x) for x in probe.stdout.strip().split(",")]

    # Extract raw grayscale frames via ffmpeg (binary output)
    cmd = [
        "ffmpeg", "-i", str(segment_path),
        "-vf", "format=gray",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode != 0:
        log.error(f"ffmpeg raw extract failed for {segment_path.name}")
        return []

    frame_size = w * h
    raw = np.frombuffer(result.stdout, dtype=np.uint8)
    n_frames = len(raw) // frame_size
    if n_frames < 2:
        return [0.0]

    frames = raw[:n_frames * frame_size].reshape(n_frames, h, w).astype(np.float64)
    ti_values = [0.0]  # first frame has no TI
    for i in range(1, n_frames):
        diff = frames[i] - frames[i - 1]
        ti_values.append(float(np.std(diff)))
    return ti_values


def compute_segment_features(segment_path: Path) -> dict | None:
    """Compute all features for a single low-resolution segment.

    Returns a flat dict with feature names as keys.
    """
    # Extract ffprobe features
    frames = extract_ffprobe_frames(segment_path)
    if not frames:
        return None

    ff_features = parse_ffprobe_features(frames)

    # Extract TI from siti-tools
    ti_values = extract_ti(segment_path)

    # Get frame rate
    fps = utils.get_video_fps(segment_path)

    # Determine scene detection (binary: 1 if any scene change detected)
    scene_detected = 1 if any(s > 0.5 for s in ff_features["scene_score"]) else 0

    # Compute statistics for each feature
    result = {}
    feature_names = ["Y", "U", "V", "SAT", "HUE", "SY", "SU", "SV"]

    for feat_name in feature_names:
        values = np.array(ff_features[feat_name], dtype=np.float64)
        if len(values) == 0:
            values = np.array([0.0])
        result[f"{feat_name}_mean"] = np.mean(values)
        result[f"{feat_name}_std"] = np.std(values)
        result[f"{feat_name}_kurtosis"] = float(stats.kurtosis(values))
        result[f"{feat_name}_skewness"] = float(stats.skew(values))

    # TI statistics
    ti_arr = np.array(ti_values, dtype=np.float64) if ti_values else np.array([0.0])
    result["TI_mean"] = np.mean(ti_arr)
    result["TI_std"] = np.std(ti_arr)
    result["TI_kurtosis"] = float(stats.kurtosis(ti_arr))
    result["TI_skewness"] = float(stats.skew(ti_arr))

    # Metadata features
    result["scene_detection"] = scene_detected
    result["frame_rate"] = round(fps, 2)

    # Identification
    result["video_name"] = segment_path.parent.name
    result["segment_name"] = segment_path.stem

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract features from low-resolution segments")
    parser.add_argument("--workers", type=int, default=config.MAX_WORKERS)
    parser.add_argument("--video", type=str, default=None,
                        help="Process segments from a single video name only")
    args = parser.parse_args()

    utils.ensure_dirs()

    # Collect all 240p segments
    if args.video:
        seg_dirs = [config.SEGMENTS_LOWRES_DIR / args.video]
    else:
        seg_dirs = sorted(
            d for d in config.SEGMENTS_LOWRES_DIR.iterdir() if d.is_dir()
        )

    all_segments = []
    for d in seg_dirs:
        all_segments.extend(sorted(d.glob("*.mp4")))

    if not all_segments:
        log.error("No low-resolution segments found. Run prepare_segments.py first.")
        return

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(compute_segment_features, seg): seg
            for seg in all_segments
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Features", unit="seg"):
            seg = futures[future]
            try:
                feat = future.result()
                if feat is not None:
                    results.append(feat)
            except Exception as e:
                log.error(f"Error processing {seg.name}: {e}")

    df = pd.DataFrame(results)
    out_path = config.DATASET_DIR / "features.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"Saved {len(df)} feature rows to {out_path}")


if __name__ == "__main__":
    main()
