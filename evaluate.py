"""Evaluate CRF predictions: encode with predicted CRFs, compute actual VMAF, report stats."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
import utils
from compute_vmaf import compute_vmaf_for_pair
from encode_segments import encode_segment
from predict import predict_from_features_file

log = utils.setup_logging("evaluate")

TARGET_VMAFS = [80, 85, 90, 95]


def evaluate_predictions(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Encode segments with predicted CRFs and compute actual VMAF."""
    results = []

    for _, row in predictions_df.iterrows():
        video_name = row["video_name"]
        segment_name = row["segment_name"]
        predicted_crf = int(row["predicted_crf"])
        target_vmaf = row["target_vmaf"]

        seg_source_path = config.SEGMENTS_SOURCE_DIR / video_name / f"{segment_name}.mp4"
        if not seg_source_path.exists():
            log.warning(f"Source segment not found: {seg_source_path}")
            continue

        # Encode with predicted CRF
        encoded_path = encode_segment(seg_source_path, predicted_crf)
        if encoded_path is None:
            log.error(f"Failed to encode {segment_name} with CRF={predicted_crf}")
            continue

        # Compute VMAF
        vmaf_result = compute_vmaf_for_pair(encoded_path, seg_source_path)
        if vmaf_result is None:
            log.error(f"Failed to compute VMAF for {encoded_path.name}")
            continue

        actual_vmaf = vmaf_result["vmaf"]
        deviation = actual_vmaf - target_vmaf
        deviation_pct = abs(deviation / target_vmaf) * 100

        results.append({
            "video_name": video_name,
            "segment_name": segment_name,
            "target_vmaf": target_vmaf,
            "predicted_crf": predicted_crf,
            "actual_vmaf": actual_vmaf,
            "deviation": deviation,
            "deviation_pct": deviation_pct,
        })

    return pd.DataFrame(results)


def print_summary(results_df: pd.DataFrame):
    """Print summary statistics matching the paper's Table 6 format."""
    log.info("\n" + "=" * 70)
    log.info("EVALUATION RESULTS")
    log.info("=" * 70)

    for target in sorted(results_df["target_vmaf"].unique()):
        subset = results_df[results_df["target_vmaf"] == target]
        log.info(f"\nTarget VMAF = {target}")
        log.info(f"  Segments evaluated: {len(subset)}")
        log.info(f"  Mean actual VMAF:   {subset['actual_vmaf'].mean():.3f}")
        log.info(f"  Mean deviation:     {subset['deviation'].mean():.3f}")
        log.info(f"  Mean |deviation|:   {subset['deviation'].abs().mean():.3f}")
        log.info(f"  Mean deviation %%:   {subset['deviation_pct'].mean():.2f}%%")
        log.info(f"  Max |deviation|:    {subset['deviation'].abs().max():.3f}")

        # Per-video breakdown
        log.info(f"  {'Video':<20} {'Avg VMAF':>10} {'Diff':>10}")
        log.info(f"  {'-'*20} {'-'*10} {'-'*10}")
        for video, vdf in subset.groupby("video_name"):
            avg = vdf["actual_vmaf"].mean()
            diff = avg - target
            log.info(f"  {video:<20} {avg:>10.3f} {diff:>+10.3f}")

    # Overall
    log.info(f"\nOverall mean deviation %%: {results_df['deviation_pct'].mean():.2f}%%")


def plot_crf_boxplots(results_df: pd.DataFrame, out_dir: Path):
    """Plot boxplots of predicted CRF values per video for each target VMAF."""
    targets = sorted(results_df["target_vmaf"].unique())
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        subset = results_df[results_df["target_vmaf"] == target]
        videos = sorted(subset["video_name"].unique())
        data = [subset[subset["video_name"] == v]["predicted_crf"].values for v in videos]
        ax.boxplot(data, tick_labels=videos)
        ax.set_title(f"Target VMAF = {target}")
        ax.set_ylabel("Predicted CRF")
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(out_dir / "crf_boxplots.png", dpi=150)
    plt.close()
    log.info(f"Saved CRF boxplots to {out_dir / 'crf_boxplots.png'}")


def plot_vmaf_deviation(results_df: pd.DataFrame, out_dir: Path):
    """Plot VMAF deviation histogram per target VMAF."""
    targets = sorted(results_df["target_vmaf"].unique())
    n = len(targets)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, target in zip(axes, targets):
        subset = results_df[results_df["target_vmaf"] == target]
        ax.hist(subset["deviation"], bins=30, edgecolor="black", alpha=0.7)
        ax.axvline(x=0, color="red", linestyle="--")
        ax.set_title(f"Target VMAF = {target}")
        ax.set_xlabel("VMAF Deviation")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y")

    plt.tight_layout()
    plt.savefig(out_dir / "vmaf_deviation.png", dpi=150)
    plt.close()
    log.info(f"Saved VMAF deviation plots to {out_dir / 'vmaf_deviation.png'}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate CRF predictions")
    parser.add_argument("--features", type=str,
                        default=str(config.DATASET_DIR / "features.parquet"),
                        help="Features parquet file for test segments")
    parser.add_argument("--target-vmafs", type=float, nargs="+", default=TARGET_VMAFS,
                        help="Target VMAF values to evaluate")
    parser.add_argument("--workers", type=int, default=config.MAX_WORKERS)
    args = parser.parse_args()

    utils.ensure_dirs()
    features_csv = Path(args.features)

    all_results = []

    for target_vmaf in args.target_vmafs:
        log.info(f"\nPredicting CRFs for target VMAF = {target_vmaf}")
        predictions = predict_from_features_file(features_csv, target_vmaf)
        log.info(f"  Generated {len(predictions)} predictions")

        log.info(f"  Encoding and computing VMAF...")
        results = evaluate_predictions(predictions)
        all_results.append(results)
        log.info(f"  Evaluated {len(results)} segments")

    results_df = pd.concat(all_results, ignore_index=True)
    out_path = config.RESULTS_DIR / "evaluation_results.csv"
    results_df.to_csv(out_path, index=False)

    print_summary(results_df)
    plot_crf_boxplots(results_df, config.RESULTS_DIR)
    plot_vmaf_deviation(results_df, config.RESULTS_DIR)

    log.info(f"\nAll results saved to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
