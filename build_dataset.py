"""Merge features + VMAF + CRF into normalized training/validation datasets."""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

import config
import utils

log = utils.setup_logging("build_dataset")

STAT_FEATURES = [
    f"{feat}_{stat}"
    for feat in config.FRAME_FEATURES
    for stat in config.FEATURE_STATS
]


def load_and_merge():
    """Load features (parquet) and vmaf_scores (csv) and merge them."""
    features_path = config.DATASET_DIR / "features.parquet"
    vmaf_path = config.DATASET_DIR / "vmaf_scores.csv"

    features_df = pd.read_parquet(features_path)
    vmaf_df = pd.read_csv(vmaf_path)

    log.info(f"Features: {len(features_df)} segments")
    log.info(f"VMAF scores: {len(vmaf_df)} rows")

    # Merge: each feature row joins with multiple VMAF rows (one per CRF)
    merged = vmaf_df.merge(features_df, on=["video_name", "segment_name"], how="inner")
    log.info(f"Merged: {len(merged)} rows")
    return merged


def build_training_data(df: pd.DataFrame):
    """Normalize features, one-hot encode fps, split train/val, save."""

    # Cap VMAF at configured threshold
    before = len(df)
    df = df[df["vmaf"] < config.VMAF_CAP].copy()
    log.info(f"After capping VMAF < {config.VMAF_CAP}: {len(df)} rows (dropped {before - len(df)})")

    # Map CRF to class index
    df["crf_class"] = df["crf"].map(config.CRF_TO_INDEX)
    # Drop rows with CRF values not in our set
    df = df.dropna(subset=["crf_class"])
    df["crf_class"] = df["crf_class"].astype(int)

    # ── Standardize numerical features ────────────────────────────────────
    scaler = StandardScaler()
    df[STAT_FEATURES] = scaler.fit_transform(df[STAT_FEATURES])

    # Also fit a scaler for VMAF (separate, since at inference we need to
    # normalize the target VMAF input the same way)
    vmaf_scaler = StandardScaler()
    df["vmaf_norm"] = vmaf_scaler.fit_transform(df[["vmaf"]])

    # ── One-hot encode frame rate ─────────────────────────────────────────
    fps_onehot_cols = [f"fps_{cat}" for cat in config.FRAME_RATE_CATEGORIES]
    for i, cat in enumerate(config.FRAME_RATE_CATEGORIES):
        df[fps_onehot_cols[i]] = (
            (df["frame_rate"] - cat).abs() < config.FPS_TOLERANCE
        ).astype(int)

    # If no category matched for a row, assign closest
    fps_sum = df[fps_onehot_cols].sum(axis=1)
    no_match = fps_sum == 0
    if no_match.any():
        log.warning(f"{no_match.sum()} rows had no FPS category match, assigning closest")
        for idx in df[no_match].index:
            closest = min(
                range(len(config.FRAME_RATE_CATEGORIES)),
                key=lambda i: abs(df.loc[idx, "frame_rate"] - config.FRAME_RATE_CATEGORIES[i]),
            )
            df.loc[idx, fps_onehot_cols[closest]] = 1

    # ── Assemble input feature columns ────────────────────────────────────
    input_cols = STAT_FEATURES + ["scene_detection"] + fps_onehot_cols + ["vmaf_norm"]
    log.info(f"Input features: {len(input_cols)} columns")

    # ── Train/validation split (grouped by video to avoid leakage) ────────
    gss = GroupShuffleSplit(n_splits=1, train_size=config.TRAIN_SPLIT, random_state=42)
    groups = df["video_name"]
    train_idx, val_idx = next(gss.split(df, groups=groups))

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    log.info(f"Train: {len(train_df)} rows from {train_df['video_name'].nunique()} videos")
    log.info(f"Val:   {len(val_df)} rows from {val_df['video_name'].nunique()} videos")

    # ── Save ──────────────────────────────────────────────────────────────
    # Save only the columns needed for training
    save_cols = input_cols + ["crf_class", "video_name", "segment_name", "crf", "vmaf"]

    train_df[save_cols].to_parquet(config.DATASET_DIR / "train.parquet")
    val_df[save_cols].to_parquet(config.DATASET_DIR / "val.parquet")

    # Save scalers for inference
    scaler_data = {
        "feature_scaler": scaler,
        "feature_columns": STAT_FEATURES,
        "vmaf_scaler": vmaf_scaler,
        "input_columns": input_cols,
        "fps_onehot_cols": fps_onehot_cols,
    }
    with open(config.DATASET_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler_data, f)

    log.info(f"Saved train.parquet, val.parquet, and scaler.pkl to {config.DATASET_DIR}")


def main():
    df = load_and_merge()
    build_training_data(df)


if __name__ == "__main__":
    main()
