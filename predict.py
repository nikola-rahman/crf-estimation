"""Inference: predict CRF for video segments given a target VMAF."""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import config
import utils
from extract_features import compute_segment_features
from model import CRFPredictor

log = utils.setup_logging("predict")


def load_model_and_scaler():
    """Load trained model and scaler from disk."""
    # Load scaler
    with open(config.DATASET_DIR / "scaler.pkl", "rb") as f:
        scaler_data = pickle.load(f)

    # Load model
    checkpoint = torch.load(config.MODELS_DIR / "best_model.pt", weights_only=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CRFPredictor(
        input_dim=checkpoint["input_dim"],
        num_classes=checkpoint["num_classes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, scaler_data, device


def prepare_input(features: dict, target_vmaf: float, scaler_data: dict) -> np.ndarray:
    """Transform raw features into model input vector."""
    feature_scaler = scaler_data["feature_scaler"]
    vmaf_scaler = scaler_data["vmaf_scaler"]
    feature_columns = scaler_data["feature_columns"]
    fps_onehot_cols = scaler_data["fps_onehot_cols"]

    # Build stat feature vector in correct order
    stat_values = [features.get(col, 0.0) for col in feature_columns]
    stat_array = np.array([stat_values], dtype=np.float64)
    stat_normalized = feature_scaler.transform(stat_array)[0]

    # Scene detection (binary)
    scene_det = features.get("scene_detection", 0)

    # Frame rate one-hot
    fps = features.get("frame_rate", 24.0)
    fps_onehot = utils.fps_to_onehot(fps)

    # Target VMAF (normalized with vmaf scaler)
    vmaf_norm = vmaf_scaler.transform(np.array([[target_vmaf]]))[0, 0]

    # Assemble: stat_features + scene_detection + fps_onehot + vmaf_norm
    input_vec = np.concatenate([
        stat_normalized,
        [scene_det],
        fps_onehot,
        [vmaf_norm],
    ])
    return input_vec


def predict_crf_for_segment(
    model: torch.nn.Module,
    features: dict,
    target_vmaf: float,
    scaler_data: dict,
    device: torch.device,
) -> int:
    """Predict the CRF class for a single segment."""
    input_vec = prepare_input(features, target_vmaf, scaler_data)
    x = torch.tensor(input_vec, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        predicted_class = logits.argmax(dim=1).item()

    return config.INDEX_TO_CRF[predicted_class]


def predict_for_video(video_path: Path, target_vmaf: float) -> list[dict]:
    """Full inference pipeline for a video: split, downscale, extract, predict."""
    from prepare_segments import split_video_to_segments, downscale_segment

    model, scaler_data, device = load_model_and_scaler()

    log.info(f"Processing {video_path.name} for target VMAF={target_vmaf}")

    # Split into source segments
    segments_source = split_video_to_segments(video_path)
    log.info(f"  Split into {len(segments_source)} segments")

    predictions = []
    for seg_src in segments_source:
        # Downscale to low resolution
        seg_lowres = downscale_segment(seg_src)
        if seg_lowres is None:
            log.error(f"  Failed to downscale {seg_src.name}")
            continue

        # Extract features
        features = compute_segment_features(seg_lowres)
        if features is None:
            log.error(f"  Failed to extract features for {seg_lowres.name}")
            continue

        # Predict CRF
        crf = predict_crf_for_segment(model, features, target_vmaf, scaler_data, device)
        predictions.append({
            "segment": seg_src.name,
            "segment_path": str(seg_src),
            "predicted_crf": crf,
            "target_vmaf": target_vmaf,
        })
        log.info(f"  {seg_src.name}: CRF={crf}")

    return predictions


def prepare_batch_input(df: pd.DataFrame, target_vmaf: float, scaler_data: dict) -> np.ndarray:
    """Transform a dataframe of features into a batch input matrix."""
    feature_scaler = scaler_data["feature_scaler"]
    vmaf_scaler = scaler_data["vmaf_scaler"]
    feature_columns = scaler_data["feature_columns"]

    # Stat features — vectorized scaling
    stat_matrix = df[feature_columns].fillna(0.0).values.astype(np.float64)
    stat_normalized = feature_scaler.transform(stat_matrix)

    # Scene detection (binary column)
    scene_det = df["scene_detection"].values.reshape(-1, 1)

    # Frame rate one-hot — vectorized
    fps_values = df["frame_rate"].values
    fps_matrix = np.zeros((len(df), config.NUM_FPS_CATEGORIES), dtype=np.float64)
    for i, cat in enumerate(config.FRAME_RATE_CATEGORIES):
        fps_matrix[:, i] = (np.abs(fps_values - cat) < config.FPS_TOLERANCE).astype(np.float64)
    # Assign closest for rows with no match
    no_match = fps_matrix.sum(axis=1) == 0
    if no_match.any():
        for row_idx in np.where(no_match)[0]:
            closest = int(np.argmin(np.abs(fps_values[row_idx] - np.array(config.FRAME_RATE_CATEGORIES))))
            fps_matrix[row_idx, closest] = 1.0

    # Target VMAF — single value broadcast to all rows
    vmaf_norm = vmaf_scaler.transform(np.array([[target_vmaf]]))[0, 0]
    vmaf_col = np.full((len(df), 1), vmaf_norm)

    return np.hstack([stat_normalized, scene_det, fps_matrix, vmaf_col])


def predict_from_features_file(features_path: Path, target_vmaf: float) -> pd.DataFrame:
    """Predict CRF for segments using pre-computed features (batch inference)."""
    model, scaler_data, device = load_model_and_scaler()

    df = pd.read_parquet(features_path)
    log.info(f"Predicting CRF for {len(df)} segments, target VMAF={target_vmaf}")

    batch_input = prepare_batch_input(df, target_vmaf, scaler_data)
    X = torch.tensor(batch_input, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(X)
        predicted_classes = logits.argmax(dim=1).cpu().numpy()

    predicted_crfs = [config.INDEX_TO_CRF[c] for c in predicted_classes]

    return pd.DataFrame({
        "video_name": df["video_name"].values,
        "segment_name": df["segment_name"].values,
        "predicted_crf": predicted_crfs,
        "target_vmaf": target_vmaf,
    })


def main():
    parser = argparse.ArgumentParser(description="Predict CRF for target VMAF")
    parser.add_argument("--video", type=str, default=None,
                        help="Path to a video file for full inference pipeline")
    parser.add_argument("--features", type=str, default=None,
                        help="Path to pre-computed features parquet file")
    parser.add_argument("--target-vmaf", type=float, required=True,
                        help="Target VMAF quality value")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path for predictions")
    args = parser.parse_args()

    if args.video:
        predictions = predict_for_video(Path(args.video), args.target_vmaf)
        df = pd.DataFrame(predictions)
    elif args.features:
        df = predict_from_features_file(Path(args.features), args.target_vmaf)
    else:
        parser.error("Provide either --video or --features")
        return

    out_path = args.output or str(config.RESULTS_DIR / "predictions.csv")
    df.to_csv(out_path, index=False)
    log.info(f"Saved {len(df)} predictions to {out_path}")


if __name__ == "__main__":
    main()
