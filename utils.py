"""Shared utilities for CRF estimation pipeline."""

import json
import logging
import subprocess
import sys
from pathlib import Path

import config


def setup_logging(name: str, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s")
        )
        logger.addHandler(handler)
    return logger


def ensure_dirs():
    """Create all required directories if they don't exist."""
    for d in [
        config.RAW_VIDEOS_DIR,
        config.SEGMENTS_SOURCE_DIR,
        config.SEGMENTS_LOWRES_DIR,
        config.ENCODED_DIR,
        config.VMAF_DIR,
        config.DATASET_DIR,
        config.MODELS_DIR,
        config.RESULTS_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def run_cmd(cmd: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    """Run a subprocess command and return the result."""
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def get_video_fps(video_path: str | Path) -> float:
    """Get the frame rate of a video using ffprobe."""
    result = run_cmd([
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "json",
        str(video_path),
    ])
    data = json.loads(result.stdout)
    fps_str = data["streams"][0]["r_frame_rate"]
    num, den = fps_str.split("/")
    return float(num) / float(den)


def get_video_resolution(video_path: str | Path) -> tuple[int, int]:
    """Get (width, height) of a video using ffprobe."""
    result = run_cmd([
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(video_path),
    ])
    data = json.loads(result.stdout)
    stream = data["streams"][0]
    return int(stream["width"]), int(stream["height"])


def fps_to_onehot(fps: float) -> list[int]:
    """Convert a frame rate value to a one-hot encoded vector."""
    vec = [0] * config.NUM_FPS_CATEGORIES
    for i, cat_fps in enumerate(config.FRAME_RATE_CATEGORIES):
        if abs(fps - cat_fps) < config.FPS_TOLERANCE:
            vec[i] = 1
            return vec
    # If no match, find the closest category
    closest = min(range(len(config.FRAME_RATE_CATEGORIES)),
                  key=lambda i: abs(fps - config.FRAME_RATE_CATEGORIES[i]))
    vec[closest] = 1
    return vec


def list_videos(directory: Path = None) -> list[Path]:
    """List all video files in a directory."""
    if directory is None:
        directory = config.RAW_VIDEOS_DIR
    extensions = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".y4m", ".yuv", ".ts"}
    videos = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )
    return videos
