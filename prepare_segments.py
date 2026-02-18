"""Split raw videos into 2-second segments and downscale for feature extraction."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import config
import utils

log = utils.setup_logging("prepare_segments")


def split_video_to_segments(video_path: Path) -> list[Path]:
    """Split a video into lossless segments at native resolution."""
    video_name = video_path.stem
    out_dir = config.SEGMENTS_SOURCE_DIR / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_pattern = str(out_dir / f"{video_name}_%04d.mp4")
    seg_dur = config.SEGMENT_DURATION
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
        "-force_key_frames", f"expr:gte(t,n_forced*{seg_dur})",
        "-an",
        "-f", "segment",
        "-segment_time", str(seg_dur),
        "-reset_timestamps", "1",
        out_pattern,
    ]
    result = utils.run_cmd(cmd, timeout=7200)
    if result.returncode != 0:
        log.error(f"Failed to split {video_path}: {result.stderr[-500:]}")
        return []

    segments = sorted(out_dir.glob(f"{video_name}_*.mp4"))
    return segments


def downscale_segment(seg_source_path: Path) -> Path | None:
    """Downscale a segment to low resolution for feature extraction (lossless, bicubic)."""
    video_name = seg_source_path.parent.name
    out_dir = config.SEGMENTS_LOWRES_DIR / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / seg_source_path.name
    if out_path.exists():
        return out_path

    # scale to low resolution keeping aspect ratio, pad if needed
    vf = f"scale={config.DOWNSCALE_WIDTH}:{config.DOWNSCALE_HEIGHT}:flags=bicubic:force_original_aspect_ratio=decrease,pad={config.DOWNSCALE_WIDTH}:{config.DOWNSCALE_HEIGHT}:(ow-iw)/2:(oh-ih)/2"
    cmd = [
        "ffmpeg", "-y", "-i", str(seg_source_path),
        "-vf", vf,
        "-c:v", "libx264", "-crf", "0", "-preset", "ultrafast",
        "-an",
        str(out_path),
    ]
    result = utils.run_cmd(cmd, timeout=600)
    if result.returncode != 0:
        log.error(f"Failed to downscale {seg_source_path.name}: {result.stderr[-300:]}")
        return None
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Split and downscale videos")
    parser.add_argument("--workers", type=int, default=config.MAX_WORKERS,
                        help="Number of parallel workers")
    parser.add_argument("--video", type=str, default=None,
                        help="Process a single video file instead of all")
    args = parser.parse_args()

    utils.ensure_dirs()

    if args.video:
        videos = [Path(args.video)]
    else:
        videos = utils.list_videos()

    if not videos:
        log.error(f"No videos found in {config.RAW_VIDEOS_DIR}")
        return

    # Phase 1: Split all videos into source segments (parallel)
    all_segments = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(split_video_to_segments, v): v for v in videos}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Splitting", unit="video"):
            try:
                segs = future.result()
                all_segments.extend(segs)
            except Exception as e:
                v = futures[future]
                log.error(f"Exception splitting {v.name}: {e}")

    # Phase 2: Downscale all segments in parallel
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(downscale_segment, s): s for s in all_segments}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Downscaling", unit="seg"):
            try:
                if future.result() is None:
                    failed += 1
            except Exception as e:
                s = futures[future]
                log.error(f"Exception downscaling {s.name}: {e}")
                failed += 1

    log.info(f"Total: {len(all_segments)} segments from {len(videos)} videos"
             + (f" ({failed} downscale failures)" if failed else ""))


if __name__ == "__main__":
    main()
