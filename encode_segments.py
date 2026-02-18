"""Encode source segments with SVT-AV1 at each CRF value."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

import config
import utils

log = utils.setup_logging("encode_segments")


def encode_segment(seg_path: Path, crf: int) -> Path | None:
    """Encode a single segment with SVT-AV1 at the given CRF."""
    video_name = seg_path.parent.name
    out_dir = config.ENCODED_DIR / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    out_name = f"{seg_path.stem}_crf{crf}.mp4"
    out_path = out_dir / out_name

    if out_path.exists():
        return out_path

    fps = utils.get_video_fps(seg_path)
    keyint = int(fps * config.SEGMENT_DURATION)

    cmd = [
        "ffmpeg", "-y", "-i", str(seg_path),
        "-c:v", "libsvtav1",
        "-crf", str(crf),
        "-preset", str(config.SVT_PRESET),
        "-g", str(keyint),
        "-b:v", "0",
        "-an",
        str(out_path),
    ]
    result = utils.run_cmd(cmd, timeout=1800)
    if result.returncode != 0:
        log.error(f"Encode failed: {seg_path.name} CRF={crf}: {result.stderr[-300:]}")
        if out_path.exists():
            out_path.unlink()
        return None
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Encode source segments with SVT-AV1")
    parser.add_argument("--workers", type=int, default=config.MAX_WORKERS)
    parser.add_argument("--video", type=str, default=None,
                        help="Encode segments from a single video name only")
    parser.add_argument("--crf", type=int, default=None,
                        help="Encode with a single CRF value only")
    args = parser.parse_args()

    utils.ensure_dirs()

    # Collect 4K segments
    if args.video:
        seg_dirs = [config.SEGMENTS_SOURCE_DIR / args.video]
    else:
        seg_dirs = sorted(
            d for d in config.SEGMENTS_SOURCE_DIR.iterdir() if d.is_dir()
        )

    all_segments = []
    for d in seg_dirs:
        all_segments.extend(sorted(d.glob("*.mp4")))

    if not all_segments:
        log.error("No source segments found. Run prepare_segments.py first.")
        return

    crf_values = [args.crf] if args.crf is not None else config.CRF_VALUES

    jobs = [
        (seg, crf)
        for seg in all_segments
        for crf in crf_values
    ]

    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(encode_segment, seg, crf): (seg, crf)
            for seg, crf in jobs
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Encoding", unit="seg"):
            seg, crf = futures[future]
            try:
                if future.result() is None:
                    failed += 1
            except Exception as e:
                log.error(f"Exception encoding {seg.name} CRF={crf}: {e}")
                failed += 1

    if failed:
        log.warning(f"{failed}/{len(jobs)} encodes failed")


if __name__ == "__main__":
    main()
