"""Download pipeline data from S3 storage."""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from minio import Minio
from minio.error import S3Error
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent

DOWNLOAD_DIRS = ["dataset", "models", "vmaf", "encoded",
                 "segments_source", "segments_lowres", "raw_videos", "wandb_logs"]


def get_client() -> Minio:
    endpoint = os.environ.get("S3_ENDPOINT")
    access_key = os.environ.get("S3_ACCESS_KEY")
    secret_key = os.environ.get("S3_SECRET_KEY")
    if not endpoint or not access_key or not secret_key:
        raise RuntimeError(
            "Set S3_ENDPOINT, S3_ACCESS_KEY, and S3_SECRET_KEY env vars. "
            "See .env.example"
        )
    return Minio(
        endpoint,
        access_key=access_key,
        secret_key=secret_key,
        secure=True,
    )


def s3_key_matches_dirs(key: str, dirs: list[str]) -> bool:
    """Check if an S3 key belongs to one of the selected directories."""
    prefixes = []
    for d in dirs:
        if d == "models":
            prefixes.append("models/")
        elif d == "dataset":
            prefixes.append("data/dataset/")
        else:
            prefixes.append(f"data/{d}/")
    return any(key.startswith(p) for p in prefixes)


def main():
    parser = argparse.ArgumentParser(description="Download data from S3")
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET", "crf-estimation"))
    parser.add_argument("--dirs", nargs="+", choices=DOWNLOAD_DIRS,
                        default=[d for d in DOWNLOAD_DIRS if d != "raw_videos"],
                        help="Which directories to download (default: all except raw_videos)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download files that already exist locally")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without downloading")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel download threads")
    args = parser.parse_args()

    client = get_client()

    print("Listing remote files...")
    all_objects = []
    for obj in client.list_objects(args.bucket, recursive=True):
        if s3_key_matches_dirs(obj.object_name, args.dirs):
            all_objects.append((obj.object_name, obj.size))
    print(f"Found {len(all_objects)} remote files")

    if args.dry_run:
        total_size = sum(size for _, size in all_objects)
        print(f"Total size: {total_size / (1024**3):.1f} GB")
        for key, size in all_objects[:20]:
            print(f"  {key} ({size / (1024**2):.1f} MB)")
        if len(all_objects) > 20:
            print(f"  ... and {len(all_objects) - 20} more")
        return

    to_download = []
    skipped = 0
    for key, size in all_objects:
        local_path = BASE_DIR / key
        if not args.force and local_path.exists() and local_path.stat().st_size == size:
            skipped += 1
            continue
        to_download.append((key, local_path))

    print(f"To download: {len(to_download)}, skipping: {skipped} existing")

    def download_one(item):
        key, local_path = item
        local_path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(5):
            try:
                client.fget_object(args.bucket, key, str(local_path))
                return
            except S3Error as e:
                if "SlowDown" in str(e) and attempt < 4:
                    time.sleep(2 ** attempt)
                else:
                    raise

    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(download_one, item): item for item in to_download}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Downloading", unit="file"):
            try:
                future.result()
            except Exception as e:
                key = futures[future][0]
                tqdm.write(f"Failed: {key}: {e}")
                failed += 1

    print(f"Done. Downloaded: {len(to_download) - failed}, "
          f"skipped: {skipped}, failed: {failed}")


if __name__ == "__main__":
    main()
