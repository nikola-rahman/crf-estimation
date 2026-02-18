"""Upload pipeline data to Hippius S3 storage."""

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

# Directories to upload, in order of importance
UPLOAD_DIRS = {
    "dataset": BASE_DIR / "data" / "dataset",
    "models": BASE_DIR / "models",
    "vmaf": BASE_DIR / "data" / "vmaf",
    "encoded": BASE_DIR / "data" / "encoded",
    "segments_source": BASE_DIR / "data" / "segments_source",
    "segments_lowres": BASE_DIR / "data" / "segments_lowres",
    "raw_videos": BASE_DIR / "data" / "raw_videos",
    "wandb_logs": BASE_DIR / "data" / "wandb_logs",
}


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


def collect_files(dirs: dict[str, Path]) -> list[tuple[str, Path]]:
    """Collect all files with their S3 keys."""
    files = []
    for label, local_dir in dirs.items():
        if not local_dir.exists():
            continue
        for f in sorted(local_dir.rglob("*")):
            if f.is_file():
                s3_key = str(f.relative_to(BASE_DIR))
                files.append((s3_key, f))
    return files


def main():
    parser = argparse.ArgumentParser(description="Upload data to Hippius S3")
    parser.add_argument("--bucket", default=os.environ.get("S3_BUCKET", "crf-estimation"))
    parser.add_argument("--dirs", nargs="+", choices=list(UPLOAD_DIRS.keys()),
                        default=list(UPLOAD_DIRS.keys()),
                        help="Which directories to upload (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Re-upload files that already exist in S3")
    parser.add_argument("--dry-run", action="store_true",
                        help="List files without uploading")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel upload threads")
    args = parser.parse_args()

    selected = {k: UPLOAD_DIRS[k] for k in args.dirs}
    files = collect_files(selected)
    print(f"Found {len(files)} files to upload")

    if args.dry_run:
        total_size = sum(f.stat().st_size for _, f in files)
        print(f"Total size: {total_size / (1024**3):.1f} GB")
        for s3_key, local_path in files[:20]:
            print(f"  {s3_key} ({local_path.stat().st_size / (1024**2):.1f} MB)")
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more")
        return

    client = get_client()

    # Create bucket if needed
    if not client.bucket_exists(args.bucket):
        client.make_bucket(args.bucket)
        print(f"Created bucket: {args.bucket}")

    existing = set()
    if not args.force:
        print("Checking existing files in S3...")
        for obj in client.list_objects(args.bucket, recursive=True):
            existing.add(obj.object_name)
        print(f"Found {len(existing)} existing files")

    to_upload = [(s3_key, local_path) for s3_key, local_path in files
                  if s3_key not in existing]
    skipped = len(files) - len(to_upload)

    def upload_one(item):
        s3_key, local_path = item
        file_size = local_path.stat().st_size
        for attempt in range(5):
            try:
                with open(local_path, "rb") as f:
                    client.put_object(args.bucket, s3_key, f, file_size)
                return
            except S3Error as e:
                if "SlowDown" in str(e) and attempt < 4:
                    time.sleep(2 ** attempt)
                else:
                    raise

    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(upload_one, item): item for item in to_upload}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Uploading", unit="file"):
            try:
                future.result()
            except Exception as e:
                s3_key = futures[future][0]
                tqdm.write(f"Failed: {s3_key}: {e}")
                failed += 1

    print(f"Done. Uploaded: {len(to_upload) - failed}, "
          f"skipped: {skipped}, failed: {failed}")


if __name__ == "__main__":
    main()
