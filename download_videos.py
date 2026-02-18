"""Download videos from URLs extracted from wandb log JSON files."""

import argparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

from tqdm import tqdm

import config
import utils

log = utils.setup_logging("download_videos")


def extract_urls(logs_dir: Path) -> list[str]:
    """Extract and deduplicate all payload URLs from wandb log JSON files."""
    all_urls = set()
    for json_file in sorted(logs_dir.glob("*.json")):
        with open(json_file) as f:
            content = f.read()
        for match in re.finditer(r"payload_urls: \[([^\]]+)\]", content):
            list_str = match.group(1)
            urls = re.findall(r"'(https://[^']+)'", list_str)
            for url in urls:
                # Unescape JSON unicode sequences (e.g. \u0026 -> &)
                url = url.encode().decode("unicode_escape")
                all_urls.add(url)
    return sorted(all_urls)


def download_one(url: str, out_dir: Path) -> str | None:
    """Download a single URL to out_dir. Returns filename on success."""
    # Extract UUID filename from URL path
    path_part = url.split("?")[0]
    filename = path_part.split("/")[-1]
    out_path = out_dir / filename

    if out_path.exists():
        return filename

    try:
        urlretrieve(url, out_path)
        return filename
    except Exception as e:
        log.error(f"Failed to download {filename}: {e}")
        if out_path.exists():
            out_path.unlink()
        return None


def main():
    parser = argparse.ArgumentParser(description="Download videos from wandb logs")
    parser.add_argument("--logs-dir", type=str, required=True,
                        help="Directory containing wandb log JSON files")
    parser.add_argument("--out-dir", type=str, default=str(config.RAW_VIDEOS_DIR),
                        help="Output directory for downloaded videos")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel downloads")
    args = parser.parse_args()

    utils.ensure_dirs()
    logs_dir = Path(args.logs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    urls = extract_urls(logs_dir)
    log.info(f"Found {len(urls)} unique URLs to download")

    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_one, url, out_dir): url for url in urls
        }
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc="Downloading", unit="file"):
            if future.result() is None:
                failed += 1

    if failed:
        log.warning(f"{failed}/{len(urls)} downloads failed")


if __name__ == "__main__":
    main()
