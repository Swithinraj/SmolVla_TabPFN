"""Download selected LeRobot LIBERO v3 episodes into this project.

The source metadata is kept intact so LeRobot can select episodes with
``--dataset.episodes``. Only data/video files containing selected episodes are
downloaded. This script never writes to the user-wide Hugging Face cache.
"""

import argparse
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq


REPO = "lerobot/libero"
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "libero"
BASE = "https://huggingface.co/datasets/lerobot/libero/resolve/main/"
MIN_FREE = 100_000_000_000


def fetch(path: str) -> None:
    target = DATASET / path
    if target.is_file() and target.stat().st_size:
        print(f"kept {path} ({target.stat().st_size:,} bytes)")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".partial")
    request = urllib.request.Request(BASE + path, headers={"User-Agent": "smolvla-enhance/1.0"})
    with urllib.request.urlopen(request, timeout=90) as source, temp.open("wb") as sink:
        expected = int(source.headers.get("Content-Length", 0))
        if shutil.disk_usage(ROOT).free - expected < MIN_FREE:
            raise RuntimeError(f"Downloading {path} would leave less than 100 GB free")
        shutil.copyfileobj(source, sink, length=1024 * 1024)
    temp.replace(target)
    print(f"saved {path} ({target.stat().st_size:,} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8, help="Number of episodes from the start")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent file downloads")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")

    for path in ("README.md", "meta/info.json", "meta/stats.json", "meta/tasks.parquet", "meta/episodes/chunk-000/file-000.parquet"):
        fetch(path)
    episodes = pq.read_table(DATASET / "meta/episodes/chunk-000/file-000.parquet").to_pandas()
    print("metadata columns:", list(episodes.columns))
    if args.metadata_only:
        print(episodes.head(min(args.episodes, 8)).to_string(index=False))
        return
    selected = episodes[episodes["episode_index"].isin(range(args.episodes))]
    if len(selected) != args.episodes:
        raise RuntimeError(f"Only {len(selected)} requested episodes found")
    info = json.loads((DATASET / "meta/info.json").read_text(encoding="utf-8"))
    files = set()
    for row in selected.to_dict("records"):
        files.add(info["data_path"].format(chunk_index=row["data/chunk_index"], file_index=row["data/file_index"]))
        for video_key in ("observation.images.image", "observation.images.image2"):
            files.add(info["video_path"].format(video_key=video_key, chunk_index=row[f"videos/{video_key}/chunk_index"], file_index=row[f"videos/{video_key}/file_index"]))
    if args.workers < 1 or args.workers > 8:
        parser.error("--workers must be between 1 and 8")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, path): path for path in sorted(files)}
        for future in as_completed(futures):
            future.result()
    manifest = {"source": REPO, "root": str(DATASET), "episodes": list(range(args.episodes)), "files": sorted(files)}
    (DATASET / "subset.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Ready: {args.episodes} episodes, {len(files)} data/video files")


if __name__ == "__main__":
    main()
