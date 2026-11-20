"""Merge any number of recording shards into one LeRobot dataset (generalises aggregate_v2.py).
  python scripts/aggregate_shards.py --out data/so101_catch_nm data/catch_nm_A data/catch_nm_B ...
Shards without sidecar.json (unfinished) are skipped with a warning. Sidecar episode indices are renumbered."""
import argparse
import json
import shutil
from pathlib import Path

from lerobot.datasets.aggregate import aggregate_datasets

ap = argparse.ArgumentParser()
ap.add_argument("shards", nargs="+")
ap.add_argument("--out", required=True)
a = ap.parse_args()
shards = []
for s in map(Path, a.shards):
    if (s / "sidecar.json").exists():
        shards.append(s)
    else:
        print("WARNING: skipping unfinished shard", s)
assert shards, "no finished shards"
out = Path(a.out)
shutil.rmtree(out, ignore_errors=True)
aggregate_datasets([f"local/{s.name}" for s in shards], f"local/{out.name}", roots=shards, aggr_root=out)
side = []
for s in shards:
    for r in json.loads((s / "sidecar.json").read_text()):
        r["shard"] = s.name
        r["episode"] = len(side)
        side.append(r)
(out / "sidecar.json").write_text(json.dumps(side))
print("merged", [s.name for s in shards], "->", out, "episodes", len(side))
