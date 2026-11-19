"""Merge the recording shards into data/so101_catch_v2 (one LeRobot dataset)."""
import json, shutil, sys
from pathlib import Path
from lerobot.datasets.aggregate import aggregate_datasets
root = Path("data")
shards = [s for s in ("A", "B") if (root / f"catch_v2_{s}" / "sidecar.json").exists()]
out = root / "so101_catch_v2"
shutil.rmtree(out, ignore_errors=True)
aggregate_datasets([f"local/catch_v2_{s}" for s in shards], "local/so101_catch_v2",
                   roots=[root / f"catch_v2_{s}" for s in shards], aggr_root=out)
side = []
for s in shards:
    side += json.loads((root / f"catch_v2_{s}" / "sidecar.json").read_text())
(out / "sidecar.json").write_text(json.dumps(side))
print("merged", shards, "->", out, "episodes", len(side))
