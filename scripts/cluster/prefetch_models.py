"""Fill .cache/ once so training (HF_HUB_OFFLINE=1) and evaluation run without the hub:
SmolVLA base, its SmolVLM2-500M backbone, and the TabPFN v2 regressor weights (CPU fit triggers the download)."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("TABPFN_MODEL_CACHE_DIR", str(ROOT / ".cache" / "tabpfn"))
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ.setdefault("TABPFN_NO_BROWSER", "1")
from huggingface_hub import snapshot_download

for rid in ("lerobot/smolvla_base", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"):
    print("downloading", rid, "->", snapshot_download(rid), flush=True)

import numpy as np
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion

r = TabPFNRegressor.create_default_for_version(ModelVersion.V2, device="cpu", n_estimators=1)
X = np.random.default_rng(0).normal(size=(50, 5))
r.fit(X, X[:, 0])
print("tabpfn ok", r.predict(X[:3]), flush=True)
print("cache:", os.environ["TABPFN_MODEL_CACHE_DIR"], list(Path(os.environ["TABPFN_MODEL_CACHE_DIR"]).rglob("*"))[:5])
sys.exit(0)
