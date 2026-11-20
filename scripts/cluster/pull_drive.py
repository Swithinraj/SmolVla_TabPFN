"""Download (part of) the shared Drive folder SmolVLA_TabPFN_SO101 (public link) with gdown into drive_pull/, then arrange it
as in the hand-over guide section 2. Re-running skips files already present.
  python scripts/cluster/pull_drive.py [--list] [--prefix checkpoints/5_catch_big,checkpoints/2_stage2] [--only checkpoints]"""
import argparse
import shutil
import sys
from pathlib import Path

import gdown

FOLDER = "18BWYkCl592-N_dSM5L2XeRnGBhfThkyl"
ROOT = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("--list", action="store_true", help="only list the remote files")
ap.add_argument("--only", default="", help="comma list of top-level folders to arrange (default: all)")
ap.add_argument("--wait-for", default="", help="comma list of remote paths to wait for (Drive upload in progress)")
ap.add_argument("--wait-hours", type=float, default=8.0)
ap.add_argument("--prefix", default="", help="comma list of remote path prefixes to download, e.g. checkpoints/5_catch_big")
a = ap.parse_args()
pull = ROOT / "drive_pull"
import time


def listing():
    fs = gdown.download_folder(id=FOLDER, output=str(pull), quiet=True, skip_download=True, retries=3)
    if fs is None:
        sys.exit("gdown returned nothing (permission / rate limit?)")
    # gdown prefixes every path with the Drive folder's own name (SmolVLA_TabPFN_SO101/...): strip it
    return [(f.id, f.path.split("/", 1)[1] if "/" in f.path else f.path) for f in fs]


def direct(fid, out):
    """Fallback: Drive's direct download endpoint (confirm=t skips the large-file virus-scan page)."""
    import requests
    url = f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"
    with requests.get(url, stream=True, timeout=120) as r:
        if r.status_code != 200 or "text/html" in r.headers.get("Content-Type", ""):
            return None
        tmp = out.with_name(out.name + ".part")
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
        tmp.replace(out)
    return str(out)


def fetch_one(fid, out, tries=4):
    """gdown, then the direct endpoint; on Drive's per-file rate limit wait and retry (1, 5, 15 min)."""
    for k in range(tries):
        try:
            r = gdown.download(id=fid, output=str(out), quiet=True, resume=True, retries=2)
            if r:
                return r
        except Exception as e:  # noqa: BLE001  (FileURLRetrievalError: rate limit / confirmation page)
            print("  gdown:", type(e).__name__, str(e)[:120], flush=True)
        try:
            r = direct(fid, out)
            if r:
                return r
        except Exception as e:  # noqa: BLE001
            print("  direct:", type(e).__name__, str(e)[:120], flush=True)
        if k < tries - 1:
            wait = (60, 300, 900)[min(k, 2)]
            print(f"  retry in {wait} s", flush=True)
            time.sleep(wait)
    return None


prefixes = [x.strip() for x in a.prefix.split(",") if x.strip()]
need = [x.strip() for x in a.wait_for.split(",") if x.strip()]
t0 = time.time()
while True:
    files = listing()
    have = {r for _, r in files}
    missing = [n for n in need if n not in have]
    if not missing:
        break
    if time.time() - t0 > a.wait_hours * 3600:
        sys.exit(f"gave up waiting for {missing}")
    print(time.strftime("%H:%M"), "waiting for Drive upload:", missing, flush=True)
    time.sleep(300)
sel = [(i, r) for i, r in files if not prefixes or any(r.startswith(px) for px in prefixes)]
if a.list:
    for _, r in sel:
        print(r)
    print(len(sel), "of", len(files), "files")
    sys.exit(0)
bad = []
for fid, rel in sel:
    out = pull / rel
    if out.exists() and out.stat().st_size > 0:
        continue
    out.parent.mkdir(parents=True, exist_ok=True)
    r = fetch_one(fid, out)
    print("got" if r else "FAILED", rel, flush=True)
    if not r:
        bad.append(rel)
print("selected", len(sel), "files; failed", len(bad), flush=True)
if bad:
    sys.exit(1)

LAYOUT = [("checkpoints/*", "checkpoints"), ("datasets/catch_v2_merged", "data/so101_catch_v2"),
          ("datasets/catch_v2_shards/A", "data/catch_v2_A"), ("datasets/catch_v2_shards/B", "data/catch_v2_B"),
          ("datasets/blocking_stage2_train", "data/so101_ball_train"),
          ("datasets/blocking_stage3_forecast_train", "data/so101_ball_fc_train"),
          ("datasets/blocking_heldout", "data/so101_ball_heldout"), ("evaluation/*", "outputs/drive_evaluation")]
only = set(filter(None, a.only.split(",")))
for src, dst in LAYOUT:
    if only and src.split("/")[0] not in only:
        continue
    srcs = list(pull.glob(src)) if "*" in src else [pull / src]
    for s in srcs:
        if not s.exists():
            print("missing on Drive:", s.relative_to(pull)); continue
        d = ROOT / dst / s.name if "*" in src else ROOT / dst
        d.parent.mkdir(parents=True, exist_ok=True)
        # copy, keep drive_pull as the pristine mirror; never rewrite a file that is already in place (running jobs read them)
        def copy_new(a_, b_):
            if not (Path(b_).exists() and Path(b_).stat().st_size == Path(a_).stat().st_size):
                shutil.copy2(a_, b_)
        shutil.copytree(s, d, dirs_exist_ok=True, copy_function=copy_new)
        print("arranged", s.relative_to(pull), "->", d.relative_to(ROOT))
