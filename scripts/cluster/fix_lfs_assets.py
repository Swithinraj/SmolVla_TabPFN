"""The repository stores the simulator meshes/textures as Git LFS *pointer* files (no git-lfs on the cluster).
Replace every pointer with the real file, accepting a download only if its sha256 and size match the pointer.
Sources tried in order: the repo's GitHub LFS, the Hugging Face EnvHub repo the assets were copied from, MuJoCo Menagerie."""
import hashlib
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "challenge/sim/so101_tabletop"
UA = {"User-Agent": "python-urllib"}


def pointers():
    for p in BASE.rglob("*"):
        if p.is_file() and p.stat().st_size < 400:
            t = p.read_bytes()
            if t.startswith(b"version https://git-lfs.github.com/spec"):
                oid = re.search(rb"oid sha256:([0-9a-f]{64})", t).group(1).decode()
                size = int(re.search(rb"size (\d+)", t).group(1))
                yield p, oid, size


def fetch(url):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
            return r.read()
    except Exception as e:  # noqa: BLE001
        return None


def candidates(rel):
    gh = "https://media.githubusercontent.com/media/Swithinraj/SmolVla_TabPFN/main/challenge/sim/so101_tabletop/" + rel
    yield gh
    for repo in ("lerobot/so101-tabletop", "lerobot/so101_tabletop", "lerobot/so101-tabletop-mujoco"):
        for kind in ("", "spaces/", "datasets/"):
            yield f"https://huggingface.co/{kind}{repo}/resolve/main/{rel}"
    if rel.startswith("robotstudio_so101/"):
        yield "https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/" + rel
        yield "https://media.githubusercontent.com/media/google-deepmind/mujoco_menagerie/main/" + rel


bad = 0
for p, oid, size in pointers():
    rel = p.relative_to(BASE).as_posix()
    ok = False
    for url in candidates(rel):
        b = fetch(url)
        if b is not None and len(b) == size and hashlib.sha256(b).hexdigest() == oid:
            p.write_bytes(b)
            print("OK  ", rel, "<-", url, flush=True)
            ok = True
            break
    if not ok:
        bad += 1
        print("MISS", rel, size, oid, flush=True)
print("missing:", bad)
sys.exit(1 if bad else 0)
