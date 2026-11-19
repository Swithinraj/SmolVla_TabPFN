"""Offline imitation check on held-out SO-101 episodes: flow-matching loss and first-action MAE (per joint).

Usage: python evaluate_so101.py --dataset data\\svla_so101_pickplace --repo lerobot/svla_so101_pickplace
         --episodes 5,25,45 --rename up:camera1,side:camera2 --models name=path name2=path2 [--samples 12]
Not a closed-loop success measure.
"""
import argparse
import gc
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_LEROBOT_HOME", str(ROOT / ".cache" / "lerobot"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts.lerobot_train import _preprocess_dataset_batch


def measure(model_dir, dataset_root, repo, episodes, rename, indices):
    config = PreTrainedConfig.from_pretrained(model_dir)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = LeRobotDatasetMetadata(repo, root=dataset_root)
    delta = resolve_delta_timestamps(config, meta, rename)
    ds = LeRobotDataset(repo, root=dataset_root, episodes=episodes, delta_timestamps=delta,
                        video_backend="pyav", return_uint8=True)
    policy = SmolVLAPolicy.from_pretrained(model_dir, config=config, local_files_only=True)
    pre, post = make_pre_post_processors(config, pretrained_path=model_dir,
                                         preprocessor_overrides={"device_processor": {"device": config.device}})
    recs = []
    for i in indices:
        s = ds[i]
        batch = torch.utils.data.default_collate([s])
        batch = _preprocess_dataset_batch(batch, ds.meta.camera_keys, rename, pre)
        policy.reset()
        torch.manual_seed(1000 + i)
        with torch.inference_mode():
            loss, _ = policy(batch)
            torch.manual_seed(2000 + i)
            pred = post(policy.select_action(batch)).detach().cpu().flatten()
        act = s["action"][0].detach().cpu().flatten()
        err = torch.abs(pred - act)
        recs.append({"episode": int(s["episode_index"]), "frame": int(s["frame_index"]), "loss": float(loss),
                     "mae": float(err.mean()), "mae_per_joint": err.tolist()})
    out = {"model": str(model_dir), "n": len(recs), "mean_loss": float(np.mean([r["loss"] for r in recs])),
           "mean_mae": float(np.mean([r["mae"] for r in recs])),
           "mae_per_joint": np.mean([r["mae_per_joint"] for r in recs], axis=0).tolist(), "samples": recs}
    del policy, pre, post, ds
    gc.collect()
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--rename", required=True, help="src:dst,... camera suffix map")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root = (ROOT / a.dataset) if not Path(a.dataset).is_absolute() else Path(a.dataset)
    episodes = [int(x) for x in a.episodes.split(",")]
    rename = {f"observation.images.{k}": f"observation.images.{v}"
              for k, v in (p.split(":") for p in a.rename.split(","))}
    probe = LeRobotDataset(a.repo, root=root, episodes=episodes, video_backend="pyav")
    col = probe.hf_dataset.data.column("episode_index").to_numpy()
    indices = []
    for e in episodes:
        pos = np.flatnonzero(col == e)
        indices += [int(pos[int((len(pos) - 1) * (k + 1) / (a.samples + 1))]) for k in range(a.samples)]
    del probe
    report = {"episodes": episodes, "frames": len(indices), "results": {}}
    for spec in a.models:
        name, path = spec.split("=", 1)
        r = measure(Path(path).resolve(), root, a.repo, episodes, rename, indices)
        report["results"][name] = r
        print(f"{name}: loss {r['mean_loss']:.4f}  MAE {r['mean_mae']:.4f}  per-joint {np.round(r['mae_per_joint'], 3)}", flush=True)
    Path(a.out).write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
