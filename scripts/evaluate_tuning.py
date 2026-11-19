"""Compare base and locally tuned SmolVLA on held-out LIBERO frames.

Episodes 3 and 5 were excluded from the local fine-tuning pass. This is an
offline imitation check, not a LIBERO rollout or success-rate measurement.
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


RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}


def measure(model_dir: Path, sample_indices: list[int], episodes: list[int]) -> dict:
    config = PreTrainedConfig.from_pretrained(model_dir)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_root = ROOT / "data" / "libero"
    meta = LeRobotDatasetMetadata("lerobot/libero", root=dataset_root)
    delta_timestamps = resolve_delta_timestamps(config, meta, RENAME_MAP)
    dataset = LeRobotDataset(
        "lerobot/libero", root=dataset_root, episodes=episodes,
        delta_timestamps=delta_timestamps, video_backend="pyav", return_uint8=True,
    )
    policy = SmolVLAPolicy.from_pretrained(model_dir, config=config, local_files_only=True)
    preprocessor, postprocessor = make_pre_post_processors(
        config, pretrained_path=model_dir,
        preprocessor_overrides={"device_processor": {"device": config.device}},
    )
    records = []
    for index in sample_indices:
        sample = dataset[index]
        batch = torch.utils.data.default_collate([sample])
        batch = _preprocess_dataset_batch(batch, dataset.meta.camera_keys, RENAME_MAP, preprocessor)
        policy.reset()
        torch.manual_seed(1000 + index)
        with torch.inference_mode():
            loss, _ = policy(batch)
            torch.manual_seed(2000 + index)
            predicted = postprocessor(policy.select_action(batch)).detach().cpu().flatten()
        actual = sample["action"][0].detach().cpu().flatten()
        records.append({
            "episode": int(sample["episode_index"]),
            "frame": int(sample["frame_index"]),
            "loss": float(loss.item()),
            "action_mae": float(torch.mean(torch.abs(predicted - actual)).item()),
        })
        print(f"{model_dir.name}: episode {records[-1]['episode']} frame {records[-1]['frame']} loss={records[-1]['loss']:.4f} MAE={records[-1]['action_mae']:.4f}", flush=True)
    result = {
        "model": str(model_dir), "device": config.device, "samples": records,
        "mean_loss": sum(r["loss"] for r in records) / len(records),
        "mean_action_mae": sum(r["action_mae"] for r in records) / len(records),
    }
    del policy, preprocessor, postprocessor, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned", type=Path, required=True, help="Path to trained pretrained_model directory")
    parser.add_argument("--samples-per-episode", type=int, default=4)
    parser.add_argument("--baseline-report", type=Path, help="Reuse the original-model measurements from a matching report")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "models" / "smolvla_libero")
    parser.add_argument("--test-episodes", type=str, default="3,5")
    args = parser.parse_args()
    if args.samples_per_episode < 1:
        parser.error("--samples-per-episode must be positive")
    episodes = [int(part.strip()) for part in args.test_episodes.split(",")]
    if not episodes or len(set(episodes)) != len(episodes):
        parser.error("--test-episodes must list distinct episode numbers")
    meta = LeRobotDatasetMetadata("lerobot/libero", root=ROOT / "data" / "libero")
    dataset = LeRobotDataset("lerobot/libero", root=ROOT / "data" / "libero", episodes=episodes, video_backend="pyav")
    episode_column = dataset.hf_dataset.data.column("episode_index").to_numpy()
    frame_column = dataset.hf_dataset.data.column("frame_index").to_numpy()
    selected = []
    for episode in episodes:
        positions = np.flatnonzero(episode_column == episode)
        selected.extend(
            int(positions[int((len(positions) - 1) * (i + 1) / (args.samples_per_episode + 1))])
            for i in range(args.samples_per_episode)
        )
    del dataset, meta
    print("Dataset indices:", selected, flush=True)
    if args.baseline_report:
        previous = json.loads(args.baseline_report.read_text(encoding="utf-8"))
        base = previous["baseline"]
        expected = [(int(episode_column[i]), int(frame_column[i])) for i in selected]
        actual = [(record["episode"], record["frame"]) for record in base["samples"]]
        if actual != expected:
            raise ValueError("Baseline report uses different frames")
        print("Reused matching baseline report:", args.baseline_report, flush=True)
    else:
        base = measure(args.baseline_dir.resolve(), selected, episodes)
    tuned = measure(args.tuned.resolve(), selected, episodes)
    report = {"note": f"Offline imitation check on episodes {episodes}, {len(selected)} fixed frames. Held out from the local training run only; check the baseline provenance before claiming generalization.", "baseline": base, "tuned": tuned}
    report["loss_change_pct"] = 100 * (tuned["mean_loss"] / base["mean_loss"] - 1)
    report["action_mae_change_pct"] = 100 * (tuned["mean_action_mae"] / base["mean_action_mae"] - 1)
    path = args.tuned.resolve().parents[2] / f"evaluation_{len(selected)}_{args.tuned.resolve().parent.name}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Mean loss: {base['mean_loss']:.4f} -> {tuned['mean_loss']:.4f} ({report['loss_change_pct']:+.1f}%)")
    print(f"Mean first-action MAE: {base['mean_action_mae']:.4f} -> {tuned['mean_action_mae']:.4f} ({report['action_mae_change_pct']:+.1f}%)")
    print("Report:", path)


if __name__ == "__main__":
    main()
