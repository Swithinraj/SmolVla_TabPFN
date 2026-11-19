"""Export offline, frame-aligned SmolVLA predictions for the two test episodes."""

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_LEROBOT_HOME", str(ROOT / ".cache/lerobot"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.scripts.lerobot_train import _preprocess_dataset_batch


MODEL = ROOT / "outputs/task7_base_aug_20261116_1748/checkpoints/024356/pretrained_model"
DATASET_ROOT = ROOT / "data/libero"
RENAME_MAP = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames per episode for a smoke check")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "viewer/predictions")
    args = parser.parse_args()
    split = json.loads((DATASET_ROOT / "task7_split.json").read_text(encoding="utf-8"))
    episodes = split["test_episodes"]
    cfg = PreTrainedConfig.from_pretrained(MODEL)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    meta = LeRobotDatasetMetadata("lerobot/libero", root=DATASET_ROOT)
    delta_timestamps = resolve_delta_timestamps(cfg, meta, RENAME_MAP)
    dataset = LeRobotDataset(
        "lerobot/libero", root=DATASET_ROOT, episodes=episodes,
        delta_timestamps=delta_timestamps, video_backend="pyav", return_uint8=True,
    )
    policy = SmolVLAPolicy.from_pretrained(MODEL, config=cfg, local_files_only=True).eval()
    preprocessor, postprocessor = make_pre_post_processors(
        cfg, pretrained_path=MODEL,
        preprocessor_overrides={"device_processor": {"device": cfg.device}},
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected = {episode: 0 for episode in episodes}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        episode = int(sample["episode_index"])
        frame = int(sample["frame_index"])
        if args.max_frames and expected[episode] >= args.max_frames:
            continue
        if frame != expected[episode]:
            raise RuntimeError(f"Episode {episode}: expected frame {expected[episode]}, got {frame}")
        batch = torch.utils.data.default_collate([sample])
        batch = _preprocess_dataset_batch(batch, dataset.meta.camera_keys, RENAME_MAP, preprocessor)
        policy.reset()
        torch.manual_seed(100000 + episode * 1000 + frame)
        with torch.inference_mode():
            loss, _ = policy(batch)
            torch.manual_seed(200000 + episode * 1000 + frame)
            prediction = postprocessor(policy.select_action(batch)).detach().cpu().flatten()
        ground_truth = sample["action"][0].detach().cpu().flatten()
        if prediction.numel() != 7 or ground_truth.numel() != 7:
            raise RuntimeError("Expected seven action dimensions")
        record = {
            "frame": frame,
            "pred": [float(value) for value in prediction],
            "loss": float(loss.item()),
            "mae": float(torch.mean(torch.abs(prediction - ground_truth)).item()),
        }
        path = args.output_dir / f"episode-{episode}.json"
        if frame == 0:
            records = []
        records.append(record)
        expected[episode] += 1
        if expected[episode] % 25 == 0:
            print(f"Episode {episode}: {expected[episode]} frames", flush=True)
        if expected[episode] == (args.max_frames or int(meta.episodes["length"][episode])):
            payload = {
                "model": "task7_base_aug_20261116_1748/checkpoints/024356",
                "episode": episode,
                "method": "Independent one-step prediction from each recorded observation; policy reset at every frame. Fixed per-frame random seeds for flow-matching loss and action sampling. No image augmentation at inference.",
                "frames": records,
                "mean_loss": sum(item["loss"] for item in records) / len(records),
                "mean_mae": sum(item["mae"] for item in records) / len(records),
            }
            temporary = path.with_suffix(".partial.json")
            temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            temporary.replace(path)
            print(f"Saved {path}: {len(records)} frames, mean loss={payload['mean_loss']:.4f}, MAE={payload['mean_mae']:.4f}", flush=True)
            records = []
    for episode in episodes:
        if expected[episode] != (args.max_frames or int(meta.episodes["length"][episode])):
            raise RuntimeError(f"Episode {episode} incomplete: {expected[episode]} frames")


if __name__ == "__main__":
    main()
