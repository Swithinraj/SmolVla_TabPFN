"""Run one SmolVLA action prediction on a downloaded LIBERO observation."""

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
os.environ.setdefault("HF_LEROBOT_HOME", str(ROOT / ".cache" / "lerobot"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def main():
    model_dir = ROOT / "models" / "smolvla_libero"
    dataset = LeRobotDataset("lerobot/libero", root=ROOT / "data" / "libero", episodes=[0], video_backend="pyav")
    observation = dataset[0]
    demo_action = observation["action"].clone()
    config = PreTrainedConfig.from_pretrained(model_dir)
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SmolVLA on {config.device}…", flush=True)
    policy = SmolVLAPolicy.from_pretrained(model_dir, config=config, local_files_only=True)
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=model_dir,
        preprocessor_overrides={"device_processor": {"device": config.device}},
    )
    with torch.inference_mode():
        predicted = postprocessor(policy.select_action(preprocessor(observation)))
    print("Task:", observation["task"])
    print("Demonstration action:", demo_action.tolist())
    print("Predicted action:", predicted.detach().cpu().flatten().tolist())
    if config.device == "cuda":
        print("Peak CUDA allocation MB:", round(torch.cuda.max_memory_allocated() / 1e6, 1))


if __name__ == "__main__":
    main()
