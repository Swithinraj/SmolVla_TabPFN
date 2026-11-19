"""Build an exact base-weight baseline with LIBERO task feature metadata."""

import os
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/task7_smoke_20261116/checkpoints/000002/pretrained_model"
TARGET = ROOT / "outputs/task7_base_baseline/pretrained_model"
BASE_WEIGHTS = ROOT / "models/smolvla_base/model.safetensors"


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    for name in (
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ):
        shutil.copy2(SOURCE / name, TARGET / name)
    if not (TARGET / "model.safetensors").exists():
        os.link(BASE_WEIGHTS, TARGET / "model.safetensors")
    shutil.copytree(SOURCE / "tokenizer", TARGET / "tokenizer", dirs_exist_ok=True)
    (TARGET.parent / "README.md").write_text(
        "Exact lerobot/smolvla_base model weights (hard-linked), with the LIBERO 8D state/7D action "
        "feature configuration and dataset normalization produced by the two-step smoke run. "
        "No smoke-run model weights are used. The source dataset's global statistics include the two test episodes.\n",
        encoding="utf-8",
    )
    print(TARGET)


if __name__ == "__main__":
    main()
