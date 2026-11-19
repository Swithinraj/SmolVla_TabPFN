"""Download a SmolVLA checkpoint into this project, keeping 100 GB free."""

import argparse
import shutil
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = (
    "README.md", "config.json", "model.safetensors",
    "policy_preprocessor.json", "policy_postprocessor.json",
    "policy_preprocessor_step_5_normalizer_processor.safetensors",
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("smolvla_libero", "smolvla_base"), default="smolvla_libero")
    args = parser.parse_args()
    out = ROOT / "models" / args.model
    base = f"https://huggingface.co/lerobot/{args.model}/resolve/main/"
    out.mkdir(parents=True, exist_ok=True)
    files = FILES + (("train_config.json",) if args.model == "smolvla_libero" else ())
    for name in files:
        target = out / name
        if target.is_file() and target.stat().st_size:
            print(f"kept {name} ({target.stat().st_size:,} bytes)", flush=True)
            continue
        part = target.with_suffix(target.suffix + ".partial")
        with urllib.request.urlopen(base + name, timeout=90) as source, part.open("wb") as sink:
            size = int(source.headers.get("Content-Length", 0))
            if shutil.disk_usage(ROOT).free - size < 100_000_000_000:
                raise RuntimeError("Checkpoint would leave less than 100 GB free on D:")
            shutil.copyfileobj(source, sink, length=1024 * 1024)
        part.replace(target)
        print(f"saved {name} ({target.stat().st_size:,} bytes)", flush=True)
    print(f"Checkpoint ready: {out}")


if __name__ == "__main__":
    main()
