"""Fine-tune SmolVLA on two LIBERO tasks with a small held-out split."""

import argparse
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--save-freq", type=int, default=20000)
    parser.add_argument("--log-freq", type=int, default=10)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    output = ROOT / "outputs" / f"tuning_{datetime.now():%Y%m%d_%H%M%S}"
    log = ROOT / ".tmp" / f"{output.name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
        str(ROOT / "train_subset.ps1"),
        "-Steps", str(args.steps), "-BatchSize", "1", "-EpisodeIds", "1,2,4",
        "-LogFreq", str(args.log_freq), "-SaveFreq", str(args.save_freq),
        "-OutputDir", str(output),
    ]
    if args.lr > 0:
        command.extend(["-LearningRate", str(args.lr)])
    print(f"Training output: {output}", flush=True)
    print("Training on episodes 1, 2, 4; episodes 3 and 5 are reserved for separate evaluation.", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if output.is_dir():
        shutil.copyfile(log, output / "training.log")
        log = output / "training.log"
    print(f"Exit code: {result.returncode}; log: {log}", flush=True)
    lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines:
        if "eval_loss=" in line or "step:" in line or "End of training" in line:
            print(line[-300:])
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
