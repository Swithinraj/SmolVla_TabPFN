"""Publish a completed LeRobot LIBERO evaluation to the local viewer."""

import json
import shutil
from pathlib import Path

import av


ROOT = Path(__file__).resolve().parents[1]
EVALUATION = ROOT / "outputs" / "task7_libero_rollout"
DESTINATION = ROOT / "viewer" / "rollouts"


def main() -> None:
    info = json.loads((EVALUATION / "eval_info.json").read_text(encoding="utf-8"))
    task = next(
        row for row in info["per_task"]
        if row["task_group"] == "libero_10" and row["task_id"] == 1
    )
    metrics = task["metrics"]
    if len(metrics["successes"]) != 1 or len(metrics["video_paths"]) != 1:
        raise ValueError("Expected one evaluated episode with one rollout video")

    source = Path(metrics["video_paths"][0])
    if not source.is_absolute():
        source = ROOT / source
    if not source.is_file():
        raise FileNotFoundError(source)

    DESTINATION.mkdir(parents=True, exist_ok=True)
    target = DESTINATION / "task7_model_rollout.mp4"
    poster = DESTINATION / "task7_model_poster.png"
    with av.open(str(source)) as recorded:
        fps = float(recorded.streams.video[0].average_rate)
        if abs(fps - 20) > 0.01:
            raise ValueError(f"Expected a 20 Hz rollout video, got {fps}")
        frames = 0
        for frame in recorded.decode(video=0):
            if frames == 0:
                frame.to_image().save(poster)
            frames += 1
    shutil.copyfile(source, target)

    descriptions = json.loads((EVALUATION / "task_descriptions.json").read_text(encoding="utf-8"))
    manifest = {
        "video": "/rollouts/task7_model_rollout.mp4",
        "poster": "/rollouts/task7_model_poster.png",
        "task": descriptions["libero_10_1"],
        "task_id": 1,
        "dataset_task_index": 7,
        "success": bool(metrics["successes"][0]),
        "steps": frames,
        "fps": 20,
        "seed": 1000,
        "note": "One independent closed-loop rollout, seed 1000. The video is timed at 20 Hz. The loss plots use recorded test demonstrations, not this rollout.",
    }
    (DESTINATION / "task7.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
