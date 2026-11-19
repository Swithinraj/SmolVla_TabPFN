"""Record the single-task LIBERO split used for this experiment."""

import json
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "libero"
TASK_INDEX = 7
TEST_EPISODES = [53, 347]
TASK_TEXT = "put both the cream cheese box and the butter in the basket"


def main() -> None:
    tasks = pd.read_parquet(DATASET / "meta" / "tasks.parquet")
    assert tasks.index[TASK_INDEX] == TASK_TEXT
    frames = ds.dataset(str(DATASET / "data"), format="parquet").to_table(
        columns=["episode_index", "task_index"]
    ).to_pandas()
    episode_tasks = frames.groupby("episode_index").task_index.first()
    task_episodes = sorted(map(int, episode_tasks[episode_tasks == TASK_INDEX].index))
    assert len(task_episodes) == 49
    assert set(TEST_EPISODES).issubset(task_episodes)
    train_episodes = [episode for episode in task_episodes if episode not in TEST_EPISODES]
    lengths = pd.read_parquet(DATASET / "meta/episodes/chunk-000/file-000.parquet").set_index("episode_index")
    report = {
        "dataset": "lerobot/libero",
        "task_index": TASK_INDEX,
        "task": TASK_TEXT,
        "train_episodes": train_episodes,
        "test_episodes": TEST_EPISODES,
        "train_frames": int(lengths.loc[train_episodes, "length"].sum()),
        "test_frames": int(lengths.loc[TEST_EPISODES, "length"].sum()),
        "selection": "Among 49 task episodes, 53 and 347 had the largest mean pixel difference between both 64x64 downsampled initial camera views (10.337 versus 7.727 median pair distance). Both starting views were visually inspected.",
        "test_limit": "Held out from this run only; the already LIBERO-tuned checkpoint has seen the source dataset.",
    }
    path = DATASET / "task7_split.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
