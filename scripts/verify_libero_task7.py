import os
from pathlib import Path

os.environ["LIBERO_CONFIG_PATH"] = str(Path(__file__).resolve().parents[1] / ".cache" / "libero")

from libero.libero import benchmark

suite = benchmark.get_benchmark_dict()["libero_10"]()
for task_id in range(len(suite.tasks)):
    task = suite.get_task(task_id)
    print(f"{task_id}: {task.language}")
