from pathlib import Path

from PIL import Image

from lerobot.envs.configs import EnvClientConfig, LiberoEnvConfig


root = Path(__file__).resolve().parents[1]
cfg = LiberoEnvConfig(
    task="libero_10",
    task_ids=[1],
    episode_length=520,
    observation_height=256,
    observation_width=256,
    server=EnvClientConfig(endpoint="tcp/127.0.0.1:7448"),
)
env = cfg.create_env(task_group="libero_10", task_id=1)
try:
    env.reset(seed=1000)
    frame = env.call("render")[0]
    print("Render:", frame.shape, frame.dtype, frame.min(), frame.max())
    Image.fromarray(frame).save(root / "outputs" / "task7_libero_rollout" / "probe.png")
finally:
    env.close()
