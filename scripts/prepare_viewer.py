"""Export the downloaded LIBERO episodes into a browser-friendly index."""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import av
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "libero"
OUT = ROOT / "viewer" / "episodes.json"


def browser_video(source: Path, relative: str) -> str:
    """Transcode AV1 to H.264, which Windows browsers decode reliably."""
    target = ROOT / "viewer" / "media" / relative
    if not target.is_file() or not target.stat().st_size:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(".partial.mp4")
        with av.open(source) as input_container, av.open(temporary, mode="w", format="mp4") as output_container:
            input_stream = input_container.streams.video[0]
            stream = output_container.add_stream("libx264", rate=input_stream.average_rate)
            stream.width = input_stream.width
            stream.height = input_stream.height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "24", "preset": "fast"}
            for index, frame in enumerate(input_container.decode(video=0)):
                reformatted = frame.reformat(format="yuv420p")
                reformatted.pts = index
                reformatted.time_base = Fraction(1, 10)
                for packet in stream.encode(reformatted):
                    output_container.mux(packet)
            for packet in stream.encode():
                output_container.mux(packet)
        temporary.replace(target)
        print(f"Preview video: {target.relative_to(ROOT)} ({target.stat().st_size:,} bytes)")
    return "/media/" + relative.replace("\\", "/")


def browser_clip(source: Path, camera: str, episode: int, start: float, length: int, fps: int = 10) -> str:
    """Transcode only one episode, so selected test episodes load quickly."""
    relative = f"episodes/episode-{episode:03d}-{camera.split('.')[-1]}.mp4"
    target = ROOT / "viewer" / "media" / relative
    if target.is_file() and target.stat().st_size:
        return "/media/" + relative
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".partial.mp4")
    with av.open(source) as input_container, av.open(temporary, mode="w", format="mp4") as output_container:
        input_stream = input_container.streams.video[0]
        output_stream = output_container.add_stream("libx264", rate=fps)
        output_stream.width = input_stream.width
        output_stream.height = input_stream.height
        output_stream.pix_fmt = "yuv420p"
        output_stream.options = {"crf": "24", "preset": "fast"}
        input_container.seek(max(0, int(start / float(input_stream.time_base))), stream=input_stream)
        written = 0
        for frame in input_container.decode(input_stream):
            if frame.time is not None and frame.time + 0.05 < start:
                continue
            if written >= length:
                break
            reformatted = frame.reformat(format="yuv420p")
            reformatted.pts = written
            reformatted.time_base = Fraction(1, fps)
            for packet in output_stream.encode(reformatted):
                output_container.mux(packet)
            written += 1
        for packet in output_stream.encode():
            output_container.mux(packet)
    if written != length:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Episode {episode} {camera}: encoded {written} frames, expected {length}")
    temporary.replace(target)
    print(f"Preview clip: {target.relative_to(ROOT)} ({target.stat().st_size:,} bytes)")
    return "/media/" + relative


def poster(source: Path, camera: str, episode_index: int, timestamp: float) -> str:
    target = ROOT / "viewer" / "posters" / camera / f"episode-{episode_index:03d}.jpg"
    if not target.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        with av.open(source) as container:
            stream = container.streams.video[0]
            container.seek(max(0, int(timestamp / float(stream.time_base))), stream=stream)
            for frame in container.decode(stream):
                if frame.time is None or frame.time + 0.05 >= timestamp:
                    frame.to_image().save(target, "JPEG", quality=90)
                    break
        if not target.is_file():
            raise RuntimeError(f"Could not make poster for {camera}, episode {episode_index}")
    return f"/posters/{camera}/episode-{episode_index:03d}.jpg"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=8, help="How many downloaded episodes to prepare for browser playback")
    parser.add_argument("--episode-ids", default="", help="Extra episode IDs to prepare as standalone clips, comma-separated")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    manifest = json.loads((DATASET / "subset.json").read_text(encoding="utf-8"))
    extra_ids = [int(value.strip()) for value in args.episode_ids.split(",") if value.strip()]
    available = set(manifest["episodes"])
    if any(episode not in available for episode in extra_ids):
        parser.error("An extra episode ID is not in the local dataset")
    selected = list(dict.fromkeys(manifest["episodes"][:args.limit] + extra_ids))
    metadata = pd.read_parquet(DATASET / "meta/episodes/chunk-000/file-000.parquet").set_index("episode_index")
    tasks_frame = pd.read_parquet(DATASET / "meta/tasks.parquet")
    tasks = dict(zip(tasks_frame["task_index"].astype(int), tasks_frame.index.astype(str)))
    result = []
    preview_urls = {}
    for episode_index in selected:
        meta = metadata.loc[episode_index]
        data_path = f"data/chunk-{int(meta['data/chunk_index']):03d}/file-{int(meta['data/file_index']):03d}.parquet"
        frames = pd.read_parquet(DATASET / data_path)
        frames = frames.loc[frames["episode_index"] == episode_index].sort_values("frame_index")
        if len(frames) != int(meta["length"]):
            raise RuntimeError(f"Episode {episode_index}: expected {meta['length']} frames, got {len(frames)}")
        videos = {}
        for camera in ("observation.images.image", "observation.images.image2"):
            prefix = f"videos/{camera}"
            relative = f"videos/{camera}/chunk-{int(meta[f'{prefix}/chunk_index']):03d}/file-{int(meta[f'{prefix}/file_index']):03d}.mp4"
            if episode_index in extra_ids:
                video_url = browser_clip(DATASET / relative, camera, episode_index, float(meta[f"{prefix}/from_timestamp"]), len(frames))
                video_start, video_end = 0.0, len(frames) / 10
            else:
                if relative not in preview_urls:
                    preview_urls[relative] = browser_video(DATASET / relative, relative)
                video_url = preview_urls[relative]
                video_start = float(meta[f"{prefix}/from_timestamp"])
                video_end = float(meta[f"{prefix}/to_timestamp"])
            videos[camera] = {
                "url": video_url,
                "poster": poster(DATASET / relative, camera, episode_index, float(meta[f"{prefix}/from_timestamp"])),
                "start": video_start,
                "end": video_end,
            }
        result.append({
            "id": episode_index,
            "task": tasks[int(frames["task_index"].iloc[0])],
            "frames": len(frames),
            "duration": round(len(frames) / 10, 3),
            "fps": 10,
            "videos": videos,
            "state": [list(map(float, values)) for values in frames["observation.state"]],
            "action": [list(map(float, values)) for values in frames["action"]],
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"source": manifest["source"], "total_episodes": len(manifest["episodes"]), "episodes": result}, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {len(result)} episodes to {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
