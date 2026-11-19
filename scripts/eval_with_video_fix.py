"""Run LeRobot eval with integer FPS for remote simulator video encoding.

LeRobot 0.7.0 sends the remote env's render_fps as a float. PyAV's add_stream
requires an integer or Fraction; this wrapper changes only the video writer's
FPS argument and leaves policy actions and simulator stepping untouched.
"""

from fractions import Fraction

from lerobot.scripts import lerobot_eval


original_write_video = lerobot_eval.write_video


def write_video_with_rational_fps(video_path, stacked_frames, fps):
    return original_write_video(video_path, stacked_frames, Fraction(fps).limit_denominator(1000))


lerobot_eval.write_video = write_video_with_rational_fps

if __name__ == "__main__":
    lerobot_eval.eval_main()
