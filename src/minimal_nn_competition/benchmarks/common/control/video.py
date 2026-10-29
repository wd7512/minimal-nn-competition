"""Stream native RGB frames to a static-site-compatible MP4; no replay physics."""

import subprocess
from contextlib import contextmanager


@contextmanager
def encoder(env, path):
    if path is None:
        yield None
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = env.render()
    height, width, _ = frame.shape
    process = subprocess.Popen(
        [
            "ffmpeg",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            str(env.metadata["render_fps"]),
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-crf",
            "28",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        stdin=subprocess.PIPE,
    )
    assert process.stdin is not None
    try:
        process.stdin.write(frame.tobytes())
        yield process.stdin
    finally:
        try:
            process.stdin.close()
        finally:
            if process.wait():
                path.unlink(missing_ok=True)
                raise RuntimeError("Native Gymnasium video encoding failed")
