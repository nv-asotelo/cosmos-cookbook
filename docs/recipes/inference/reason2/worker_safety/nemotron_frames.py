# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""FFmpeg-based frame extraction for NVIDIA-Nemotron-Nano-12B-v2-VL-BF16.

Extracts frames at configurable FPS, enforces MIN_FRAMES=8 (pads last frame),
caps at MAX_FRAMES=128.
"""

import os
import subprocess
import tempfile
from pathlib import Path

MIN_FRAMES = 8
MAX_FRAMES = 8


def extract_frames(
    video_path: str,
    fps: float = 2.0,
    out_dir: str = None,
) -> tuple[list[str], str]:
    """Extract frames from a video file using ffmpeg.

    Args:
        video_path: Path to the input video file.
        fps: Frame rate at which to sample frames (default: 2.0).
        out_dir: Directory to write frames into. If None, a temporary
                 directory is created automatically.

    Returns:
        A tuple of (sorted_frame_paths, cleanup_dir) where:
        - sorted_frame_paths is a list of absolute paths to extracted PNG frames.
        - cleanup_dir is the directory that should be removed after use.

    Raises:
        RuntimeError: If ffmpeg exits with a non-zero return code.
    """
    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="nemotron_frames_")

    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    output_pattern = os.path.join(out_dir, "frame_%05d.png")

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        output_pattern,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed with exit code {result.returncode}.\n"
            f"stderr: {result.stderr}"
        )

    frame_paths = sorted(
        str(p) for p in Path(out_dir).glob("frame_*.png")
    )

    if len(frame_paths) == 0:
        raise RuntimeError(
            f"ffmpeg produced no frames from '{video_path}'. "
            "Check that the file is a valid video and ffmpeg is installed."
        )

    # Pad up to MIN_FRAMES by duplicating the last frame
    if len(frame_paths) < MIN_FRAMES:
        last_frame = frame_paths[-1]
        while len(frame_paths) < MIN_FRAMES:
            frame_paths.append(last_frame)

    # Truncate to MAX_FRAMES
    if len(frame_paths) > MAX_FRAMES:
        frame_paths = frame_paths[:MAX_FRAMES]

    return frame_paths, out_dir
