# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

"""
run_headless.py — Headless runner for the Worker Safety recipe.

Runs the full worker_safety.py inference pipeline on a machine with no
display, then either exits cleanly (default) or serves the FiftyOne UI
on a configurable port so a remote user can port-forward and browse results.

Usage
-----
# Default: run on the built-in HuggingFace dataset
python run_headless.py

# Bring your own videos (local directory on the instance)
python run_headless.py --video-dir ~/my_videos/

# Use a different HuggingFace dataset
python run_headless.py --hf-dataset your-org/your-dataset

# Inference + serve FiftyOne on port 5151 (port-forward from local machine)
python run_headless.py --serve

# Full custom example
python run_headless.py --video-dir ~/my_videos/ --results ~/my_results.json --serve --port 5151

Upload your own videos to the instance first:
  brev copy ./my_videos/ <instance-name>:/home/shadeform/my_videos/

Output
------
inference_results.json  — one record per video:
  {
    "filepath":     "/path/to/video.mp4",
    "ground_truth": "Safe Walkway",          # from dataset metadata (if available)
    "safety_label": "Safe Walkway",          # Cosmos Reason2 prediction
    "hazard":       false,                   # is_hazardous flag from model
    "description":  "...",                   # video_description from model
    "cosmos_error": null                     # populated on inference failure
  }

Port-forward (when --serve is active)
--------------------------------------
On your LOCAL machine:
  brev port-forward <instance-name> -p 5151:5151
Then open: http://localhost:5151
"""

import argparse
import json
import os
import pathlib
import sys

# Supported video extensions
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

# Default HuggingFace dataset for the Worker Safety recipe
DEFAULT_HF_DATASET = "pjramg/Safe_Unsafe_Test"


def parse_args():
    p = argparse.ArgumentParser(description="Headless Worker Safety inference")
    p.add_argument(
        "--serve",
        action="store_true",
        help="After inference, launch FiftyOne UI and block until Ctrl+C",
    )
    p.add_argument(
        "--port",
        type=int,
        default=5151,
        help="Port to serve FiftyOne on (default: 5151)",
    )
    p.add_argument(
        "--results",
        default="inference_results.json",
        help="Path to write JSON results (default: inference_results.json)",
    )
    p.add_argument(
        "--video-dir",
        default=None,
        metavar="PATH",
        help=(
            "Directory of video files to run inference on. "
            "Supported formats: .mp4 .avi .mov .mkv .webm .m4v. "
            "Upload videos first with: "
            "brev copy ./my_videos/ <instance>:/home/shadeform/my_videos/"
        ),
    )
    p.add_argument(
        "--hf-dataset",
        default=DEFAULT_HF_DATASET,
        metavar="ORG/REPO",
        help=(
            "HuggingFace dataset slug to load (default: %(default)s). "
            "Ignored when --video-dir is set."
        ),
    )
    return p.parse_args()


def patch_fiftyone_launch():
    """Replace fo.launch_app with a no-op so headless exec doesn't block."""
    import fiftyone as fo

    _noop = type("_Session", (), {"wait": lambda self: None})()
    fo.launch_app = lambda *args, **kwargs: _noop


def load_custom_dataset(video_dir):
    """
    Create a FiftyOne dataset from a local directory of video files.
    Returns the dataset name so it can be used by patch_dataset_loading().
    """
    import fiftyone as fo

    video_dir = pathlib.Path(video_dir).expanduser().resolve()
    if not video_dir.is_dir():
        print(f"ERROR: --video-dir '{video_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    videos = sorted(
        f for f in video_dir.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not videos:
        print(
            f"ERROR: no video files found in '{video_dir}'. "
            f"Supported: {', '.join(sorted(VIDEO_EXTENSIONS))}",
            file=sys.stderr,
        )
        sys.exit(1)

    name = "custom_inference"
    if fo.dataset_exists(name):
        fo.delete_dataset(name)

    dataset = fo.Dataset(name=name, persistent=True)
    dataset.add_samples([fo.Sample(filepath=str(v)) for v in videos])
    print(f"Loaded {len(dataset)} videos from '{video_dir}'")
    return name


def patch_dataset_loading(custom_name, hf_slug):
    """
    Monkey-patch fiftyone so worker_safety.py uses our pre-loaded dataset
    instead of making a HuggingFace download call.
    """
    import fiftyone as fo
    import fiftyone.utils.huggingface as fouh

    _original_load = fo.load_dataset

    def _patched_load(name, *args, **kwargs):
        if name in (hf_slug, custom_name):
            return _original_load(custom_name)
        return _original_load(name, *args, **kwargs)

    fo.load_dataset = _patched_load
    # Also intercept the load_from_hub call at the top of worker_safety.py
    fouh.load_from_hub = lambda *args, **kwargs: _original_load(custom_name)


def run_inference():
    """Execute worker_safety.py in the current directory."""
    recipe = pathlib.Path(__file__).parent / "worker_safety.py"
    if not recipe.exists():
        print(f"ERROR: {recipe} not found. Run from the worker_safety directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Running {recipe} ...")
    exec(compile(recipe.read_text(), str(recipe), "exec"), {"__file__": str(recipe)})


def save_results(output_path):
    """Export every sample's fields to JSON."""
    import fiftyone as fo

    datasets = fo.list_datasets()
    if not datasets:
        print("ERROR: no FiftyOne datasets found after inference.", file=sys.stderr)
        sys.exit(1)

    dataset = fo.load_dataset(datasets[0])
    print(f"Dataset: {datasets[0]}  ({len(dataset)} samples)")

    records = []
    for sample in dataset:
        cosmos = sample.get_field("cosmos_analysis") or {}
        hazard_info = cosmos.get("hazard_detection", {}) if isinstance(cosmos, dict) else {}
        records.append(
            {
                "filepath": sample.filepath,
                "ground_truth": (
                    sample.get_field("ground_truth").label
                    if sample.get_field("ground_truth")
                    else None
                ),
                "safety_label": (
                    sample.get_field("safety_label").label
                    if sample.get_field("safety_label")
                    else None
                ),
                "hazard": hazard_info.get("is_hazardous"),
                "description": cosmos.get("video_description") if isinstance(cosmos, dict) else None,
                "cosmos_error": sample.get_field("cosmos_error"),
            }
        )

    output_path = pathlib.Path(output_path)
    output_path.write_text(json.dumps(records, indent=2))

    labeled = [r for r in records if r["safety_label"]]
    errors = [r for r in records if r["cosmos_error"]]
    safe = [r for r in labeled if r.get("hazard") is False]
    unsafe = [r for r in labeled if r.get("hazard") is True]

    print(f"\nResults written to: {output_path}")
    print(f"  Total samples : {len(records)}")
    print(f"  Classified    : {len(labeled)}")
    print(f"    Safe        : {len(safe)}")
    print(f"    Unsafe      : {len(unsafe)}")
    print(f"  Errors        : {len(errors)}")

    return datasets[0]


def serve_fiftyone(dataset_name, port):
    """Launch FiftyOne UI on the given port and block until Ctrl+C."""
    import fiftyone as fo

    dataset = fo.load_dataset(dataset_name)
    print(f"\nLaunching FiftyOne on port {port} ...")
    print(f"  On your LOCAL machine run:")
    print(f"    brev port-forward <instance-name> -p {port}:{port}")
    print(f"  Then open: http://localhost:{port}")
    print("  Press Ctrl+C to stop.\n")

    session = fo.launch_app(dataset, port=port, address="0.0.0.0", remote=True)
    session.wait()


def main():
    args = parse_args()

    # Patch fo.launch_app before executing the recipe so the recipe's own
    # `session = fo.launch_app(dataset); session.wait()` becomes a no-op.
    patch_fiftyone_launch()

    # If the user supplied their own videos, load them into FiftyOne and
    # redirect worker_safety.py's dataset calls to that local dataset.
    if args.video_dir:
        custom_name = load_custom_dataset(args.video_dir)
        patch_dataset_loading(custom_name, args.hf_dataset)
    elif args.hf_dataset != DEFAULT_HF_DATASET:
        # User specified a different HF dataset but no local dir — pass through.
        # worker_safety.py will download it from HuggingFace as normal.
        print(f"Using HuggingFace dataset: {args.hf_dataset}")

    run_inference()

    dataset_name = save_results(args.results)

    if args.serve:
        # Restore the real fo.launch_app now that inference is complete
        import importlib
        import fiftyone as fo
        importlib.reload(fo)
        serve_fiftyone(dataset_name, args.port)
    else:
        print("\nDone. Re-run with --serve to launch the FiftyOne UI.")


if __name__ == "__main__":
    main()
