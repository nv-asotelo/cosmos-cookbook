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
# Inference only — save results to JSON and exit
python run_headless.py

# Inference + serve FiftyOne on port 5151 (port-forward from local machine)
python run_headless.py --serve

# Specify a different port
python run_headless.py --serve --port 5152

# If weights are stored outside the default path, override MODEL_DIR
MODEL_DIR=/data/models python run_headless.py --serve

Output
------
inference_results.json  — one record per video:
  {
    "filepath":     "/path/to/video.mp4",
    "ground_truth": "Safe Walkway",          # from dataset metadata
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
    return p.parse_args()


def patch_fiftyone_launch():
    """Replace fo.launch_app with a no-op so headless exec doesn't block."""
    import fiftyone as fo

    _noop = type("_Session", (), {"wait": lambda self: None})()
    fo.launch_app = lambda *args, **kwargs: _noop


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
