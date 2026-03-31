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
restore_labels.py — Re-apply Cosmos Reason2 inference labels to a fresh FiftyOne dataset.

Use this when migrating to a new instance to avoid re-running inference.
Reads inference_results.json (produced by run_headless.py) and writes
safety_label + cosmos_analysis fields back into the FiftyOne dataset.

Usage
-----
# On the new instance, after the dataset has been loaded from HuggingFace:
python restore_labels.py --results ~/inference_results.json

# Then optionally serve FiftyOne to verify:
python restore_labels.py --results ~/inference_results.json --serve --port 5151
"""

import argparse
import json
import pathlib
import sys


def parse_args():
    p = argparse.ArgumentParser(description="Restore CR2 labels from inference_results.json")
    p.add_argument(
        "--results",
        default="inference_results.json",
        help="Path to inference_results.json (default: inference_results.json)",
    )
    p.add_argument(
        "--serve",
        action="store_true",
        help="Launch FiftyOne UI after restoring labels",
    )
    p.add_argument(
        "--port",
        type=int,
        default=5151,
        help="Port for FiftyOne UI (default: 5151)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    results_path = pathlib.Path(args.results).expanduser()
    if not results_path.exists():
        print(f"ERROR: {results_path} not found.", file=sys.stderr)
        sys.exit(1)

    import fiftyone as fo

    datasets = fo.list_datasets()
    if not datasets:
        print("ERROR: No FiftyOne datasets found. Load the dataset first:", file=sys.stderr)
        print("  python -c \"import fiftyone.utils.huggingface as fouh; fouh.load_from_hub('pjramg/Safe_Unsafe_Test', persistent=True)\"", file=sys.stderr)
        sys.exit(1)

    dataset = fo.load_dataset(datasets[0])
    print(f"Dataset: {datasets[0]}  ({len(dataset)} samples)")

    with open(results_path) as f:
        results = json.load(f)

    # Build lookup by filename (basename) — filepaths differ between instances
    lookup = {pathlib.Path(r["filepath"]).name: r for r in results}

    matched = 0
    skipped = 0
    for sample in dataset.iter_samples(autosave=True, progress=True):
        name = pathlib.Path(sample.filepath).name
        r = lookup.get(name)
        if r is None:
            skipped += 1
            continue

        if r.get("safety_label"):
            sample["safety_label"] = fo.Classification(label=r["safety_label"])

        if r.get("description") or r.get("hazard") is not None:
            sample["cosmos_analysis"] = {
                "prediction_label": r.get("safety_label"),
                "video_description": r.get("description"),
                "hazard_detection": {
                    "is_hazardous": r.get("hazard"),
                },
            }

        if r.get("cosmos_error"):
            sample["cosmos_error"] = r["cosmos_error"]

        matched += 1

    print(f"\nRestored labels: {matched} samples matched, {skipped} skipped")
    print(f"Label distribution: {dataset.count_values('safety_label.label')}")

    if args.serve:
        print(f"\nLaunching FiftyOne on port {args.port} ...")
        print(f"  brev port-forward <instance-name> -p {args.port}:{args.port}")
        print(f"  Then open: http://localhost:{args.port}")
        session = fo.launch_app(dataset, port=args.port, address="0.0.0.0", remote=True)
        session.wait()
    else:
        print("\nDone. Re-run with --serve to launch the FiftyOne UI.")


if __name__ == "__main__":
    main()
