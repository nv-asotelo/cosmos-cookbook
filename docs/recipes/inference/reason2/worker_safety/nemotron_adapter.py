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

"""NemotronAdapter — inference wrapper for NVIDIA-Nemotron-Nano-12B-v2-VL-BF16.

Loads model once, exposes run_inference(video_path) that extracts frames and
runs the worker-safety classification prompt.
"""

import json
import re
import shutil

from PIL import Image

from harness_benchmark import SYSTEM_INSTRUCTIONS, USER_PROMPT_CONTENT
from nemotron_frames import extract_frames


class NemotronAdapter:
    """Inference adapter for NVIDIA-Nemotron-Nano-12B-v2-VL-BF16.

    Loads the model, processor, and tokenizer once on construction.
    Call run_inference(video_path) to classify a video clip.
    """

    MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-BF16"

    # /no_think disables chain-of-thought reasoning as required by model card.
    NEMOTRON_SYSTEM_PROMPT = "/no_think\n" + SYSTEM_INSTRUCTIONS

    def __init__(self, model_id: str = None, device: str = "cuda:0"):
        """Load the model, processor, and tokenizer.

        Args:
            model_id: HuggingFace model ID to load. Defaults to MODEL_ID.
            device: Device map string passed to from_pretrained (e.g. "cuda:0").
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        self.model_id = model_id or self.MODEL_ID

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        print(f"[NEMOTRON] Model loaded: {self.model_id}")

    def run_inference(
        self,
        video_path: str,
        fps: float = 2.0,
        max_new_tokens: int = 256,
    ) -> dict:
        """Run worker-safety classification on a video file.

        Extracts frames from the video, builds a multi-image conversation,
        generates a response, and parses it as JSON.

        Args:
            video_path: Path to the input video file.
            fps: Frame sampling rate for ffmpeg extraction (default: 2.0).
            max_new_tokens: Maximum tokens to generate (default: 256).

        Returns:
            A dict containing the parsed JSON classification result, or an
            error dict with keys "nemotron_error" (and optionally "raw_output")
            if generation or JSON parsing failed.
        """
        cleanup_dir = None
        output_text = ""

        try:
            frame_paths, cleanup_dir = extract_frames(video_path, fps=fps)

            # Nemotron processor requires PIL images, not file path strings
            pil_images = [Image.open(p).convert("RGB") for p in frame_paths]

            messages = [
                {"role": "system", "content": self.NEMOTRON_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": p} for p in frame_paths],
                        {"type": "text", "text": USER_PROMPT_CONTENT},
                    ],
                },
            ]

            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                text=[text],
                images=pil_images,
                return_tensors="pt",
            ).to(self.model.device)

            output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

            output_text = self.processor.batch_decode(
                [output_ids[0][inputs.input_ids.shape[-1]:]],
                skip_special_tokens=True,
            )[0]

            # Strip chain-of-thought think blocks
            output_text = re.sub(
                r"<think>.*?</think>", "", output_text, flags=re.DOTALL
            ).strip()

            # Strip ```json fences
            output_text = (
                output_text.replace("```json", "").replace("```", "").strip()
            )

            return json.loads(output_text)

        except json.JSONDecodeError as e:
            return {
                "nemotron_error": f"JSON parse error: {e}",
                "raw_output": output_text,
            }

        except Exception as e:
            return {"nemotron_error": str(e)}

        finally:
            if cleanup_dir is not None:
                shutil.rmtree(cleanup_dir, ignore_errors=True)
