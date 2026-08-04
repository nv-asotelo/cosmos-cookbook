import importlib.util
import io
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "byo_video_batch_inference.py"
SPEC = importlib.util.spec_from_file_location("byo_video_batch_inference_test", SCRIPT)
batch = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(batch)


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


class FakeRequests:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, headers=None, timeout=None, **kwargs):
        self.calls.append({
            "url": url,
            "headers": dict(headers or {}),
            "timeout": timeout,
            **kwargs,
        })
        return self.response


class HostedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.original_comparison = json.loads(json.dumps(batch.STATE["comparison"]))
        self.original_videos = list(batch.STATE["videos"])

    def tearDown(self):
        with batch.STATE_LOCK:
            batch.STATE["comparison"] = self.original_comparison
            batch.STATE["videos"] = self.original_videos

    def test_dynamic_catalog_uses_configured_auth_and_gates_modalities(self):
        runtime_key = "opaque-runtime-value-for-unit-test"
        fake = FakeRequests(FakeResponse(200, {"data": [
            {"id": "vendor/video-model", "name": "Video Model", "input_modalities": ["text", "video"]},
            {"id": "vendor/text-model", "name": "Text Model", "input_modalities": ["text"]},
            {"id": "vendor/unknown-model", "name": "Unknown Model"},
            {"id": "vendor/safe-model", "name": f"echo {runtime_key}", "input_modalities": ["text"]},
            {"id": f"vendor/{runtime_key}/model", "name": "must be skipped"},
        ]}))
        with mock.patch.object(batch, "requests", fake), \
             mock.patch.object(batch, "NVIDIA_HOSTED_CATALOG_URL", "https://catalog.invalid/models"), \
             mock.patch.object(batch, "NVIDIA_HOSTED_AUTH_HEADER", "X-Test-Key"), \
             mock.patch.object(batch, "NVIDIA_HOSTED_AUTH_SCHEME", "Token"):
            result = batch.discover_hosted_models(runtime_key)

        self.assertEqual(fake.calls[0]["url"], "https://catalog.invalid/models")
        self.assertEqual(fake.calls[0]["headers"]["X-Test-Key"], f"Token {runtime_key}")
        self.assertFalse(fake.calls[0]["allow_redirects"])
        models = {item["id"]: item for item in result["models"]}
        self.assertTrue(models["vendor/video-model"]["supports_video"])
        self.assertFalse(models["vendor/text-model"]["supports_video"])
        self.assertFalse(models["vendor/unknown-model"]["supports_video"])
        self.assertNotIn(runtime_key, json.dumps(result))
        self.assertNotIn(runtime_key, json.dumps(batch.snapshot()))

    def test_catalog_accepts_name_ids_and_top_level_capability_flags(self):
        fake = FakeRequests(FakeResponse(200, {"data": [
            {
                "name": "vendor/top-level-video",
                "supports_video": True,
                "supports_vlm": True,
                "supports_text": False,
            },
        ]}))
        with mock.patch.object(batch, "requests", fake), \
             mock.patch.object(batch, "NVIDIA_HOSTED_CATALOG_URL", "https://catalog.invalid/models"):
            result = batch.discover_hosted_models("opaque-runtime-value-for-unit-test")

        model = result["models"][0]
        self.assertEqual(model["id"], "vendor/top-level-video")
        self.assertTrue(model["supports_video"])
        self.assertTrue(model["supports_image"])
        self.assertFalse(model["supports_text"])

    def test_catalog_redirect_is_not_followed(self):
        fake = FakeRequests(FakeResponse(302, {}))
        with mock.patch.object(batch, "requests", fake), \
             mock.patch.object(batch, "NVIDIA_HOSTED_CATALOG_URL", "https://catalog.invalid/models"):
            result = batch.discover_hosted_models("opaque-runtime-value-for-unit-test")
        self.assertFalse(fake.calls[0]["allow_redirects"])
        self.assertEqual(result["source"], "curated")

    def test_hosted_endpoint_validation_requires_https_except_loopback_opt_in(self):
        self.assertEqual(
            batch.validated_hosted_endpoint("https://example.invalid/v1/models"),
            "https://example.invalid/v1/models",
        )
        with mock.patch.dict(os.environ, {"NVIDIA_HOSTED_ALLOW_HTTP": "1"}):
            self.assertEqual(
                batch.validated_hosted_endpoint("http://127.0.0.1:18080/v1/models"),
                "http://127.0.0.1:18080/v1/models",
            )
            with self.assertRaises(RuntimeError):
                batch.validated_hosted_endpoint("http://example.invalid/v1/models")
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("NVIDIA_HOSTED_ALLOW_HTTP", None)
            with self.assertRaises(RuntimeError):
                batch.validated_hosted_endpoint("http://localhost:18080/v1/models")
        for invalid in (
            "https://user:pass@example.invalid/v1/models",
            "https://example.invalid:not-a-port/v1/models",
            "https://example.invalid/v1/models?token=value",
            "https://example.invalid/v1/models#fragment",
            "https://example.invalid/v1/ models",
        ):
            with self.assertRaises(RuntimeError):
                batch.validated_hosted_endpoint(invalid)

    def test_hosted_auth_scheme_rejects_header_injection(self):
        with mock.patch.object(batch, "NVIDIA_HOSTED_AUTH_SCHEME", "Bearer\r\nX-Evil"):
            with self.assertRaises(RuntimeError):
                batch.hosted_auth_headers("opaque-runtime-value-for-unit-test")

    def test_model_selection_rejects_non_video_and_unknown_custom_ids(self):
        with batch.STATE_LOCK:
            comparison = dict(batch.STATE["comparison"])
            comparison["catalog"] = [
                {"id": "video-ok", "supports_video": True},
                {"id": "text-only", "supports_video": False},
            ]
            batch.STATE["comparison"] = comparison
        self.assertEqual(batch.normalized_comparison_models(["video-ok"]), ["video-ok"])
        with self.assertRaises(batch.ClientInputError):
            batch.normalized_comparison_models(["text-only"])
        with self.assertRaises(batch.ClientInputError):
            batch.normalized_comparison_models(["not-in-catalog"])
        with self.assertRaises(batch.ClientInputError):
            batch.normalized_comparison_models(
                ["https://untrusted.invalid/model"],
                {"https://untrusted.invalid/model": "sampled_frames"},
            )
        self.assertEqual(
            batch.normalized_comparison_models(
                ["not-in-catalog"], {"not-in-catalog": "sampled_frames"}
            ),
            ["not-in-catalog"],
        )

    def test_id_only_confirmed_contracts_choose_explicit_media_strategy(self):
        cosmos = batch.hosted_model_capabilities("nvidia/cosmos3-super-reasoner", {"id": "nvidia/cosmos3-super-reasoner"})
        gemma = batch.hosted_model_capabilities("google/gemma-4-31b-it", {"id": "google/gemma-4-31b-it"})
        claude = batch.hosted_model_capabilities("anthropic/claude-opus-5", {"id": "anthropic/claude-opus-5"})
        qwen = batch.hosted_model_capabilities("qwen/qwen3.6-35b-a3b", {"id": "qwen/qwen3.6-35b-a3b"})
        nemotron_id = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
        nemotron = batch.hosted_model_capabilities(nemotron_id, {"id": nemotron_id})
        gemma_false_positive = batch.hosted_model_capabilities(
            "google/gemma-2-4b-it", {"id": "google/gemma-2-4b-it"}
        )
        claude_false_positive = batch.hosted_model_capabilities(
            "anthropic/claude-opus-4-5", {"id": "anthropic/claude-opus-4-5"}
        )
        self.assertEqual(cosmos["comparison_media_strategy"], "native_video")
        self.assertEqual(gemma["comparison_media_strategy"], "sampled_frames")
        self.assertEqual(claude["comparison_media_strategy"], "sampled_frames")
        self.assertEqual(qwen["comparison_media_strategy"], "sampled_frames")
        self.assertEqual(nemotron["comparison_media_strategy"], "native_video")
        self.assertIsNone(gemma_false_positive["comparison_media_strategy"])
        self.assertIsNone(claude_false_positive["comparison_media_strategy"])

    def test_sampled_frame_strategy_records_media_and_orders_images_first(self):
        captured = {}

        def fake_content(*args, **kwargs):
            self.assertTrue(kwargs.get("force_frames"))
            self.assertEqual(kwargs.get("backend"), "hosted_openai")
            self.assertEqual(args[5], 12)
            return ([
                {"type": "text", "text": "question"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AA=="}},
            ], {"mode": "image_frames", "frames_passed": 1})

        def fake_post(base_url, headers, payload, **kwargs):
            captured["content"] = payload["messages"][1]["content"]
            captured["payload"] = payload
            captured["allow_redirects"] = kwargs.get("allow_redirects")
            return {"text": "answer", "metrics": {"ttft_seconds": 0.1}}

        video = {"id": "v1", "name": "reference.mp4", "filepath": "/tmp/reference.mp4"}
        target = {
            "source": "hosted_endpoint", "label": "Hosted", "model": "image-model",
            "backend": "hosted_openai", "base_url": "https://example.invalid/v1/chat/completions",
            "headers": {"Authorization": "Bearer hidden"}, "media_strategy": "sampled_frames", "_runtime_secret": "",
        }
        variant = {
            "id": "spot", "label": "Current", "system_prompt": "private system prompt", "user_prompt": "private user question",
            "params": {
                "fps": 1, "max_pixels": 65536, "max_frames": 12, "max_tokens": 321,
                "temperature": 0.2, "top_p": 0.8, "repetition_penalty": 1.4,
                "prompt_mode": "runtime_form",
            },
        }
        with mock.patch.object(batch, "content_for_video", side_effect=fake_content), \
             mock.patch.object(batch, "post_chat_completion", side_effect=fake_post):
            result = batch.run_comparison_case(video, target, variant)
        self.assertEqual([item["type"] for item in captured["content"]], ["image_url", "text"])
        self.assertEqual(captured["payload"]["max_tokens"], 321)
        self.assertNotIn("nvext", captured["payload"])
        self.assertNotIn("repetition_penalty", captured["payload"])
        self.assertFalse(captured["allow_redirects"])
        self.assertEqual(result["media_strategy"], "sampled_frames")
        self.assertEqual(result["plan"]["media_strategy"], "sampled_frames")
        shape = result["request_shape"]
        self.assertEqual(shape["transport"], "openai_chat_completions")
        self.assertEqual(shape["parameters"]["max_tokens"], 321)
        self.assertEqual(shape["media"]["name"], "reference.mp4")
        self.assertEqual(shape["media"]["payload"], "omitted")
        encoded = json.dumps(result)
        self.assertNotIn("private system prompt", encoded)
        self.assertNotIn("private user question", encoded)
        self.assertNotIn("Bearer hidden", encoded)
        self.assertNotIn("AA==", encoded)

    def test_hosted_openai_sampled_frames_are_not_nim_clamped(self):
        captured = {}

        def fake_extract(path, fps, max_frames, max_pixels):
            captured["max_frames"] = max_frames
            return ["AA=="] * max_frames

        with mock.patch.object(batch, "get_video_meta", return_value={
            "width": 320, "height": 180, "duration_s": 12, "total_frames": 360,
        }), mock.patch.object(batch, "extract_frames_b64", side_effect=fake_extract):
            content, plan = batch.content_for_video(
                "/tmp/reference.mp4",
                "question",
                "vendor/image-model",
                1,
                65536,
                12,
                backend="hosted_openai",
                force_frames=True,
            )

        self.assertEqual(captured["max_frames"], 12)
        self.assertEqual(plan["frames_passed"], 12)
        self.assertEqual(sum(item["type"] == "image_url" for item in content), 12)

    def test_recorded_error_redacts_arbitrary_runtime_key_and_keeps_latency(self):
        runtime_key = "opaque-runtime-value-for-unit-test"
        video = {"id": "v1", "name": "reference.mp4", "source": "reference_upload"}
        target = {
            "source": "hosted_endpoint",
            "label": "Hosted endpoint",
            "model": "video-ok",
            "base_url": "https://private-host.invalid/v1/chat/completions",
            "_runtime_secret": runtime_key,
        }
        variant = {
            "id": "spot",
            "label": "Current workflow",
            "system_prompt": "private system prompt",
            "user_prompt": "private user question",
        }
        with mock.patch.object(
            batch,
            "run_comparison_case",
            side_effect=RuntimeError(
                f"service at private-host.invalid echoed {runtime_key}; private system prompt; private user question"
            ),
        ):
            result = batch.run_comparison_case_recorded(video, target, variant)
        encoded = json.dumps(result)
        self.assertNotIn(runtime_key, encoded)
        self.assertNotIn("private-host.invalid", encoded)
        self.assertNotIn("private system prompt", encoded)
        self.assertNotIn("private user question", encoded)
        self.assertEqual(result["status"], "error")
        self.assertIsInstance(result["latency_seconds"], float)
        self.assertIn("[REDACTED]", result["error"])
        self.assertIn("request_shape", result)

    def test_comparison_payload_rejects_runtime_and_secret_like_model_values(self):
        runtime_key = "opaque-runtime-value-for-unit-test"
        cases = [
            {"models": [runtime_key]},
            {"models": ["vendor/" + "sk-" + "unit-test-secret-value"]},
            {"models": [f"vendor/{runtime_key}/model"]},
            {"models": ["vendor/model"], "variants": [{"user_prompt": f"use {runtime_key}"}]},
        ]
        for payload in cases:
            with self.subTest(payload=payload):
                with self.assertRaises(batch.ClientInputError) as caught:
                    batch.validate_comparison_payload_secrets(payload, runtime_key)
                self.assertNotIn(runtime_key, str(caught.exception))

    def test_reference_upload_storage_is_private_and_aggregate_bounded(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(batch, "REFERENCE_VIDEO_DIR", Path(tmp) / "references"), \
             mock.patch.object(batch, "REFERENCE_VIDEO_MAX_BYTES", 10), \
             mock.patch.object(batch, "REFERENCE_VIDEO_TOTAL_MAX_BYTES", 12):
            stored = batch.store_reference_video(io.BytesIO(b"abcd"), 4, "reference.mp4")
            self.assertEqual(stored.read_bytes(), b"abcd")
            self.assertEqual(stat.S_IMODE(batch.REFERENCE_VIDEO_DIR.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(stored.stat().st_mode), 0o600)
            with self.assertRaises(batch.ClientInputError):
                batch.store_reference_video(io.BytesIO(b"123456789"), 9, "second.mp4")
            self.assertEqual(len(list(batch.REFERENCE_VIDEO_DIR.iterdir())), 1)

    def test_reference_upload_appends_without_replacing_dataset_videos(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "reference.mp4"
            source.write_bytes(b"not-a-real-video-but-nonempty")
            results_file = Path(tmp) / "state.json"
            with batch.STATE_LOCK:
                batch.STATE["videos"] = [{"id": "dataset-1", "name": "dataset.mp4", "source": "hf_hub"}]

            def fake_attach(video):
                video["meta"] = {"duration_s": 1.0}
                return video

            with mock.patch.object(batch, "attach_meta", side_effect=fake_attach), \
                 mock.patch.object(batch, "RESULTS_FILE", results_file):
                uploaded = batch.register_reference_video(source, "reference.mp4")

            ids = [item["id"] for item in batch.snapshot()["videos"]]
            self.assertIn("dataset-1", ids)
            self.assertIn(uploaded["id"], ids)
            self.assertTrue(results_file.exists())

    def test_comparison_persistence_never_contains_runtime_key(self):
        runtime_key = "opaque-runtime-value-for-unit-test"
        video = {"id": "v1", "name": "reference.mp4", "source": "reference_upload"}
        variant = {"id": "spot", "label": "Current workflow", "params": {}}
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(batch, "RESULTS_FILE", Path(tmp) / "state.json"), \
             mock.patch.object(batch, "EXPORT_DIR", Path(tmp) / "exports"), \
             mock.patch.object(batch, "selected_videos_for_ids", return_value=[video]), \
             mock.patch.object(batch, "run_comparison_case", side_effect=RuntimeError(f"echo {runtime_key}")):
            batch.run_hosted_comparison(
                ["v1"], ["video-ok"], {"video-ok": "sampled_frames"}, runtime_key, "spot", [variant], 1, False
            )
            persisted = batch.RESULTS_FILE.read_text(encoding="utf-8")
            persisted_mode = stat.S_IMODE(batch.RESULTS_FILE.stat().st_mode)
            report = batch.create_comparison_export("html").read_text(encoding="utf-8")
        state_text = json.dumps(batch.snapshot())
        self.assertNotIn(runtime_key, persisted)
        self.assertNotIn(runtime_key, state_text)
        self.assertNotIn(runtime_key, report)
        self.assertNotIn("inference-api.nvidia.com", report)
        self.assertEqual(persisted_mode, 0o600)
        self.assertIn("Media strategy", report)
        self.assertEqual(batch.snapshot()["comparison"]["run"]["status"], "complete")
        self.assertEqual(batch.snapshot()["comparison"]["results"][0]["status"], "error")
        self.assertEqual(batch.snapshot()["comparison"]["results"][0]["source"], "hosted_endpoint")
        self.assertIn("request_shape", batch.snapshot()["comparison"]["results"][0])

    def test_loaded_checkpoint_and_comparison_export_permissions_are_private(self):
        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(batch, "detect_server", return_value={
                 "model": str(Path(tmp) / "secret-checkpoint"),
                 "backend": "vllm",
                 "base_url": "http://localhost:8000/v1",
             }):
            target = batch.loaded_comparison_target()
        self.assertEqual(target["model"], "secret-checkpoint")

        with tempfile.TemporaryDirectory() as tmp, \
             mock.patch.object(batch, "EXPORT_DIR", Path(tmp) / "exports"):
            with batch.STATE_LOCK:
                comparison = dict(batch.STATE["comparison"])
                comparison.update({
                    "run": {"run_id": "test", "status": "complete", "summary": {"groups": []}},
                    "results": [{
                        "video_name": "reference.mp4",
                        "source": "hosted_endpoint",
                        "model": "vendor/model",
                        "status": "ok",
                        "response": "answer",
                    }],
                })
                batch.STATE["comparison"] = comparison
            export = batch.create_comparison_export("json")
            self.assertEqual(stat.S_IMODE(batch.EXPORT_DIR.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(export.stat().st_mode), 0o600)

    def test_curated_hints_do_not_invent_request_ids(self):
        curated = batch.curated_hosted_models()
        request_ids = {item["id"] for item in curated if item.get("id")}
        self.assertEqual(request_ids, set())
        kimi = next(item for item in curated if item["family"] == "Kimi")
        self.assertFalse(kimi.get("id"))
        self.assertIsNone(kimi.get("supports_video"))


if __name__ == "__main__":
    unittest.main()
