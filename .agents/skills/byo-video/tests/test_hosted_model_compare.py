#!/usr/bin/env python3
"""Focused offline tests for BYO-video hosted comparison helpers."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
import ast
from pathlib import Path
from unittest.mock import patch


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import hosted_model_compare as hmc  # noqa: E402


DUMMY_KEY = "unit-test-key-material-12345"


class FakeResponse:
    def __init__(self, payload, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self._payload


class HostedModelCompareTests(unittest.TestCase):
    def test_gradio_key_wiring_retains_only_masked_page_field_with_explicit_clear(self):
        def is_clear_update(node):
            return (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"
                and any(
                    keyword.arg == "value"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value == ""
                    for keyword in node.keywords
                )
            )

        gradio_source = (SCRIPTS_DIR / "gradio_cr2_byo.py").read_text(encoding="utf-8")
        tree = ast.parse(gradio_source)
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        discovery = functions["_hosted_discover_ui"]
        discovery_segment = ast.get_source_segment(gradio_source, discovery)
        self.assertNotIn('gr.update(value="")', discovery_segment)
        discovery_yields = sorted(
            (node for node in ast.walk(discovery) if isinstance(node, ast.Yield)),
            key=lambda node: node.lineno,
        )
        self.assertGreaterEqual(len(discovery_yields), 3)
        self.assertIsInstance(discovery_yields[0].value, ast.Tuple)
        self.assertEqual(len(discovery_yields[0].value.elts), 4)
        self.assertFalse(is_clear_update(discovery_yields[0].value.elts[-1]))
        comparison_segment = ast.get_source_segment(
            gradio_source,
            functions["_run_hosted_compare_ui"],
        )
        self.assertNotIn('gr.update(value="")', comparison_segment)
        comparison_yields = sorted(
            (
                node
                for node in ast.walk(functions["_run_hosted_compare_ui"])
                if isinstance(node, ast.Yield)
            ),
            key=lambda node: node.lineno,
        )
        for yielded in comparison_yields:
            if isinstance(yielded.value, ast.Tuple):
                self.assertFalse(is_clear_update(yielded.value.elts[-1]))

        clear_returns = [
            node
            for node in ast.walk(functions["_clear_hosted_runtime_key"])
            if isinstance(node, ast.Return)
        ]
        self.assertEqual(len(clear_returns), 1)
        self.assertIsInstance(clear_returns[0].value, ast.Constant)
        self.assertEqual(clear_returns[0].value.value, "")

        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if isinstance(call.func, ast.Attribute) and call.func.attr in {"State", "BrowserState"}:
                serialized = ast.dump(call)
                self.assertNotIn("hosted_runtime_key", serialized)

        discovery_click_found = False
        comparison_click_found = False
        clear_click_found = False
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            if not (isinstance(call.func, ast.Attribute) and call.func.attr == "click"):
                continue
            keywords = {item.arg: item.value for item in call.keywords if item.arg}
            fn = keywords.get("fn")
            outputs = keywords.get("outputs")
            if not isinstance(fn, ast.Name) or not isinstance(outputs, ast.List):
                continue
            output_names = [item.id for item in outputs.elts if isinstance(item, ast.Name)]
            if fn.id == "_hosted_discover_ui":
                discovery_click_found = True
                self.assertEqual(output_names[-1], "hosted_runtime_key")
            if fn.id == "_run_hosted_compare_ui":
                comparison_click_found = True
                self.assertEqual(output_names[-1], "hosted_runtime_key")
            if fn.id == "_clear_hosted_runtime_key":
                clear_click_found = True
                self.assertEqual(output_names, ["hosted_runtime_key"])
        self.assertTrue(discovery_click_found)
        self.assertTrue(comparison_click_found)
        self.assertTrue(clear_click_found)

    def test_vite_key_lives_in_app_memory_and_run_design_is_self_explanatory(self):
        repo_root = Path(__file__).resolve().parents[4]
        app_source = (repo_root / "apps/nvidia-build-reason-vite/src/App.tsx").read_text(encoding="utf-8")
        workbench = app_source[app_source.index("function ComparisonWorkbench"):]

        self.assertIn('const [hostedRuntimeKey, setHostedRuntimeKey] = useState("")', app_source)
        self.assertIn("runtimeKey={hostedRuntimeKey}", app_source)
        self.assertNotIn('const [runtimeKey, setRuntimeKey] = useState("")', workbench)
        self.assertNotIn('setRuntimeKey("")', app_source)
        self.assertIn('updateRuntimeKey("")', workbench)
        self.assertNotIn("localStorage", app_source)
        self.assertNotIn("sessionStorage", app_source)
        self.assertIn("Enter once for this page session", workbench)
        self.assertIn("refreshing or closing", workbench)
        self.assertIn("What this runs", workbench)
        self.assertIn("Planned model runs", workbench)
        self.assertIn('variantLabel: mode === "ablation" ? variantLabel : "Current settings"', workbench)
        self.assertIn('attempt.data.code === "INVALID_CATALOG_CONTEXT"', workbench)
        self.assertIn("refreshing models and retrying automatically", workbench)
        self.assertIn("runtimeKeyRef.current.trim() !== apiKey", workbench)
        self.assertIn("refreshed.capabilityToken, refreshed.models", workbench)
        self.assertGreaterEqual(workbench.count("sendComparison("), 3)

    def test_default_provider_base_uses_inference_api(self):
        config = hmc.hosted_provider_config({})
        self.assertEqual(config["models_url"], "https://inference-api.nvidia.com/v1/models")
        self.assertEqual(config["chat_url"], "https://inference-api.nvidia.com/v1/chat/completions")
        self.assertEqual(hmc.CURATED_FALLBACK_MODEL_IDS, ())
        self.assertTrue(all("/" not in label for label in hmc.CURATED_FALLBACK_LABELS))

    def test_provider_config_and_x_api_key_discovery_are_request_scoped(self):
        config = hmc.hosted_provider_config(
            {
                "BYO_VIDEO_HOSTED_API_BASE": "https://provider.invalid/v1",
                "BYO_VIDEO_HOSTED_AUTH_SCHEME": "x-api-key",
                "BYO_VIDEO_HOSTED_PROVIDER_LABEL": "Test provider",
            }
        )
        captured = {}

        def fake_get(url, **kwargs):
            captured["url"] = url
            captured["headers"] = kwargs["headers"]
            captured["allow_redirects"] = kwargs["allow_redirects"]
            return FakeResponse(
                {
                    "data": [
                        {
                            "id": "provider/gemini-catalog-id",
                            "owned_by": f"owner-{DUMMY_KEY}",
                            "capabilities": {"input_modalities": ["text", "image", "video"]},
                            "metadata": {"note": f"echo-{DUMMY_KEY}"},
                        },
                        {"id": "provider/qwen-catalog-id"},
                        {"id": DUMMY_KEY},
                        {"id": f"provider/{DUMMY_KEY}/model"},
                        {"id": "https://untrusted.invalid/model"},
                    ]
                }
            )

        models = hmc.discover_hosted_models(
            DUMMY_KEY,
            provider_config=config,
            http_get=fake_get,
        )
        self.assertEqual(captured["url"], "https://provider.invalid/v1/models")
        self.assertEqual(captured["headers"]["X-API-Key"], DUMMY_KEY)
        self.assertFalse(captured.get("allow_redirects", True))
        self.assertNotIn(DUMMY_KEY, json.dumps(models))
        by_id = {item["id"]: item for item in models}
        self.assertEqual(
            by_id["provider/gemini-catalog-id"]["input_modalities"],
            ["text", "image", "video"],
        )
        self.assertEqual(by_id["provider/qwen-catalog-id"]["input_modalities"], ["text"])

    def test_curated_selection_uses_discovered_ids_without_inventing_slugs(self):
        discovered = [
            {"id": "portal/qwen-exact-a"},
            {"id": "portal/nemotron-exact-b"},
            {"id": "portal/claude-exact-c"},
            {"id": "portal/gpt-exact-d"},
            {"id": "portal/kimi-exact-e"},
        ]
        selected, family_map = hmc.curated_model_selection(discovered)
        self.assertEqual(
            selected,
            [
                "portal/qwen-exact-a",
                "portal/nemotron-exact-b",
                "portal/claude-exact-c",
                "portal/gpt-exact-d",
                "portal/kimi-exact-e",
            ],
        )
        self.assertEqual(family_map["Gemini"], [])

    def test_curated_selection_prefers_requested_frontier_versions_over_parameter_counts(self):
        discovered = [
            {"id": "nvidia/openai/gpt-oss-120b"},
            {"id": "openai/openai/gpt-5.6-sol"},
            {"id": "nvidia/nvidia/nemotron-3-super-120b-long-ctx"},
            {"id": "nvidia/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"},
            {"id": "nvidia/qwen/qwen3-5-397b-a17b"},
            {"id": "nvidia/qwen/qwen3.6-35b-a3b"},
        ]
        selected, family_map = hmc.curated_model_selection(discovered)
        self.assertIn("openai/openai/gpt-5.6-sol", selected)
        self.assertIn("nvidia/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", selected)
        self.assertIn("nvidia/qwen/qwen3.6-35b-a3b", selected)
        self.assertEqual(family_map["GPT"][0], "openai/openai/gpt-5.6-sol")

    def test_only_confirmed_contracts_imply_native_video(self):
        self.assertEqual(
            hmc.infer_input_modalities("provider/cosmos3-nano-reasoner"),
            ["text", "image", "video"],
        )
        self.assertEqual(
            hmc.infer_input_modalities("some-provider/qwen-new-vl"),
            ["text"],
        )
        self.assertEqual(
            hmc.infer_input_modalities("provider/gemma-2-4b-it"),
            ["text"],
        )
        self.assertEqual(
            hmc.infer_input_modalities("provider/claude-opus-4-5"),
            ["text"],
        )
        self.assertIn(
            "video",
            hmc.infer_input_modalities("provider/catalog-video-model"),
        )

    def test_auto_and_explicit_media_strategy_resolution(self):
        strategy, modalities, source = hmc.resolve_media_strategy(
            "provider/cosmos3-super-reasoner",
            False,
            "Auto capability-gated",
        )
        self.assertEqual(strategy, "native_video_url")
        self.assertEqual(modalities, ["text", "image", "video"])
        self.assertEqual(source, "catalog_or_confirmed_tag")

        native, _, native_source = hmc.resolve_media_strategy(
            "provider/catalog-video-model",
            False,
            "Auto capability-gated",
        )
        self.assertEqual(native, "native_video_url")
        self.assertEqual(native_source, "catalog_or_confirmed_tag")

        blocked, _, blocked_source = hmc.resolve_media_strategy(
            "provider/unknown-model",
            False,
            "Auto capability-gated",
        )
        self.assertIsNone(blocked)
        self.assertEqual(blocked_source, "capability_unconfirmed")

        forced, _, forced_source = hmc.resolve_media_strategy(
            "provider/unknown-model",
            False,
            "Force sampled image frames (custom contract)",
        )
        self.assertEqual(forced, "sampled_image_frames")
        self.assertEqual(forced_source, "explicit_user_override")

    def test_unknown_video_capability_is_reported_without_calling_endpoint(self):
        calls = []

        def must_not_run(*args, **kwargs):
            calls.append((args, kwargs))
            raise AssertionError("endpoint should not be called")

        with tempfile.TemporaryDirectory() as temp_dir:
            media = os.path.join(temp_dir, "reference.mp4")
            with open(media, "wb") as handle:
                handle.write(b"not-a-real-video-needed-for-shape-test")
            results = hmc.run_hosted_comparisons(
                DUMMY_KEY,
                ["provider/unknown-frontier-model"],
                [{"name": "spot", "prompt": "Describe it", "system_prompt": "Be precise"}],
                media,
                False,
                request_fn=must_not_run,
            )
        self.assertEqual(calls, [])
        self.assertEqual(results[0]["status"], "unsupported_media")
        self.assertTrue(results[0]["request_shape"]["not_sent"])

    def test_credential_or_malformed_value_cannot_be_used_as_model_id(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            media = os.path.join(temp_dir, "reference.jpg")
            with open(media, "wb") as handle:
                handle.write(b"jpeg-fixture")
            for model_id in (
                DUMMY_KEY,
                f"provider/{DUMMY_KEY}/model",
                "https://untrusted.invalid/model",
                "model id with spaces",
            ):
                with self.subTest(model_id=model_id):
                    with self.assertRaisesRegex(hmc.HostedCompareError, "model ID is invalid"):
                        hmc.run_hosted_comparisons(
                            DUMMY_KEY,
                            [model_id],
                            [{"name": "spot", "prompt": "Describe", "system_prompt": ""}],
                            media,
                            True,
                        )

        with self.assertRaisesRegex(hmc.HostedCompareError, "model ID is invalid"):
            hmc.invoke_hosted_model(
                DUMMY_KEY,
                f"provider/{DUMMY_KEY}/model",
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
                {"kind": "image", "filename": "reference.jpg"},
                "Describe",
                "",
                http_post=lambda *_args, **_kwargs: self.fail("network must not run"),
            )

    def test_supported_matrix_redacts_key_and_media_from_return_value(self):
        def fake_invoke(
            api_key,
            model_id,
            media_item,
            media_descriptor,
            prompt,
            system_prompt,
            **kwargs,
        ):
            return {
                "status": "ok",
                "latency_s": 0.125,
                "output": f"response accidentally echoed {api_key}",
                "error": "",
                "usage": {"completion_tokens": 4},
                "request_shape": {
                    "Authorization": api_key,
                    "media": media_item,
                    "model": model_id,
                },
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            media = os.path.join(temp_dir, "reference.jpg")
            with open(media, "wb") as handle:
                handle.write(b"jpeg-fixture")
            results = hmc.run_hosted_comparisons(
                DUMMY_KEY,
                ["portal/catalog-confirmed-model"],
                [{"name": "spot", "prompt": "Describe it", "system_prompt": "Be precise"}],
                media,
                True,
                catalog_modalities={
                    "portal/catalog-confirmed-model": ["text", "image"]
                },
                request_fn=fake_invoke,
            )
            json_path, report_path, document = hmc.write_redacted_exports(
                results,
                mode="Spot comparison",
                media_descriptor=hmc.describe_media_source(media, True),
                output_dir=temp_dir,
            )
            json_text = Path(json_path).read_text(encoding="utf-8")
            report_text = Path(report_path).read_text(encoding="utf-8")

        serialized = json.dumps(document)
        for text in (serialized, json_text, report_text):
            self.assertNotIn(DUMMY_KEY, text)
            self.assertNotIn("anBlZy1maXh0dXJl", text)
        self.assertEqual(results[0]["status"], "ok")
        self.assertEqual(results[0]["media_adaptation"]["strategy"], "image_url")
        self.assertEqual(
            results[0]["request_shape"]["media_adaptation"]["sent_as"],
            "image_url",
        )

    def test_media_sources_are_upload_scoped_and_remote_sources_are_rejected(self):
        for source in ("https://example.invalid/reference.mp4", "file:///etc/hosts", "/etc/hosts"):
            with self.subTest(source=source):
                with self.assertRaises(hmc.HostedCompareError):
                    hmc.describe_media_source(source, False)

    def test_endpoint_urls_require_https_unless_local_test_opt_in(self):
        with self.assertRaises(hmc.HostedCompareError):
            hmc.hosted_provider_config({"BYO_VIDEO_HOSTED_API_BASE": "http://127.0.0.1:18080/v1"})
        config = hmc.hosted_provider_config(
            {
                "BYO_VIDEO_HOSTED_API_BASE": "http://127.0.0.1:18080/v1",
                "BYO_VIDEO_HOSTED_ALLOW_HTTP": "1",
            }
        )
        self.assertEqual(config["chat_url"], "http://127.0.0.1:18080/v1/chat/completions")
        with self.assertRaises(hmc.HostedCompareError):
            hmc.hosted_provider_config(
                {
                    "BYO_VIDEO_HOSTED_API_BASE": "http://untrusted.invalid/v1",
                    "BYO_VIDEO_HOSTED_ALLOW_HTTP": "1",
                }
            )
        with self.assertRaises(hmc.HostedCompareError):
            hmc.hosted_provider_config(
                {
                    "BYO_VIDEO_HOSTED_API_BASE": "https://provider.invalid/v1",
                    "BYO_VIDEO_HOSTED_AUTH_HEADER": "Authorization\r\nX-Leak",
                }
            )

    def test_discovery_refuses_redirects_without_following_them(self):
        captured = {}

        def fake_get(_url, **kwargs):
            captured.update(kwargs)
            return FakeResponse({}, status_code=302, text="redirect")

        with self.assertRaisesRegex(hmc.HostedCompareError, "refused an endpoint redirect"):
            hmc.discover_hosted_models(DUMMY_KEY, http_get=fake_get)
        self.assertFalse(captured["allow_redirects"])

    def test_ablation_template_expansion_and_limits(self):
        variants = hmc.parse_ablation_variants(
            "Workflow ablation",
            "BASE USER",
            "BASE SYSTEM",
            json.dumps(
                [
                    {"name": "full", "prompt": "{prompt}", "system_prompt": "{system}"},
                    {"name": "direct", "prompt": "Direct: {prompt}", "system_prompt": ""},
                ]
            ),
        )
        self.assertEqual(variants[0]["system_prompt"], "BASE SYSTEM")
        self.assertEqual(variants[1]["prompt"], "Direct: BASE USER")

    def test_payload_is_media_first_and_report_shape_has_no_host(self):
        media_item = {
            "type": "video_url",
            "video_url": {"url": "data:video/mp4;base64,QUJDRA=="},
        }
        payload = hmc.build_chat_payload(
            "nvidia/cosmos3-super-reasoner",
            media_item,
            "What happens?",
            "Analyze carefully",
        )
        content = payload["messages"][-1]["content"]
        self.assertEqual(content[0]["type"], "video_url")
        self.assertEqual(content[1], {"type": "text", "text": "What happens?"})
        shape = hmc.redacted_request_shape(
            payload,
            {"kind": "video", "filename": "reference.mp4"},
            hmc.hosted_provider_config(
                {
                    "BYO_VIDEO_HOSTED_API_BASE": "https://private.invalid/v1",
                    "BYO_VIDEO_HOSTED_PROVIDER_LABEL": "Configured provider",
                }
            ),
        )
        serialized = json.dumps(shape)
        self.assertNotIn("private.invalid", serialized)
        self.assertNotIn("QUJDRA==", serialized)
        self.assertNotIn("What happens?", serialized)
        self.assertNotIn("Analyze carefully", serialized)
        self.assertNotIn("Authorization", serialized)
        self.assertNotIn("Configured provider", serialized)
        self.assertEqual(shape["provider"], "Hosted endpoint")
        self.assertEqual(shape["path"], "/v1/chat/completions")
        self.assertEqual(shape["parameters_sent"]["max_tokens"], 512)

    def test_loaded_shape_distinguishes_sent_parameters_from_ui_settings(self):
        shape = hmc.build_loaded_request_shape(
            "nvidia/cosmos3-nano-reasoner",
            "nim_local",
            {"kind": "video", "filename": "reference.mp4"},
            "What happens?",
            "Be concise.",
            fps=2,
            max_pixels=524288,
            temperature=0.6,
            top_p=0.9,
            repetition_penalty=1.1,
            max_tokens=321,
        )
        self.assertEqual(shape["parameters_sent"]["temperature"], 0.6)
        self.assertEqual(shape["parameters_sent"]["repetition_penalty_location"], "nvext")
        self.assertEqual(shape["parameters_sent"]["max_tokens"], "omitted_by_loaded_endpoint_contract")
        self.assertEqual(shape["settings_snapshot"]["max_tokens"], 321)

    def test_invoke_sends_token_limit_and_redacts_configured_host_from_output(self):
        captured = {}
        config = hmc.hosted_provider_config(
            {
                "BYO_VIDEO_HOSTED_API_BASE": "https://private.invalid/v1",
                "BYO_VIDEO_HOSTED_PROVIDER_LABEL": "Configured provider",
            }
        )

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["payload"] = kwargs["json"]
            captured["headers"] = kwargs["headers"]
            captured["allow_redirects"] = kwargs["allow_redirects"]
            return FakeResponse(
                {
                    "choices": [
                        {
                            "message": {
                                "content": f"result from private.invalid accidentally echoed {DUMMY_KEY}"
                            }
                        }
                    ],
                    "usage": {"completion_tokens": 3},
                }
            )

        outcome = hmc.invoke_hosted_model(
            DUMMY_KEY,
            "provider/model",
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
            {"kind": "image", "filename": "reference.jpg"},
            "RAW USER PROMPT",
            "RAW SYSTEM PROMPT",
            max_tokens=321,
            provider_config=config,
            http_post=fake_post,
        )
        self.assertEqual(captured["payload"]["max_tokens"], 321)
        self.assertFalse(captured["allow_redirects"])
        self.assertIn(DUMMY_KEY, captured["headers"]["Authorization"])
        serialized = json.dumps(outcome)
        self.assertNotIn(DUMMY_KEY, serialized)
        self.assertNotIn("private.invalid", serialized)
        self.assertNotIn("RAW USER PROMPT", serialized)
        self.assertNotIn("RAW SYSTEM PROMPT", serialized)
        self.assertEqual(outcome["request_shape"]["parameters_sent"]["max_tokens"], 321)

    def test_http_error_echo_cannot_put_raw_prompts_in_report_fields(self):
        def fake_post(_url, **_kwargs):
            return FakeResponse(
                {},
                status_code=400,
                text="bad request included RAW USER PROMPT and RAW SYSTEM PROMPT",
            )

        outcome = hmc.invoke_hosted_model(
            DUMMY_KEY,
            "provider/model",
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,QUJD"}},
            {"kind": "image", "filename": "reference.jpg"},
            "RAW USER PROMPT",
            "RAW SYSTEM PROMPT",
            http_post=fake_post,
        )
        serialized = json.dumps(outcome)
        self.assertEqual(outcome["status"], "error")
        self.assertNotIn("RAW USER PROMPT", serialized)
        self.assertNotIn("RAW SYSTEM PROMPT", serialized)

    def test_runtime_key_in_variant_is_rejected_before_media_or_network(self):
        def network_must_not_run(**_kwargs):
            self.fail("network must not run")

        with self.assertRaises(hmc.HostedCompareError) as caught:
            hmc.run_hosted_comparisons(
                DUMMY_KEY,
                ["provider/model"],
                [
                    {
                        "id": "unsafe",
                        "label": "Unsafe",
                        "prompt": f"analyze {DUMMY_KEY}",
                        "system_prompt": "",
                    }
                ],
                "/media/must-not-be-read.mp4",
                False,
                request_fn=network_must_not_run,
            )
        self.assertNotIn(DUMMY_KEY, str(caught.exception))

    def test_export_collapses_absolute_checkpoint_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint = os.path.join(temp_dir, "private-checkpoint")
            os.mkdir(checkpoint)
            result = {
                "backend": "loaded_model",
                "model": checkpoint,
                "variant": "spot",
                "status": "ok",
                "latency_s": 0.1,
                "output": "answer",
                "error": "",
                "request_shape": {"model": checkpoint},
            }
            json_path, report_path, document = hmc.write_redacted_exports(
                [result],
                mode="Spot comparison",
                media_descriptor={"kind": "video", "filename": "reference.mp4"},
                output_dir=temp_dir,
            )
            serialized = "\n".join(
                (
                    json.dumps(document),
                    Path(json_path).read_text(encoding="utf-8"),
                    Path(report_path).read_text(encoding="utf-8"),
                )
            )
        self.assertNotIn(checkpoint, serialized)
        self.assertIn("private-checkpoint", serialized)

    def test_runtime_endpoint_override_is_validated_before_key_use(self):
        with self.assertRaises(hmc.HostedCompareError):
            hmc.discover_hosted_models(
                DUMMY_KEY,
                endpoint="http://untrusted.invalid/v1/models",
                http_get=lambda *_args, **_kwargs: self.fail("network must not run"),
            )

        with patch.dict(os.environ, {"BYO_VIDEO_HOSTED_ALLOW_HTTP": "1"}):
            config = hmc.hosted_provider_config(
                {
                    "BYO_VIDEO_HOSTED_API_BASE": "http://127.0.0.1:18080/v1",
                    "BYO_VIDEO_HOSTED_ALLOW_HTTP": "1",
                }
            )
            self.assertEqual(config["auth_header"], "Authorization")

    def test_gradio_comparison_source_guards_and_token_forwarding(self):
        gradio_source = (SCRIPTS_DIR / "gradio_cr2_byo.py").read_text(encoding="utf-8")
        tree = ast.parse(gradio_source)
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        }
        resolver_text = ast.get_source_segment(gradio_source, functions["_resolve_compare_media"])
        self.assertNotIn("_resolve_server_media_source", resolver_text)
        self.assertIn("frontend uploads only", resolver_text)
        baseline_text = ast.get_source_segment(gradio_source, functions["_baseline_failed"])
        self.assertIn("[alpamayo error", baseline_text)
        self.assertIn("adapter not responding", baseline_text)

        comparison = functions["_run_hosted_compare_ui"]
        comparison_text = ast.get_source_segment(gradio_source, comparison)
        self.assertLess(
            comparison_text.index("if key in workflow_fields"),
            comparison_text.index("_run_loaded_baseline_matrix("),
        )
        matrix_calls = [
            node
            for node in ast.walk(comparison)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_run_hosted_endpoint_matrix"
        ]
        self.assertEqual(len(matrix_calls), 1)
        keywords = {item.arg: item.value for item in matrix_calls[0].keywords if item.arg}
        self.assertIn("max_tokens", keywords)
        self.assertIn("max_new_tokens", ast.unparse(keywords["max_tokens"]))

        for function_name in ("_hosted_discover_ui", "_run_hosted_compare_ui"):
            function_text = ast.get_source_segment(gradio_source, functions[function_name])
            copy_index = function_text.index("key = str(runtime_key")
            clear_index = function_text.index('runtime_key = ""', copy_index)
            first_yield_index = function_text.index("yield", clear_index)
            self.assertLess(clear_index, first_yield_index)


if __name__ == "__main__":
    unittest.main()
