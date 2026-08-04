#!/usr/bin/env python3
"""Secret-safe helpers for BYO-video hosted endpoint comparisons.

This module deliberately has no Gradio dependency.  The generic Gradio app,
tests, and future frontends can all use the same model discovery, multimodal
request shaping, workflow-ablation parsing, and redacted report export code.

Runtime API keys are accepted only as function arguments.  They are never
stored in module globals, written to disk, included in return values, or sent
to a logger.  Callers should use a password input and avoid placing the key in
Gradio State.
"""

from __future__ import annotations

import base64
import copy
import datetime as _dt
import hashlib
import html
import json
import mimetypes
import os
import re
import tempfile
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence


DEFAULT_API_BASE = "https://inference-api.nvidia.com/v1"
KEY_MANAGEMENT_URL = "https://inference.nvidia.com/key-management"

MAX_SELECTED_MODELS = 8
MAX_ABLATION_VARIANTS = 8
MAX_COMPARISON_RUNS = 48

# Authenticated /v1/models discovery is authoritative.  PDF model strings are
# labels, not callable provider-prefixed IDs, so there are deliberately no
# preselected endpoint IDs before discovery.
CURATED_FALLBACK_MODEL_IDS: tuple[str, ...] = ()
CURATED_FALLBACK_LABELS = (
    "cosmos3-super-reasoner",
    "cosmos3-nano-reasoner",
    "gemma-4-31b-it",
)

# Exact IDs and modalities confirmed by the supplied comparison catalogs.  Do
# not grow this map from model-name intuition: unknown image/video capability
# is blocked until authenticated discovery explicitly declares it.
AUTHORITATIVE_STATIC_MODALITIES: dict[str, tuple[str, ...]] = {}

# Family/version tags are capability rules only, never endpoint IDs. They are
# applied to an exact ID returned by authenticated discovery. The supplied PDF
# establishes image multimodality; native video is enabled only for model
# contracts separately confirmed by their Build/NIM request shape.
_CATALOG_TAG_MEDIA_PROFILES = (
    (re.compile(r"(?i)(?:^|/)cosmos3[-_.]?(?:nano|super)[-_.]?reasoner(?:$|[-_.:])"), ("text", "image", "video"), "native_video_url"),
    (re.compile(r"(?i)(?:^|/)nemotron[-_.]?3[-_.]?nano[-_.]?omni[-_.]?30b[-_.]?a3b[-_.]?reasoning(?:$|[-_.:])"), ("text", "image", "video"), "native_video_url"),
    (re.compile(r"(?i)(?:^|/)qwen3(?:[._-]?5[-_.]?397b[-_.]?a17b|[._-]?6[-_.]?35b[-_.]?a3b)(?:$|[-_.:])"), ("text", "image"), "sampled_image_frames"),
    (re.compile(r"(?i)(?:^|/)gemma[-_.]?4[-_.]?31b[-_.]?it(?:$|[-_.:])"), ("text", "image"), "sampled_image_frames"),
    (re.compile(r"(?i)(?:^|/)gemini[-_.]?3[._-]?6[-_.]?flash(?:$|[-_.:])"), ("text", "image"), "sampled_image_frames"),
    (re.compile(r"(?i)(?:^|/)claude[-_.]?opus[-_.]?(?:5|4[-_.]?8)(?:$|[-_.:])"), ("text", "image"), "sampled_image_frames"),
    (re.compile(r"(?i)(?:^|/)gpt[-_.]?5[._-]?6[-_.]?sol(?:$|[-_.:])"), ("text", "image"), "sampled_image_frames"),
    (re.compile(r"(?i)(?:^|/)kimi[-_.]?k2[._-]?6(?:$|[-_.:])"), ("text", "image"), "sampled_image_frames"),
)
_VIDEO_CAPABILITY_TAG_RE = re.compile(
    r"(?i)(?:^|[/_.-])(?:video|video_url|video-input)(?:$|[/_.-])"
)

CURATED_FAMILIES = (
    ("Cosmos", ("cosmos",)),
    ("Qwen", ("qwen",)),
    ("Nemotron", ("nemotron",)),
    ("Gemma", ("gemma",)),
    ("Gemini", ("gemini",)),
    ("Claude", ("claude",)),
    ("GPT", ("gpt",)),
    ("Kimi", ("kimi",)),
)

# Prefer the catalog versions requested by this workflow before falling back to
# a generic family rank.  This prevents parameter counts in unrelated routes
# (for example GPT-OSS 120B) from outranking a newer frontier-version label.
_CURATED_FAMILY_PREFERENCES = {
    "Cosmos": (re.compile(r"(?i)cosmos3[-_/].*(?:super|nano).*reason"),),
    "Qwen": (
        re.compile(r"(?i)qwen3[._-]?6.*35b.*a3b"),
        re.compile(r"(?i)qwen3(?:[._-]?5|-5).*397b.*a17b"),
    ),
    "Nemotron": (re.compile(r"(?i)nemotron[-_/].*3.*nano.*omni"),),
    "Gemma": (re.compile(r"(?i)gemma[-_/].*4.*31b"),),
    "Gemini": (re.compile(r"(?i)gemini[-_/].*3[._-]?6.*flash"),),
    "Claude": (
        re.compile(r"(?i)claude[-_/].*opus[-_/]?5(?:$|[-_/])"),
        re.compile(r"(?i)claude[-_/].*sonnet[-_/]?5(?:$|[-_/])"),
    ),
    "GPT": (re.compile(r"(?i)gpt[-_/]?5[._-]?6[-_/]?sol(?:$|[-_/])"),),
    "Kimi": (re.compile(r"(?i)kimi[-_/]?k2[._-]?6(?:$|[-_/])"),),
}

DEFAULT_ABLATION_SPEC = json.dumps(
    [
        {
            "name": "full_workflow",
            "system_prompt": "{system}",
            "prompt": "{prompt}",
        },
        {
            "name": "no_system_prompt",
            "system_prompt": "",
            "prompt": "{prompt}",
        },
    ],
    indent=2,
)


class HostedCompareError(RuntimeError):
    """A user-actionable hosted comparison error with no secret material."""


_SECRET_FIELD_NAMES = {
    "authorization",
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "password",
    "secret",
    "credential",
    "credentials",
}
_TOKEN_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\b(?:nvapi-|sk-|hf_)[A-Za-z0-9._~+/=-]{6,}"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}"),
)
_DATA_URL_RE = re.compile(
    r"(?P<prefix>data:[^;,\s]+(?:;[^,\s]+)*;base64,)(?P<data>[A-Za-z0-9+/=]+)",
    re.IGNORECASE,
)
_SENSITIVE_QUERY_RE = re.compile(
    r"(?i)([?&](?:api[_-]?key|key|token|access_token|signature|sig)=)[^&#\s]+"
)
_MODEL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/+-]{0,239}$")


def _is_secret_field(name: Any) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(name or "").lower()).strip("_")
    return normalized in _SECRET_FIELD_NAMES or normalized.endswith("_api_key")


def _redact_string(value: str, secrets: Sequence[str] = ()) -> str:
    text = str(value)
    for secret in secrets:
        if secret:
            text = text.replace(str(secret), "<redacted>")
    text = _DATA_URL_RE.sub(
        lambda match: (
            f"{match.group('prefix')}<redacted-media:{len(match.group('data'))} base64 chars>"
        ),
        text,
    )
    text = _SENSITIVE_QUERY_RE.sub(r"\1<redacted>", text)
    for pattern in _TOKEN_PATTERNS:
        text = pattern.sub("<redacted>", text)
    text = re.sub(
        r"/(?:Users|home|tmp|var/tmp|workspace)/[^\s\"'<>]+",
        "<redacted-path>",
        text,
    )
    text = re.sub(r"[A-Za-z]:\\[^\r\n\"'<>]+", "<redacted-path>", text)
    return text


def _redact_configured_urls(value: Any, urls: Sequence[str]) -> str:
    """Remove configured provider hosts from user-facing errors and exports."""

    text = str(value or "")
    replacements = set()
    for url in urls:
        raw = str(url or "").strip()
        if not raw:
            continue
        replacements.add(raw)
        try:
            parsed = urllib.parse.urlparse(raw)
            if parsed.scheme and parsed.netloc:
                replacements.add(f"{parsed.scheme}://{parsed.netloc}")
            if parsed.netloc:
                replacements.add(parsed.netloc)
            if parsed.hostname:
                replacements.add(parsed.hostname)
        except Exception:
            pass
    for raw in sorted(replacements, key=len, reverse=True):
        text = text.replace(raw, "<configured-endpoint>")
    return text


def redact_report_text(
    value: Any,
    secrets: Sequence[str] = (),
    configured_values: Sequence[str] = (),
) -> str:
    """Redact credentials plus configured endpoints/paths from report text."""

    return str(
        redact_sensitive(
            _redact_configured_urls(value, configured_values),
            secrets,
        )
    )


def report_safe_model_label(value: Any) -> str:
    """Keep callable IDs while collapsing filesystem checkpoints to basenames."""

    raw = str(value or "").strip()
    if not raw:
        return ""
    path_value = raw.removeprefix("file://")
    expanded = os.path.abspath(os.path.expanduser(path_value))
    if raw.startswith("file://") or os.path.isabs(path_value) or os.path.exists(expanded):
        return os.path.basename(path_value.rstrip(os.sep)) or "loaded-checkpoint"
    return str(redact_sensitive(raw))


def _validated_endpoint_url(value: str, label: str, env: Mapping[str, str]) -> str:
    """Require credential-bearing endpoint configuration to use HTTPS."""

    raw = str(value or "").strip()
    try:
        parsed = urllib.parse.urlparse(raw)
    except Exception as exc:
        raise HostedCompareError(f"The configured {label} URL is invalid.") from exc
    allow_http = str(env.get("BYO_VIDEO_HOSTED_ALLOW_HTTP") or "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    loopback_host = (parsed.hostname or "").lower() in {"localhost", "127.0.0.1", "::1"}
    if not parsed.hostname or parsed.username or parsed.password:
        raise HostedCompareError(f"The configured {label} URL is invalid.")
    if parsed.scheme != "https" and not (
        allow_http and parsed.scheme == "http" and loopback_host
    ):
        raise HostedCompareError(f"The configured {label} URL must use HTTPS.")
    if parsed.query or parsed.fragment:
        raise HostedCompareError(f"The configured {label} URL must not contain a query or fragment.")
    return raw.rstrip("/")


def redact_sensitive(value: Any, secrets: Sequence[str] = ()) -> Any:
    """Return a recursively redacted copy suitable for UI or disk export.

    ``secrets`` lets the caller remove an exact runtime key even when it does
    not use a familiar prefix.  It is used only during this call and is never
    retained.
    """

    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            out[key] = "<redacted>" if _is_secret_field(key) else redact_sensitive(item, secrets)
        return out
    if isinstance(value, list):
        return [redact_sensitive(item, secrets) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive(item, secrets) for item in value)
    if isinstance(value, str):
        return _redact_string(value, secrets)
    return copy.deepcopy(value)


def validate_runtime_key(api_key: str) -> str:
    """Validate without logging, caching, or embedding the key in an error."""

    key = str(api_key or "").strip()
    if not key:
        raise HostedCompareError(
            f"Enter a runtime NVIDIA API key from {KEY_MANAGEMENT_URL}."
        )
    if len(key) < 12 or any(char.isspace() for char in key):
        raise HostedCompareError("The runtime API key format is not valid.")
    return key


def validate_hosted_model_id(model_id: Any, api_key: str = "") -> str:
    """Accept only bounded catalog-style IDs and never a pasted credential."""

    value = str(model_id or "").strip()
    key = str(api_key or "")
    if (
        not value
        or (key and key in value)
        or "://" in value
        or not _MODEL_ID_RE.fullmatch(value)
        or any(pattern.search(value) for pattern in _TOKEN_PATTERNS)
    ):
        raise HostedCompareError("A selected hosted model ID is invalid.")
    return value


def hosted_provider_config(env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Resolve an OpenAI-compatible provider without exposing it in reports.

    Supported environment variables:

    - ``BYO_VIDEO_HOSTED_API_BASE``: base ending in ``/v1``
    - ``BYO_VIDEO_HOSTED_MODELS_URL``: full catalog URL override
    - ``BYO_VIDEO_HOSTED_CHAT_URL``: full chat-completions URL override
    - ``BYO_VIDEO_HOSTED_AUTH_SCHEME``: ``bearer`` (default) or ``x-api-key``
    - ``BYO_VIDEO_HOSTED_AUTH_HEADER``: custom header name override
    - ``BYO_VIDEO_HOSTED_PROVIDER_LABEL``: safe user-facing provider label

    URLs and custom labels remain process configuration. Reports use a fixed
    neutral provider label plus relative request paths.
    """

    source = os.environ if env is None else env
    base = str(source.get("BYO_VIDEO_HOSTED_API_BASE") or DEFAULT_API_BASE).rstrip("/")
    models_url = _validated_endpoint_url(
        str(source.get("BYO_VIDEO_HOSTED_MODELS_URL") or f"{base}/models"),
        "model catalog",
        source,
    )
    chat_url = _validated_endpoint_url(
        str(source.get("BYO_VIDEO_HOSTED_CHAT_URL") or f"{base}/chat/completions"),
        "chat completion",
        source,
    )
    scheme = str(source.get("BYO_VIDEO_HOSTED_AUTH_SCHEME") or "bearer").strip().lower()
    if scheme not in ("bearer", "x-api-key", "x_api_key", "apikey"):
        raise HostedCompareError(
            "BYO_VIDEO_HOSTED_AUTH_SCHEME must be 'bearer' or 'x-api-key'."
        )
    default_header = "Authorization" if scheme == "bearer" else "X-API-Key"
    auth_header = str(source.get("BYO_VIDEO_HOSTED_AUTH_HEADER") or default_header).strip()
    if not re.fullmatch(r"[A-Za-z0-9-]+", auth_header):
        raise HostedCompareError("The hosted auth header name is invalid.")
    provider_label = str(source.get("BYO_VIDEO_HOSTED_PROVIDER_LABEL") or "Hosted endpoint").strip()
    provider_label = re.sub(r"[\r\n\t]+", " ", provider_label)[:80] or "Hosted endpoint"
    if "://" in provider_label:
        provider_label = "Configured hosted endpoint"
    return {
        "models_url": models_url,
        "chat_url": chat_url,
        "auth_scheme": "bearer" if scheme == "bearer" else "x-api-key",
        "auth_header": auth_header,
        "provider_label": provider_label,
    }


def _validated_provider_config(
    provider_config: Mapping[str, str] | None,
) -> dict[str, str]:
    """Validate caller-supplied normalized configuration before using a key."""

    if provider_config is None:
        return hosted_provider_config()
    raw = dict(provider_config)
    models_url = _validated_endpoint_url(
        str(raw.get("models_url") or ""), "model catalog", os.environ
    )
    chat_url = _validated_endpoint_url(
        str(raw.get("chat_url") or ""), "chat completion", os.environ
    )
    scheme = str(raw.get("auth_scheme") or "bearer").strip().lower()
    if scheme not in ("bearer", "x-api-key", "x_api_key", "apikey"):
        raise HostedCompareError("The hosted authentication scheme is invalid.")
    default_header = "Authorization" if scheme == "bearer" else "X-API-Key"
    auth_header = str(raw.get("auth_header") or default_header).strip()
    if not re.fullmatch(r"[A-Za-z0-9-]+", auth_header):
        raise HostedCompareError("The hosted auth header name is invalid.")
    provider_label = str(raw.get("provider_label") or "Hosted endpoint").strip()
    provider_label = re.sub(r"[\r\n\t]+", " ", provider_label)[:80] or "Hosted endpoint"
    if "://" in provider_label:
        provider_label = "Configured hosted endpoint"
    return {
        "models_url": models_url,
        "chat_url": chat_url,
        "auth_scheme": "bearer" if scheme == "bearer" else "x-api-key",
        "auth_header": auth_header,
        "provider_label": provider_label,
    }


def _runtime_safe_provider_config(
    provider_config: Mapping[str, str] | None,
    api_key: str,
) -> dict[str, str]:
    """Validate provider metadata and prevent a key from entering its URLs."""

    config = _validated_provider_config(provider_config)
    key = str(api_key or "")
    if key and any(key in str(config.get(field) or "") for field in ("models_url", "chat_url")):
        raise HostedCompareError("The configured hosted endpoint URL is invalid.")
    config["provider_label"] = str(redact_sensitive(config.get("provider_label", ""), (key,)))
    return config


def _runtime_auth_headers(api_key: str, config: Mapping[str, str]) -> dict[str, str]:
    """Construct ephemeral auth headers; callers must not return this mapping."""

    if config.get("auth_scheme") == "bearer":
        auth_value = f"Bearer {api_key}"
    else:
        auth_value = api_key
    return {
        str(config.get("auth_header") or "Authorization"): auth_value,
        "Accept": "application/json",
    }


def _safe_response_json(response: Any, api_key: str) -> Any:
    try:
        return response.json()
    except Exception as exc:
        preview = redact_sensitive(getattr(response, "text", "")[:500], (api_key,))
        raise HostedCompareError(f"Endpoint returned invalid JSON. {preview}") from exc


def _raise_for_status(response: Any, api_key: str, operation: str) -> None:
    status = int(getattr(response, "status_code", 200) or 200)
    if 200 <= status < 300:
        return
    if 300 <= status < 400:
        raise HostedCompareError(f"{operation} refused an endpoint redirect (HTTP {status}).")
    preview = redact_sensitive(getattr(response, "text", "")[:500], (api_key,))
    if status in (401, 403):
        raise HostedCompareError(f"{operation} authorization failed (HTTP {status}).")
    suffix = f" {preview}" if preview else ""
    raise HostedCompareError(f"{operation} failed (HTTP {status}).{suffix}")


def _default_get(url: str, **kwargs: Any) -> Any:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise HostedCompareError("The requests package is required for hosted comparisons.") from exc
    return requests.get(url, **kwargs)


def _default_post(url: str, **kwargs: Any) -> Any:
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise HostedCompareError("The requests package is required for hosted comparisons.") from exc
    return requests.post(url, **kwargs)


_NON_CHAT_HINTS = (
    "embed",
    "embedding",
    "rerank",
    "retrieval",
    "nvclip",
    "parakeet",
    "whisper",
    "tts",
    "asr",
    "guardrail",
    "content-safety",
    "parse",
)


def _is_comparison_candidate(model_id: str) -> bool:
    lowered = model_id.lower()
    return bool(lowered) and not any(hint in lowered for hint in _NON_CHAT_HINTS)


def _normalize_modality(value: Any) -> str | None:
    token = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    if token in ("text", "language", "prompt"):
        return "text"
    if token in ("image", "images", "vision", "image_url"):
        return "image"
    if token in ("video", "videos", "video_url"):
        return "video"
    return None


def _modalities_from_metadata(source: Mapping[str, Any] | None) -> set[str]:
    if not isinstance(source, Mapping):
        return set()
    out: set[str] = set()
    candidate_values: list[Any] = []
    for key in ("input_modalities", "modalities", "supported_inputs", "inputs"):
        if key in source:
            candidate_values.append(source.get(key))
    for nested_key in ("capabilities", "metadata"):
        nested = source.get(nested_key)
        if isinstance(nested, Mapping):
            for key in ("input_modalities", "modalities", "supported_inputs", "inputs"):
                if key in nested:
                    candidate_values.append(nested.get(key))
            for modality in ("text", "image", "video"):
                for key in (f"supports_{modality}", f"{modality}_input"):
                    if nested.get(key) is True:
                        out.add(modality)
        elif isinstance(nested, (list, tuple, str)):
            candidate_values.append(nested)
    for modality in ("text", "image", "video"):
        for key in (f"supports_{modality}", f"{modality}_input"):
            if source.get(key) is True:
                out.add(modality)
    for raw in candidate_values:
        if isinstance(raw, str):
            values = re.split(r"[,|\s]+", raw)
        elif isinstance(raw, Mapping):
            values = [key for key, enabled in raw.items() if enabled]
        elif isinstance(raw, (list, tuple, set)):
            values = list(raw)
        else:
            continue
        for value in values:
            normalized = _normalize_modality(value)
            if normalized:
                out.add(normalized)
    return out


def infer_input_modalities(
    model_id: str,
    catalog_metadata: Mapping[str, Any] | None = None,
) -> list[str]:
    """Infer endpoint input support, preferring catalog-declared capabilities."""

    declared = _modalities_from_metadata(catalog_metadata)
    if declared:
        declared.add("text")
        return [item for item in ("text", "image", "video") if item in declared]

    exact = AUTHORITATIVE_STATIC_MODALITIES.get(str(model_id or "").strip().lower())
    if exact:
        return list(exact)
    exact_discovered_id = str(model_id or "").strip()
    if _VIDEO_CAPABILITY_TAG_RE.search(exact_discovered_id):
        return ["text", "image", "video"]
    for pattern, modalities, _strategy in _CATALOG_TAG_MEDIA_PROFILES:
        if pattern.search(exact_discovered_id):
            return list(modalities)
    # Text is the only safe assumption for an OpenAI-compatible chat model.
    # In particular, do not infer vision/video support from strings such as
    # "omni", "vl", or a frontier family name.
    return ["text"]


def infer_video_strategy(
    model_id: str,
    catalog_metadata: Mapping[str, Any] | None = None,
) -> str | None:
    """Return a confirmed automatic video transport, or ``None`` if unknown."""

    declared = _modalities_from_metadata(catalog_metadata)
    if "video" in declared:
        return "native_video_url"
    if "image" in declared:
        return "sampled_image_frames"
    lowered = str(model_id or "").strip().lower()
    if lowered in AUTHORITATIVE_STATIC_MODALITIES:
        modalities = AUTHORITATIVE_STATIC_MODALITIES[lowered]
        return "native_video_url" if "video" in modalities else "sampled_image_frames"
    if _VIDEO_CAPABILITY_TAG_RE.search(str(model_id or "")):
        return "native_video_url"
    for pattern, _modalities, strategy in _CATALOG_TAG_MEDIA_PROFILES:
        if pattern.search(str(model_id or "")):
            return strategy
    return None


def model_supports_media(
    model_id: str,
    media_kind: str,
    catalog_modalities: Mapping[str, Sequence[str]] | None = None,
) -> tuple[bool, list[str]]:
    """Return support + normalized modalities for an image/video request."""

    declared = None
    if catalog_modalities:
        declared = catalog_modalities.get(model_id) or catalog_modalities.get(model_id.lower())
    modalities = [
        item
        for item in ("text", "image", "video")
        if item in {
            normalized
            for normalized in (_normalize_modality(value) for value in (declared or ()))
            if normalized
        }
    ]
    if not modalities:
        modalities = infer_input_modalities(model_id)
    kind = "image" if str(media_kind).lower() == "image" else "video"
    return kind in modalities, modalities


def resolve_media_strategy(
    model_id: str,
    is_image: bool,
    requested_strategy: str = "auto",
    catalog_modalities: Mapping[str, Sequence[str]] | None = None,
) -> tuple[str | None, list[str], str]:
    """Resolve transport with explicit overrides and capability-safe auto mode."""

    raw = str(requested_strategy or "auto").strip().lower()
    explicit_native = "force" in raw and "native" in raw
    explicit_frames = "force" in raw and ("frame" in raw or "image" in raw)
    declared = None
    if catalog_modalities:
        declared = catalog_modalities.get(model_id) or catalog_modalities.get(model_id.lower())
    modalities = [
        item
        for item in ("text", "image", "video")
        if item in {
            normalized
            for normalized in (_normalize_modality(value) for value in (declared or ()))
            if normalized
        }
    ]
    if not modalities:
        modalities = infer_input_modalities(model_id)

    if is_image:
        if explicit_native or explicit_frames:
            return "image_url", modalities, "explicit_user_override"
        if "image" in modalities:
            return "image_url", modalities, "catalog_or_confirmed_tag"
        return None, modalities, "capability_unconfirmed"

    if explicit_native:
        return "native_video_url", modalities, "explicit_user_override"
    if explicit_frames:
        return "sampled_image_frames", modalities, "explicit_user_override"
    if "video" in modalities:
        return "native_video_url", modalities, "catalog_or_confirmed_tag"
    if "image" in modalities:
        return "sampled_image_frames", modalities, "catalog_or_confirmed_tag"
    return None, modalities, "capability_unconfirmed"


def discover_hosted_models(
    api_key: str,
    *,
    endpoint: str | None = None,
    provider_config: Mapping[str, str] | None = None,
    http_get: Callable[..., Any] | None = None,
    timeout_s: float = 20.0,
) -> list[dict[str, Any]]:
    """Discover the authenticated hosted model catalog.

    The return value contains only public model metadata; the credential is
    never included.  Non-chat services such as embeddings and ASR are filtered
    from the comparison picker.
    """

    key = validate_runtime_key(api_key)
    config = _runtime_safe_provider_config(provider_config, key)
    catalog_url = _validated_endpoint_url(
        str(endpoint or config["models_url"]), "model catalog", os.environ
    )
    getter = http_get or _default_get
    try:
        response = getter(
            catalog_url,
            headers=_runtime_auth_headers(key, config),
            timeout=timeout_s,
            allow_redirects=False,
        )
        _raise_for_status(response, key, "Model discovery")
        payload = _safe_response_json(response, key)
    except HostedCompareError as exc:
        safe = redact_sensitive(
            _redact_configured_urls(str(exc), (catalog_url, config.get("chat_url", ""))),
            (key,),
        )
        raise HostedCompareError(safe) from exc
    except Exception as exc:
        safe = redact_sensitive(
            _redact_configured_urls(str(exc), (catalog_url, config.get("chat_url", ""))),
            (key,),
        )
        raise HostedCompareError(f"Model discovery failed: {safe}") from exc

    raw_models = payload.get("data", payload.get("models", [])) if isinstance(payload, Mapping) else []
    if not isinstance(raw_models, list):
        raise HostedCompareError("Model discovery returned an unexpected catalog shape.")

    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_models:
        if isinstance(raw, str):
            model_id = raw.strip()
            source: Mapping[str, Any] = {}
        elif isinstance(raw, Mapping):
            model_id = str(raw.get("id") or raw.get("name") or "").strip()
            source = raw
        else:
            continue
        if (
            not model_id
            or key in model_id
            or model_id in seen
            or "://" in model_id
            or not _MODEL_ID_RE.fullmatch(model_id)
            or any(pattern.search(model_id) for pattern in _TOKEN_PATTERNS)
            or not _is_comparison_candidate(model_id)
        ):
            continue
        seen.add(model_id)
        metadata = source.get("metadata") if isinstance(source.get("metadata"), Mapping) else {}
        capabilities = source.get("capabilities")
        if not isinstance(capabilities, (list, tuple, dict, str)):
            capabilities = None
        input_modalities = infer_input_modalities(model_id, source)
        models.append(
            {
                "id": model_id,
                "owned_by": redact_sensitive(
                    str(source.get("owned_by") or source.get("publisher") or ""), (key,)
                ),
                "capabilities": redact_sensitive(capabilities, (key,)),
                "input_modalities": input_modalities,
                "metadata": redact_sensitive(metadata, (key,)),
            }
        )
    models.sort(key=lambda item: item["id"].lower())
    return models


def _natural_model_rank(model_id: str) -> tuple[Any, ...]:
    lowered = model_id.lower()
    numeric = tuple(float(token) for token in re.findall(r"\d+(?:\.\d+)?", lowered))
    quality = (
        1 if "reason" in lowered else 0,
        1 if "instruct" in lowered or "chat" in lowered else 0,
        0 if "preview" in lowered else 1,
    )
    return numeric, quality, lowered


def curated_model_selection(
    models: Iterable[Mapping[str, Any] | str],
    *,
    exclude_model_ids: Iterable[str] = (),
) -> tuple[list[str], dict[str, list[str]]]:
    """Pick one high-ranked discovered candidate per requested model family.

    The complete family map is also returned so the UI can expose every
    available endpoint while defaulting to a compact comparison set.
    """

    ids = []
    for model in models:
        model_id = str(model.get("id") if isinstance(model, Mapping) else model).strip()
        if model_id and model_id not in ids:
            ids.append(model_id)
    excluded = {str(item).strip().lower() for item in exclude_model_ids if str(item).strip()}
    selected: list[str] = []
    family_map: dict[str, list[str]] = {}
    for family, hints in CURATED_FAMILIES:
        matches = [
            model_id
            for model_id in ids
            if model_id.lower() not in excluded
            and any(hint in model_id.lower() for hint in hints)
        ]
        for preferred_pattern in _CURATED_FAMILY_PREFERENCES.get(family, ()):
            preferred = [model_id for model_id in matches if preferred_pattern.search(model_id)]
            if preferred:
                matches = preferred
                break
        matches.sort(key=_natural_model_rank, reverse=True)
        family_map[family] = matches
        if matches and matches[0] not in selected:
            selected.append(matches[0])
    return selected[:MAX_SELECTED_MODELS], family_map


def discovery_summary_html(models: Sequence[Mapping[str, Any]], family_map: Mapping[str, Sequence[str]]) -> str:
    counts = []
    for family, _hints in CURATED_FAMILIES:
        matches = list(family_map.get(family, ()))
        if matches:
            counts.append(f"<b>{html.escape(family)}</b>: {len(matches)}")
        else:
            counts.append(f"<b>{html.escape(family)}</b>: unavailable")
    return (
        '<div style="border:1px solid #3b82f6;border-radius:6px;padding:9px 12px;'
        'background:#0f1e36;color:#dbeafe;font-size:12px">'
        f"Discovered <b>{len(models)}</b> chat-capable endpoint models. "
        + " · ".join(counts)
        + "<br><span style=\"color:#93c5fd\">The picker defaults to one discovered candidate per family; "
        "review the full list before running.</span></div>"
    )


def parse_ablation_variants(
    mode: str,
    base_prompt: str,
    base_system_prompt: str,
    raw_spec: str | Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, str]]:
    """Build a bounded list of spot or workflow-ablation prompt variants."""

    normalized_mode = str(mode or "spot").strip().lower()
    if "ablation" not in normalized_mode:
        return [
            {
                "name": "spot",
                "prompt": str(base_prompt or ""),
                "system_prompt": str(base_system_prompt or ""),
            }
        ]

    if raw_spec is None or (isinstance(raw_spec, str) and not raw_spec.strip()):
        parsed: Any = json.loads(DEFAULT_ABLATION_SPEC)
    elif isinstance(raw_spec, str):
        try:
            parsed = json.loads(raw_spec)
        except json.JSONDecodeError as exc:
            raise HostedCompareError(f"Ablation JSON is invalid at line {exc.lineno}, column {exc.colno}.") from exc
    else:
        parsed = list(raw_spec)

    if isinstance(parsed, Mapping):
        parsed = parsed.get("variants", [])
    if not isinstance(parsed, list) or not parsed:
        raise HostedCompareError("Ablation JSON must be a non-empty list of variant objects.")
    if len(parsed) > MAX_ABLATION_VARIANTS:
        raise HostedCompareError(f"Use at most {MAX_ABLATION_VARIANTS} ablation variants per run.")

    variants: list[dict[str, str]] = []
    seen_names: set[str] = set()
    for index, item in enumerate(parsed, 1):
        if not isinstance(item, Mapping):
            raise HostedCompareError(f"Ablation variant {index} must be a JSON object.")
        name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(item.get("name") or f"variant_{index}"))[:80]
        if not name or name in seen_names:
            raise HostedCompareError(f"Ablation variant names must be unique (problem at item {index}).")
        seen_names.add(name)
        prompt_template = str(item.get("prompt", "{prompt}"))
        system_template = str(item.get("system_prompt", "{system}"))
        variants.append(
            {
                "name": name,
                "prompt": prompt_template.replace("{prompt}", str(base_prompt or "")).replace(
                    "{system}", str(base_system_prompt or "")
                ),
                "system_prompt": system_template.replace("{prompt}", str(base_prompt or "")).replace(
                    "{system}", str(base_system_prompt or "")
                ),
            }
        )
    return variants


def _media_mime(path: str, is_image: bool) -> str:
    guessed = mimetypes.guess_type(path)[0]
    if guessed and guessed.startswith("image/") == bool(is_image):
        return guessed
    return "image/jpeg" if is_image else "video/mp4"


def _local_path_from_source(source: str) -> str:
    if source.startswith("file://"):
        raise HostedCompareError("File URLs are not accepted for hosted comparison; upload the media instead.")
    if source.startswith("http://") or source.startswith("https://"):
        raise HostedCompareError("Remote media URLs are not accepted for hosted comparison; upload the media instead.")
    path = os.path.realpath(os.path.abspath(os.path.expanduser(source)))
    configured = [
        os.path.realpath(os.path.abspath(os.path.expanduser(item)))
        for item in str(os.environ.get("BYO_VIDEO_COMPARE_MEDIA_ROOTS") or "").split(os.pathsep)
        if item.strip()
    ]
    roots = [os.path.realpath(tempfile.gettempdir()), *configured]
    if not any(path == root or path.startswith(root.rstrip(os.sep) + os.sep) for root in roots):
        raise HostedCompareError("Hosted comparison media must come from a frontend upload.")
    extension = Path(path).suffix.lower()
    if extension not in {".mp4", ".mov", ".mkv", ".webm", ".jpg", ".jpeg", ".png", ".webp"}:
        raise HostedCompareError("The uploaded comparison media type is not supported.")
    return path


def describe_media_source(media_source: str, is_image: bool) -> dict[str, Any]:
    """Return a report-safe descriptor without reading or encoding media bytes."""

    source = str(media_source or "").strip()
    if not source:
        raise HostedCompareError("Upload or select a reference video/image before comparing models.")
    local_path = _local_path_from_source(source)
    mime = _media_mime(local_path, is_image)
    if not os.path.isfile(local_path):
        raise HostedCompareError("The selected reference media file is not available on the server.")
    try:
        media_bytes = os.path.getsize(local_path)
    except OSError as exc:
        raise HostedCompareError("The selected reference media file could not be inspected.") from exc
    max_bytes = int(os.environ.get("BYO_VIDEO_COMPARE_MAX_MEDIA_BYTES", str(512 * 1024 * 1024)))
    if media_bytes <= 0 or media_bytes > max(1, max_bytes):
        raise HostedCompareError("The uploaded comparison media is empty or exceeds the configured size limit.")
    return {
        "kind": "image" if is_image else "video",
        "source": "uploaded_or_server_file",
        "mime_type": mime,
        "filename": os.path.basename(local_path),
        "bytes": media_bytes,
    }


def prepare_media_item(media_source: str, is_image: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """Prepare one OpenAI-compatible media item plus a secret-free descriptor."""

    source = str(media_source or "").strip()
    descriptor = describe_media_source(source, is_image)
    item_type = "image_url" if is_image else "video_url"
    local_path = _local_path_from_source(source)
    try:
        with open(local_path, "rb") as handle:
            raw = handle.read()
    except OSError as exc:
        raise HostedCompareError("The selected reference media file could not be read.") from exc
    media_url = f"data:{descriptor['mime_type']};base64,{base64.b64encode(raw).decode('ascii')}"
    return {"type": item_type, item_type: {"url": media_url}}, descriptor


def _sampled_frame_to_data_url(frame: Any, max_pixels: int) -> str:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise HostedCompareError("Pillow is required for sampled-frame comparison mode.") from exc
    image = frame.to_image().convert("RGB")
    width, height = image.size
    if max_pixels > 0 and width * height > max_pixels:
        scale = (float(max_pixels) / float(width * height)) ** 0.5
        target = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = image.resize(target, Image.Resampling.LANCZOS)
    import io

    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=False, progressive=False)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def sample_video_as_image_items(
    media_source: str,
    *,
    frame_count: int = 8,
    max_pixels: int = 524288,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Deterministically sample evenly spaced JPEG frames across a video."""

    try:
        import av
    except ImportError as exc:  # pragma: no cover - deployment dependency guard
        raise HostedCompareError("PyAV is required for sampled-frame comparison mode.") from exc
    source = _local_path_from_source(str(media_source or "").strip())
    requested = max(1, min(int(frame_count or 8), 32))
    try:
        container = av.open(source)
        stream = container.streams.video[0]
    except Exception as exc:
        raise HostedCompareError("The reference video could not be opened for deterministic frame sampling.") from exc

    try:
        total_frames = int(stream.frames or 0)
        if total_frames <= 0 and stream.duration is not None and stream.average_rate:
            total_frames = max(
                1,
                int(round(float(stream.duration * stream.time_base) * float(stream.average_rate))),
            )
        if total_frames <= 0:
            # Count once, then reopen for a deterministic evenly-spaced pass.
            total_frames = sum(1 for _frame in container.decode(stream))
            container.close()
            container = av.open(source)
            stream = container.streams.video[0]
        if total_frames <= 0:
            raise HostedCompareError("The reference video contains no decodable frames.")
        actual_count = min(requested, total_frames)
        if actual_count == 1:
            target_indices = [0]
        else:
            target_indices = sorted(
                {
                    int(round(index * (total_frames - 1) / (actual_count - 1)))
                    for index in range(actual_count)
                }
            )
        target_set = set(target_indices)
        items = []
        captured_indices = []
        for frame_index, frame in enumerate(container.decode(stream)):
            if frame_index not in target_set:
                continue
            items.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _sampled_frame_to_data_url(frame, int(max_pixels or 524288))
                    },
                }
            )
            captured_indices.append(frame_index)
            if len(items) >= len(target_indices):
                break
    except HostedCompareError:
        raise
    except Exception as exc:
        raise HostedCompareError("Deterministic video frame sampling failed.") from exc
    finally:
        try:
            container.close()
        except Exception:
            pass
    if not items:
        raise HostedCompareError("The reference video produced no sampled image frames.")
    adaptation = {
        "strategy": "sampled_image_frames",
        "frame_count": len(items),
        "frame_indices": captured_indices,
        "total_video_frames": total_frames,
        "max_pixels_per_frame": int(max_pixels or 524288),
        "deterministic": True,
    }
    return items, adaptation


def prepare_media_items_for_strategy(
    media_source: str,
    is_image: bool,
    strategy: str,
    *,
    sampled_frame_count: int = 8,
    sampled_frame_max_pixels: int = 524288,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Prepare media content and record any video-to-frame adaptation."""

    descriptor = describe_media_source(media_source, is_image)
    if is_image:
        item, descriptor = prepare_media_item(media_source, True)
        return [item], descriptor, {
            "strategy": "image_url",
            "reference_kind": "image",
            "sent_as": "image_url",
            "deterministic": True,
        }
    if strategy == "native_video_url":
        item, descriptor = prepare_media_item(media_source, False)
        return [item], descriptor, {
            "strategy": "native_video_url",
            "reference_kind": "video",
            "sent_as": "video_url",
            "deterministic": True,
        }
    if strategy == "sampled_image_frames":
        items, adaptation = sample_video_as_image_items(
            media_source,
            frame_count=sampled_frame_count,
            max_pixels=sampled_frame_max_pixels,
        )
        adaptation.update({"reference_kind": "video", "sent_as": "image_url[]"})
        return items, descriptor, adaptation
    raise HostedCompareError(f"Unsupported media strategy: {strategy}")


def build_chat_payload(
    model_id: str,
    media_item: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    prompt: str,
    system_prompt: str,
    *,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Build the canonical media-first chat-completions request body."""

    messages = []
    if str(system_prompt or "").strip():
        messages.append({"role": "system", "content": str(system_prompt)})
    if isinstance(media_item, Mapping):
        media_items = [copy.deepcopy(dict(media_item))]
    else:
        media_items = [copy.deepcopy(dict(item)) for item in media_item]
    messages.append(
        {
            "role": "user",
            "content": media_items + [{"type": "text", "text": str(prompt or "")}],
        }
    )
    return {
        "model": str(model_id),
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": max(1, int(max_tokens)),
        "stream": False,
    }


def redacted_request_shape(
    payload: Mapping[str, Any],
    media_descriptor: Mapping[str, Any],
    provider_config: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Describe request structure without prompts, auth headers, or media bytes."""

    config = _validated_provider_config(provider_config)
    messages = list(payload.get("messages") or [])
    system_text = ""
    prompt_text = ""
    content_types: list[str] = []
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        content = message.get("content")
        if message.get("role") == "system" and isinstance(content, str):
            system_text = content
        if message.get("role") != "user" or not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, Mapping):
                continue
            item_type = str(item.get("type") or "")
            if item_type:
                content_types.append(item_type)
            if item_type == "text":
                prompt_text += str(item.get("text") or "")
    return {
        "provider": "Hosted endpoint",
        "method": "POST",
        "path": "/v1/chat/completions",
        "runtime_credential": "omitted",
        "message_roles": [str(message.get("role") or "") for message in messages if isinstance(message, Mapping)],
        "content_types": content_types,
        "prompt_chars": len(prompt_text),
        "prompt_sha256_16": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16],
        "system_prompt_chars": len(system_text),
        "system_prompt_sha256_16": hashlib.sha256(system_text.encode("utf-8")).hexdigest()[:16],
        "parameters_sent": {
            key: payload.get(key)
            for key in ("temperature", "top_p", "max_tokens", "stream")
            if key in payload
        },
        "media": redact_sensitive(dict(media_descriptor)),
    }


def _response_text(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        raise HostedCompareError("Endpoint response was not a JSON object.")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise HostedCompareError("Endpoint response did not contain a completion choice.")
    choice = choices[0] if isinstance(choices[0], Mapping) else {}
    message = choice.get("message") if isinstance(choice.get("message"), Mapping) else {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        if parts:
            return "\n".join(parts)
    if isinstance(choice.get("text"), str):
        return choice["text"]
    raise HostedCompareError("Endpoint returned a completion with no text content.")


def invoke_hosted_model(
    api_key: str,
    model_id: str,
    media_item: Mapping[str, Any] | Sequence[Mapping[str, Any]],
    media_descriptor: Mapping[str, Any],
    prompt: str,
    system_prompt: str,
    *,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 512,
    endpoint: str | None = None,
    provider_config: Mapping[str, str] | None = None,
    timeout_s: float = 300.0,
    http_post: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Run one endpoint model and return only redacted, report-safe fields."""

    key = validate_runtime_key(api_key)
    config = _runtime_safe_provider_config(provider_config, key)
    model_id = validate_hosted_model_id(model_id, key)
    chat_url = _validated_endpoint_url(
        str(endpoint or config["chat_url"]), "chat completion", os.environ
    )
    payload = build_chat_payload(
        model_id,
        media_item,
        prompt,
        system_prompt,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    request_shape = redacted_request_shape(payload, media_descriptor, config)
    poster = http_post or _default_post
    started = time.perf_counter()
    try:
        response = poster(
            chat_url,
            headers={**_runtime_auth_headers(key, config), "Content-Type": "application/json"},
            json=payload,
            timeout=timeout_s,
            allow_redirects=False,
        )
        latency_s = time.perf_counter() - started
        _raise_for_status(response, key, f"Model {model_id}")
        response_payload = _safe_response_json(response, key)
        output = redact_sensitive(
            _redact_configured_urls(
                _response_text(response_payload),
                (chat_url, config.get("models_url", "")),
            ),
            (key,),
        )
        usage = response_payload.get("usage", {}) if isinstance(response_payload, Mapping) else {}
        return {
            "status": "ok",
            "latency_s": round(latency_s, 3),
            "output": output,
            "error": "",
            "usage": redact_sensitive(usage, (key,)),
            "request_shape": request_shape,
        }
    except Exception as exc:
        latency_s = time.perf_counter() - started
        error_secrets = tuple(
            value for value in (key, str(prompt or ""), str(system_prompt or "")) if value
        )
        safe_error = redact_sensitive(
            _redact_configured_urls(str(exc), (chat_url, config.get("models_url", ""))),
            error_secrets,
        )
        return {
            "status": "error",
            "latency_s": round(latency_s, 3),
            "output": "",
            "error": safe_error,
            "usage": {},
            "request_shape": request_shape,
        }


def run_hosted_comparisons(
    api_key: str,
    model_ids: Sequence[str],
    variants: Sequence[Mapping[str, str]],
    media_source: str,
    is_image: bool,
    *,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 512,
    concurrency: int = 4,
    catalog_modalities: Mapping[str, Sequence[str]] | None = None,
    media_strategy: str = "auto",
    sampled_frame_count: int = 8,
    sampled_frame_max_pixels: int = 524288,
    provider_config: Mapping[str, str] | None = None,
    request_fn: Callable[..., Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Run a bounded model × workflow matrix concurrently.

    ``request_fn`` exists for deterministic smoke tests.  It receives the same
    keyword arguments as :func:`invoke_hosted_model`.
    """

    key = validate_runtime_key(api_key)
    config = _runtime_safe_provider_config(provider_config, key)
    clean_models = []
    for raw in model_ids:
        model_id = str(raw or "").strip()
        if model_id:
            model_id = validate_hosted_model_id(model_id, key)
        if model_id and model_id not in clean_models:
            clean_models.append(model_id)
    if not clean_models:
        raise HostedCompareError("Select at least one hosted endpoint model.")
    if len(clean_models) > MAX_SELECTED_MODELS:
        raise HostedCompareError(f"Select at most {MAX_SELECTED_MODELS} hosted models per run.")
    if not variants:
        raise HostedCompareError("No spot/ablation variants were provided.")
    if key in json.dumps([dict(item) for item in variants], default=str):
        raise HostedCompareError("Credentials must not be included in prompt or workflow fields.")
    if len(variants) > MAX_ABLATION_VARIANTS:
        raise HostedCompareError(f"Use at most {MAX_ABLATION_VARIANTS} ablation variants.")
    task_count = len(clean_models) * len(variants)
    if task_count > MAX_COMPARISON_RUNS:
        raise HostedCompareError(f"The comparison matrix is capped at {MAX_COMPARISON_RUNS} endpoint calls.")

    descriptor = describe_media_source(media_source, is_image)
    media_kind = "image" if is_image else "video"
    invoker = request_fn or invoke_hosted_model
    tasks = []
    for variant_index, variant in enumerate(variants):
        for model_index, model_id in enumerate(clean_models):
            strategy, modalities, strategy_source = resolve_media_strategy(
                model_id,
                is_image,
                media_strategy,
                catalog_modalities,
            )
            tasks.append(
                (
                    variant_index,
                    model_index,
                    model_id,
                    dict(variant),
                    strategy,
                    modalities,
                    strategy_source,
                )
            )

    results: list[dict[str, Any]] = []
    supported_tasks = [task for task in tasks if task[4] is not None]
    prepared_by_strategy: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    preparation_errors: dict[str, str] = {}
    for strategy in sorted({str(task[4]) for task in supported_tasks}):
        try:
            media_items, prepared_descriptor, adaptation = prepare_media_items_for_strategy(
                media_source,
                is_image,
                strategy,
                sampled_frame_count=sampled_frame_count,
                sampled_frame_max_pixels=sampled_frame_max_pixels,
            )
            descriptor = prepared_descriptor
            prepared_by_strategy[strategy] = (media_items, adaptation)
        except Exception as exc:
            preparation_errors[strategy] = redact_sensitive(str(exc), (key,))

    for variant_index, model_index, model_id, variant, strategy, modalities, strategy_source in tasks:
        if strategy is not None and str(strategy) not in preparation_errors:
            continue
        placeholder_item = {
            "type": f"{media_kind}_url",
            f"{media_kind}_url": {"url": "<reference-media-not-sent>"},
        }
        payload = build_chat_payload(
            model_id,
            placeholder_item,
            variant.get("prompt", ""),
            variant.get("system_prompt", ""),
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        shape = redacted_request_shape(payload, descriptor, config)
        shape["not_sent"] = True
        prep_error = preparation_errors.get(str(strategy)) if strategy is not None else None
        shape["reason"] = "media_preparation_error" if prep_error else "unsupported_media"
        adaptation = {
            "strategy": str(strategy) if prep_error else "blocked",
            "selection_source": strategy_source,
            "requested_strategy": str(media_strategy),
            "reference_kind": media_kind,
            "sent_as": "not_sent",
        }
        shape["media_adaptation"] = adaptation
        results.append(
            {
                "_sort": [variant_index, model_index],
                "backend": "hosted_endpoint",
                "provider": "Hosted endpoint",
                "model": model_id,
                "variant": variant.get("name", f"variant_{variant_index + 1}"),
                "status": "media_preparation_error" if prep_error else "unsupported_media",
                "latency_s": 0.0,
                "output": "",
                "error": prep_error or (
                    f"{media_kind} capability is not confirmed for this endpoint model "
                    f"(confirmed inputs: {', '.join(modalities) or 'text'}); no request was sent. "
                    "Choose an explicit custom media strategy only if the endpoint contract is known."
                ),
                "usage": {},
                "input_modalities": modalities,
                "media_adaptation": adaptation,
                "request_shape": shape,
            }
        )

    supported_tasks = [task for task in supported_tasks if str(task[4]) in prepared_by_strategy]
    worker_count = max(1, min(int(concurrency or 1), 8, len(supported_tasks) or 1))
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="hosted-compare") as pool:
        future_map = {}
        for (
            variant_index,
            model_index,
            model_id,
            variant,
            strategy,
            modalities,
            strategy_source,
        ) in supported_tasks:
            media_items, base_adaptation = prepared_by_strategy[str(strategy)]
            adaptation = {
                **base_adaptation,
                "selection_source": strategy_source,
                "requested_strategy": str(media_strategy),
            }
            future = pool.submit(
                invoker,
                key,
                model_id,
                media_items,
                descriptor,
                variant.get("prompt", ""),
                variant.get("system_prompt", ""),
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                provider_config=config,
            )
            future_map[future] = (
                variant_index,
                model_index,
                model_id,
                variant,
                modalities,
                media_items,
                adaptation,
            )
        for future in as_completed(future_map):
            (
                variant_index,
                model_index,
                model_id,
                variant,
                modalities,
                media_items,
                adaptation,
            ) = future_map[future]
            try:
                outcome = dict(future.result())
            except Exception as exc:  # defensive for injected/custom invokers
                outcome = {
                    "status": "error",
                    "latency_s": 0.0,
                    "output": "",
                    "error": redact_sensitive(str(exc), (key,)),
                    "usage": {},
                    "request_shape": redacted_request_shape(
                        build_chat_payload(
                            model_id,
                            media_items,
                            variant.get("prompt", ""),
                            variant.get("system_prompt", ""),
                            temperature=temperature,
                            top_p=top_p,
                            max_tokens=max_tokens,
                        ),
                        descriptor,
                        config,
                    ),
                }
            request_shape = dict(outcome.get("request_shape") or {})
            request_shape["media_adaptation"] = adaptation
            outcome["request_shape"] = request_shape
            results.append(
                redact_sensitive(
                    {
                        "_sort": [variant_index, model_index],
                        "backend": "hosted_endpoint",
                        "provider": "Hosted endpoint",
                        "model": model_id,
                        "variant": variant.get("name", f"variant_{variant_index + 1}"),
                        "input_modalities": modalities,
                        "media_adaptation": adaptation,
                        **outcome,
                    },
                    (key,),
                )
            )
    results.sort(key=lambda row: tuple(row.get("_sort", (999, 999))))
    for row in results:
        row.pop("_sort", None)
    return redact_sensitive(results, (key,))


def build_loaded_request_shape(
    model_id: str,
    backend: str,
    media_descriptor: Mapping[str, Any],
    prompt: str,
    system_prompt: str,
    *,
    fps: int,
    max_pixels: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Redacted shape summary for the currently loaded local/NIM baseline."""

    prompt_text = str(prompt or "")
    system_text = str(system_prompt or "")
    return redact_sensitive(
        {
            "transport": str(backend),
            "model": report_safe_model_label(model_id),
            "message_roles": ["system", "user"],
            "content_types": [f"{media_descriptor.get('kind', 'video')}_reference", "text"],
            "prompt_chars": len(prompt_text),
            "prompt_sha256_16": hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16],
            "system_prompt_chars": len(system_text),
            "system_prompt_sha256_16": hashlib.sha256(system_text.encode("utf-8")).hexdigest()[:16],
            "media": dict(media_descriptor),
            "parameters_sent": {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "repetition_penalty": float(repetition_penalty),
                "repetition_penalty_location": (
                    "nvext" if str(backend).lower() == "nim_local" else "root"
                ),
                "max_tokens": "omitted_by_loaded_endpoint_contract",
            },
            "settings_snapshot": {
                "fps": int(fps),
                "max_pixels": int(max_pixels),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "repetition_penalty": float(repetition_penalty),
                "max_tokens": max(1, int(max_tokens)),
            },
        }
    )


def _markdown_safe(value: Any) -> str:
    return str(value if value is not None else "").replace("```", "` ` `")


def write_redacted_exports(
    results: Sequence[Mapping[str, Any]],
    *,
    mode: str,
    media_descriptor: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    output_dir: str | None = None,
    secrets: Sequence[str] = (),
) -> tuple[str, str, dict[str, Any]]:
    """Write secret/media-redacted JSON and Markdown comparison artifacts."""

    generated_at = _dt.datetime.now(_dt.timezone.utc).isoformat()
    safe_results = []
    for item in results:
        row = dict(item)
        if "model" in row:
            row["model"] = report_safe_model_label(row.get("model"))
        request_shape = row.get("request_shape")
        if isinstance(request_shape, Mapping) and "model" in request_shape:
            row["request_shape"] = {
                **dict(request_shape),
                "model": report_safe_model_label(request_shape.get("model")),
            }
        safe_results.append(row)
    document = redact_sensitive(
        {
            "schema_version": "1.0",
            "generated_at": generated_at,
            "mode": str(mode),
            "media": dict(media_descriptor),
            "metadata": dict(metadata or {}),
            "results": safe_results,
        },
        secrets,
    )
    target_dir = output_dir or tempfile.mkdtemp(prefix="byo-video-compare-")
    os.makedirs(target_dir, mode=0o700, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    json_path = os.path.join(target_dir, f"hosted-comparison-{stamp}.json")
    report_path = os.path.join(target_dir, f"hosted-comparison-{stamp}.md")

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)

    rows = []
    for result in document["results"]:
        latency = result.get("latency_s")
        latency_text = f"{latency:.3f}" if isinstance(latency, (int, float)) else "—"
        rows.append(
            f"| {_markdown_safe(result.get('variant'))} | {_markdown_safe(result.get('backend'))} | "
            f"{_markdown_safe(result.get('model'))} | {_markdown_safe(result.get('status'))} | {latency_text} |"
        )
    markdown = [
        "# BYO-video model comparison",
        "",
        f"Generated: `{generated_at}`  ",
        f"Mode: `{_markdown_safe(document['mode'])}`  ",
        f"Reference: `{_markdown_safe(document['media'].get('filename', 'media'))}`",
        "",
        "| Variant | Backend | Model | Status | Latency (s) |",
        "|---|---|---|---|---:|",
        *rows,
        "",
        "## Per-run results",
        "",
    ]
    for result in document["results"]:
        markdown.extend(
            [
                f"### {_markdown_safe(result.get('variant'))} · {_markdown_safe(result.get('model'))}",
                "",
                f"Status: `{_markdown_safe(result.get('status'))}` · Latency: `{_markdown_safe(result.get('latency_s'))}s`",
                "",
                "Output:",
                "",
                "```text",
                _markdown_safe(result.get("output") or result.get("error") or ""),
                "```",
                "",
                "Redacted request shape:",
                "",
                "```json",
                json.dumps(result.get("request_shape", {}), indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(markdown))
    return json_path, report_path, document


def results_html(results: Sequence[Mapping[str, Any]]) -> str:
    """Render status/latency/output/request-shape cards for Gradio."""

    if not results:
        return '<div style="color:#94a3b8">No comparison results yet.</div>'
    cards = []
    for result in results:
        status = str(result.get("status") or "error")
        ok = status == "ok"
        unsupported = status == "unsupported_media"
        border = "#22c55e" if ok else ("#f59e0b" if unsupported else "#f87171")
        badge = "OK" if ok else ("UNSUPPORTED MEDIA" if unsupported else status.upper())
        output = result.get("output") if ok else result.get("error")
        request_json = json.dumps(result.get("request_shape", {}), indent=2, ensure_ascii=False)
        usage_json = json.dumps(result.get("usage", {}), ensure_ascii=False)
        cards.append(
            '<div style="background:#0f172a;border:1px solid #334155;border-left:4px solid '
            f'{border};border-radius:7px;padding:12px 14px;margin:9px 0;color:#e2e8f0">'
            f'<div style="font-weight:700;color:#fff">{html.escape(str(result.get("variant", "spot")))} · '
            f'{html.escape(str(result.get("model", "model")))}</div>'
            f'<div style="font-size:12px;color:{border};margin:3px 0">{badge} · '
            f'{html.escape(str(result.get("backend", "")))} · {html.escape(str(result.get("latency_s", "—")))}s'
            f' · usage {html.escape(usage_json)}</div>'
            '<pre style="white-space:pre-wrap;overflow-wrap:anywhere;background:#111827;border-radius:5px;'
            f'padding:9px;color:#f8fafc">{html.escape(str(output or ""))}</pre>'
            '<details><summary style="cursor:pointer;color:#93c5fd">Redacted request shape</summary>'
            '<pre style="white-space:pre-wrap;overflow-wrap:anywhere;background:#020617;border-radius:5px;'
            f'padding:9px;color:#cbd5e1">{html.escape(request_json)}</pre></details></div>'
        )
    return "".join(cards)
