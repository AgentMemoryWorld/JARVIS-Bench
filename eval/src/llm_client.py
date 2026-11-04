"""OpenAI client with disk-cache by hash(prompt + model + params + image bytes).

Cache layout: eval/runs/.cache/{ab}/{full_hash}.json
A cache hit is free + instant, enabling safe retries / mid-run crashes.
Reads OPENAI_API_KEY from environment.

Multi-modal: pass `image_paths=[...]` to send images alongside the user text
(uses OpenAI's vision-capable chat-completions format with base64 data URLs).
The cache key is built from a content hash of each image so the cache is
invalidated automatically if the file on disk changes.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

CACHE_DIR = Path(__file__).resolve().parents[1] / "runs" / ".cache"

# Provider routing — extend to add more API providers.
PROVIDER_CONFIG: dict[str, dict[str, str | None]] = {
    "openai":        {"base_url": None, "api_key_env": "OPENAI_API_KEY"},
    "openrouter":    {"base_url": "https://openrouter.ai/api/v1", "api_key_env": "OPENROUTER_API_KEY"},
    # vertex_gemini uses Application Default Credentials (ADC) via google-genai;
    # no API key env var. project_id and location are passed per-call from model
    # config (see eval/configs/models.yaml).
    "vertex_gemini": {"base_url": None, "api_key_env": None},
    # google_genai_apikey hits Google AI Studio's generativelanguage.googleapis.com
    # directly with a project-bound API key. Generally has much higher quota
    # than Vertex's region buckets but can return 503 UNAVAILABLE during peak
    # demand on hot models (gemini-2.5-pro etc.).
    "google_genai_apikey": {"base_url": None, "api_key_env": "GEMINI_API_KEY"},
    # bedrock calls the Converse API with the standard AWS credential chain
    # (environment variables, ~/.aws, SSO). `region` and per-model
    # `additional_request_fields` come from eval/configs/models.yaml.
    "bedrock": {"base_url": None, "api_key_env": None},
}


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _encode_image_to_data_url(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    if suffix == "jpg":
        suffix = "jpeg"
    if suffix not in ("png", "jpeg", "gif", "webp"):
        # default to png; OpenAI usually accepts and re-detects from bytes
        suffix = "png"
    with open(path, "rb") as f:
        b = base64.b64encode(f.read()).decode("ascii")
    return f"data:image/{suffix};base64,{b}"


def _cache_key(
    model_id: str,
    system: str,
    user: str,
    params: dict[str, Any],
    image_paths: list[Path] | None = None,
) -> str:
    h = hashlib.sha256()
    h.update(model_id.encode("utf-8"))
    h.update(b"\x1f")
    h.update(system.encode("utf-8"))
    h.update(b"\x1f")
    h.update(user.encode("utf-8"))
    h.update(b"\x1f")
    h.update(json.dumps(params, sort_keys=True).encode("utf-8"))
    if image_paths:
        h.update(b"\x1f")
        for p in image_paths:
            h.update(b"|")
            h.update(_file_sha256(p).encode("ascii"))
    return h.hexdigest()


def _cache_path(key: str) -> Path:
    return CACHE_DIR / key[:2] / f"{key}.json"


def call_llm(
    *,
    model_id: str,
    system: str,
    user: str,
    temperature: float = 0.0,
    max_tokens: int = 2000,
    use_cache: bool = True,
    json_mode: bool = True,
    image_paths: list[str | Path] | None = None,
    provider: str = "openai",
    project_id: str | None = None,
    location: str | None = None,
    thinking_budget: int | None = None,
    reasoning: dict | None = None,
    provider_routing: dict | None = None,
    region: str | None = None,
    additional_request_fields: dict | None = None,
    cache_point: bool = False,
) -> dict[str, Any]:
    """Call the chat-completions API and return a normalized dict.

    If `image_paths` is given, each image is base64-inlined into a multimodal
    user message (vision). Cache key includes a sha256 of each image's bytes.

    Returns: {model_id, content, prompt_tokens, completion_tokens, total_tokens,
              elapsed_s, cache_key, cached: bool}
    """
    img_paths_resolved: list[Path] = []
    if image_paths:
        for ip in image_paths:
            p = Path(ip)
            if not p.exists():
                raise FileNotFoundError(f"image not found: {p}")
            img_paths_resolved.append(p)

    params = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "json_mode": json_mode,
        "n_images": len(img_paths_resolved),
        "provider": provider,
    }
    if reasoning:
        # canonicalize so dict key order doesn't change cache key
        params["reasoning"] = {k: reasoning[k] for k in sorted(reasoning)}
    if provider_routing:
        # Same canonicalization for OpenRouter `provider` routing knob so
        # changing it forces a fresh API call.
        params["provider_routing"] = {k: provider_routing[k] for k in sorted(provider_routing)}
    if provider == "vertex_gemini":
        # project_id/location can change which backend serves the model, so
        # include them in the cache key.
        params["project_id"] = project_id
        params["location"] = location
        if thinking_budget is not None:
            params["thinking_budget"] = thinking_budget
    elif provider == "google_genai_apikey":
        # thinking_budget changes the response, so it has to be in the key.
        if thinking_budget is not None:
            params["thinking_budget"] = thinking_budget
    elif provider == "bedrock":
        # region and request fields can change the response, so they are in the key.
        params["region"] = region
        if additional_request_fields:
            params["additional_request_fields"] = additional_request_fields
    key = _cache_key(model_id, system, user, params, img_paths_resolved or None)
    cpath = _cache_path(key)

    if use_cache and cpath.exists():
        rec = json.load(open(cpath, encoding="utf-8"))
        # Skip empty cached responses (e.g. reasoning ate all tokens last time).
        if rec.get("content"):
            rec["cached"] = True
            return rec
        # Otherwise fall through and re-call.

    if provider not in PROVIDER_CONFIG:
        raise ValueError(
            f"unknown provider: {provider}; known: {list(PROVIDER_CONFIG)}"
        )

    # ----- Amazon Bedrock (Converse API) branch -----
    if provider == "bedrock":
        return _call_bedrock(
            model_id=model_id,
            system=system,
            user=user,
            max_tokens=max_tokens,
            img_paths_resolved=img_paths_resolved,
            region=region,
            additional_request_fields=additional_request_fields,
            cache_point=cache_point,
            cache_key=key,
            cache_path=cpath,
        )

    # ----- Vertex AI (Gemini) branch -----
    if provider == "vertex_gemini":
        return _call_vertex_gemini(
            model_id=model_id,
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            img_paths_resolved=img_paths_resolved,
            project_id=project_id,
            location=location,
            cache_key=key,
            cache_path=cpath,
            thinking_budget=thinking_budget,
        )

    cfg = PROVIDER_CONFIG[provider]
    api_key_env = cfg["api_key_env"]
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"{api_key_env} not set (provider={provider}). Export it before running."
        )

    # ----- Google AI Studio (Gemini API key) branch -----
    if provider == "google_genai_apikey":
        return _call_google_genai_apikey(
            model_id=model_id,
            system=system,
            user=user,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=json_mode,
            img_paths_resolved=img_paths_resolved,
            api_key=api_key,
            cache_key=key,
            cache_path=cpath,
            thinking_budget=thinking_budget,
        )

    # Lazy import so dry-run / aggregation don't require openai installed.
    from openai import OpenAI

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if cfg["base_url"]:
        client_kwargs["base_url"] = cfg["base_url"]
    client = OpenAI(**client_kwargs)

    if img_paths_resolved:
        user_content: list[dict[str, Any]] = [{"type": "text", "text": user}]
        for p in img_paths_resolved:
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _encode_image_to_data_url(p)},
                }
            )
    else:
        user_content = user  # type: ignore[assignment]

    create_kwargs: dict[str, Any] = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        create_kwargs["response_format"] = {"type": "json_object"}
    extra_body: dict[str, Any] = {}
    if reasoning:
        # OpenRouter / OpenAI reasoning controls go via extra_body so the
        # standard SDK doesn't strip them.
        extra_body["reasoning"] = dict(reasoning)
    if provider_routing and provider == "openrouter":
        # OpenRouter `provider` routing knob: { order, sort, ignore,
        # allow_fallbacks, ... }. Use to bias / pin upstream provider when
        # the default route returns empty content (e.g. glm-5, minimax-m2.5).
        extra_body["provider"] = dict(provider_routing)
    if extra_body:
        create_kwargs["extra_body"] = extra_body

    def _looks_unsupported(err_msg: str, key: str) -> bool:
        m = err_msg.lower()
        return key.lower() in m and ("unsupported" in m or "not supported" in m or "invalid" in m)

    t0 = time.time()
    resp = None
    last_err: Exception | None = None
    rl_attempts = 0  # rate-limit (429) backoff attempts
    # Up to ~10 attempts total: swap kwargs on validation errors,
    # exponential backoff on rate-limit / transient errors.
    for _ in range(10):
        try:
            resp = client.chat.completions.create(**create_kwargs)
            break
        except Exception as e:
            last_err = e
            msg = str(e)
            adjusted = False

            # --- 1. Validation errors: adjust kwargs once and retry ---
            # GPT-5 / o-series: max_tokens -> max_completion_tokens
            if "max_tokens" in create_kwargs and (
                "max_completion_tokens" in msg or _looks_unsupported(msg, "max_tokens")
            ):
                create_kwargs["max_completion_tokens"] = create_kwargs.pop("max_tokens")
                adjusted = True
            # Reasoning models reject non-default temperature.
            elif "temperature" in create_kwargs and _looks_unsupported(msg, "temperature"):
                create_kwargs.pop("temperature", None)
                adjusted = True
            # Some models reject response_format / json mode.
            elif "response_format" in create_kwargs and (
                "response_format" in msg or "json" in msg.lower()
            ):
                create_kwargs.pop("response_format", None)
                adjusted = True

            # --- 2. Rate-limit (429) and transient errors: exponential backoff ---
            elif "rate_limit" in msg.lower() or "429" in msg or "tokens per min" in msg.lower():
                rl_attempts += 1
                if rl_attempts > 6:
                    raise
                wait_s = min(60, 5 * (2 ** (rl_attempts - 1)))  # 5,10,20,40,60,60
                print(f"  [rate-limit] sleep {wait_s}s and retry...", flush=True)
                time.sleep(wait_s)
                adjusted = True
            elif "timeout" in msg.lower() or "connection" in msg.lower():
                rl_attempts += 1
                if rl_attempts > 4:
                    raise
                wait_s = 5 * rl_attempts
                print(f"  [transient] sleep {wait_s}s and retry...", flush=True)
                time.sleep(wait_s)
                adjusted = True

            if not adjusted:
                raise
    if resp is None:
        raise last_err if last_err else RuntimeError("LLM call failed without exception")
    elapsed = time.time() - t0

    usage = getattr(resp, "usage", None)
    rec: dict[str, Any] = {
        "model_id": model_id,
        "content": resp.choices[0].message.content or "",
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
        "elapsed_s": round(elapsed, 3),
        "cache_key": key,
        "cached": False,
    }

    # Only cache successful, non-empty responses. Otherwise the next run will
    # blindly reuse the empty result. We still return the empty rec so the
    # caller sees what happened.
    if rec["content"]:
        cpath.parent.mkdir(parents=True, exist_ok=True)
        # atomic write so concurrent workers can't corrupt cache
        tmp = cpath.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        tmp.replace(cpath)
    else:
        # Annotate so the caller / parser logs are clearer.
        rec["empty_response"] = True
        rec["finish_reason"] = getattr(resp.choices[0], "finish_reason", None)
    return rec


def _call_vertex_gemini(
    *,
    model_id: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    img_paths_resolved: list[Path],
    project_id: str | None,
    location: str | None,
    cache_key: str,
    cache_path: Path,
    thinking_budget: int | None = None,
) -> dict[str, Any]:
    """Call Vertex AI Gemini via google-genai SDK using ADC.

    Returns the same normalized dict shape as the OpenAI path:
    {model_id, content, prompt_tokens, completion_tokens, total_tokens,
     elapsed_s, cache_key, cached: bool}.
    """
    if not project_id or not location:
        raise RuntimeError(
            f"vertex_gemini requires `project_id` and `location` in models.yaml "
            f"(got project_id={project_id!r}, location={location!r})"
        )

    # Lazy import so non-vertex runs don't require google-genai installed.
    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(vertexai=True, project=project_id, location=location)

    # Build contents: text + optional images.
    parts: list[Any] = [genai_types.Part.from_text(text=user)]
    for p in img_paths_resolved:
        suffix = p.suffix.lower().lstrip(".")
        if suffix == "jpg":
            suffix = "jpeg"
        if suffix not in ("png", "jpeg", "gif", "webp"):
            suffix = "png"
        with open(p, "rb") as f:
            data = f.read()
        parts.append(genai_types.Part.from_bytes(data=data, mime_type=f"image/{suffix}"))

    config_kwargs: dict[str, Any] = {
        "system_instruction": system,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"
    # Thinking budget control (Gemini 2.5+ models support extended thinking).
    # Set to 0 to disable thinking (avoids it eating output budget on Flash
    # variants). Set to a positive integer to cap thinking tokens. None means
    # use the model's default behavior.
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
            thinking_budget=thinking_budget,
        )

    t0 = time.time()
    resp = None
    last_err: Exception | None = None
    rl_attempts = 0
    for _ in range(10):
        try:
            resp = client.models.generate_content(
                model=model_id,
                contents=parts,
                config=genai_types.GenerateContentConfig(**config_kwargs),
            )
            break
        except Exception as e:
            last_err = e
            msg = str(e)
            cls_name = type(e).__name__
            # Rate-limit / quota — backoff and retry.
            if (
                "ResourceExhausted" in cls_name
                or "429" in msg
                or "quota" in msg.lower()
                or "rate" in msg.lower()
            ):
                rl_attempts += 1
                if rl_attempts > 6:
                    raise
                wait_s = min(60, 5 * (2 ** (rl_attempts - 1)))  # 5,10,20,40,60,60
                print(f"  [rate-limit] sleep {wait_s}s and retry...", flush=True)
                time.sleep(wait_s)
                continue
            # Transient server / network errors — short backoff.
            if (
                "DeadlineExceeded" in cls_name
                or "ServiceUnavailable" in cls_name
                or "InternalServerError" in cls_name
                or "Unavailable" in cls_name
                or "timeout" in msg.lower()
                or "connection" in msg.lower()
            ):
                rl_attempts += 1
                if rl_attempts > 4:
                    raise
                wait_s = 5 * rl_attempts
                print(f"  [transient] sleep {wait_s}s and retry...", flush=True)
                time.sleep(wait_s)
                continue
            raise
    if resp is None:
        raise last_err if last_err else RuntimeError("Vertex Gemini call failed without exception")
    elapsed = time.time() - t0

    # `resp.text` joins the text parts; if the model returned only function
    # calls or got blocked, .text may be empty.
    text = ""
    try:
        text = resp.text or ""
    except Exception:
        text = ""

    usage = getattr(resp, "usage_metadata", None)
    rec: dict[str, Any] = {
        "model_id": model_id,
        "content": text,
        "prompt_tokens": getattr(usage, "prompt_token_count", 0) if usage else 0,
        "completion_tokens": getattr(usage, "candidates_token_count", 0) if usage else 0,
        "total_tokens": getattr(usage, "total_token_count", 0) if usage else 0,
        "elapsed_s": round(elapsed, 3),
        "cache_key": cache_key,
        "cached": False,
    }

    if rec["content"]:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        tmp.replace(cache_path)
    else:
        rec["empty_response"] = True
        finish_reason = None
        try:
            cands = getattr(resp, "candidates", None) or []
            if cands:
                finish_reason = getattr(cands[0], "finish_reason", None)
                if finish_reason is not None and not isinstance(finish_reason, (str, int)):
                    finish_reason = str(finish_reason)
        except Exception:
            pass
        rec["finish_reason"] = finish_reason
    return rec


def _call_google_genai_apikey(
    *,
    model_id: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    json_mode: bool,
    img_paths_resolved: list[Path],
    api_key: str,
    cache_key: str,
    cache_path: Path,
    thinking_budget: int | None = None,
) -> dict[str, Any]:
    """Call Gemini via Google AI Studio (generativelanguage.googleapis.com)
    with a project-bound API key.

    Same normalized return shape as the Vertex / OpenAI paths.
    """
    from google import genai
    from google.genai import types as genai_types

    client = genai.Client(api_key=api_key)

    parts: list[Any] = [genai_types.Part.from_text(text=user)]
    for p in img_paths_resolved:
        suffix = p.suffix.lower().lstrip(".")
        if suffix == "jpg":
            suffix = "jpeg"
        if suffix not in ("png", "jpeg", "gif", "webp"):
            suffix = "png"
        with open(p, "rb") as f:
            data = f.read()
        parts.append(genai_types.Part.from_bytes(data=data, mime_type=f"image/{suffix}"))

    config_kwargs: dict[str, Any] = {
        "system_instruction": system,
        "temperature": temperature,
        "max_output_tokens": max_tokens,
    }
    if json_mode:
        config_kwargs["response_mime_type"] = "application/json"
    if thinking_budget is not None:
        config_kwargs["thinking_config"] = genai_types.ThinkingConfig(
            thinking_budget=thinking_budget,
        )

    t0 = time.time()
    resp = None
    last_err: Exception | None = None
    rl_attempts = 0
    # Up to 10 retries: handle 429 (quota) and 503 (server overload, common
    # on hot models like gemini-2.5-pro during peak demand) with backoff.
    for _ in range(10):
        try:
            resp = client.models.generate_content(
                model=model_id,
                contents=parts,
                config=genai_types.GenerateContentConfig(**config_kwargs),
            )
            break
        except Exception as e:
            last_err = e
            msg = str(e)
            cls_name = type(e).__name__
            # Rate-limit / quota.
            if (
                "ResourceExhausted" in cls_name
                or "429" in msg
                or "quota" in msg.lower()
                or "rate" in msg.lower()
            ):
                rl_attempts += 1
                if rl_attempts > 6:
                    raise
                wait_s = min(60, 5 * (2 ** (rl_attempts - 1)))  # 5,10,20,40,60,60
                print(f"  [rate-limit] sleep {wait_s}s and retry...", flush=True)
                time.sleep(wait_s)
                continue
            # 503 UNAVAILABLE / transient server errors. AI Studio's hot models
            # often return 503 during peak hours — back off longer.
            if (
                "503" in msg
                or "UNAVAILABLE" in msg
                or "Unavailable" in cls_name
                or "ServerError" in cls_name
                or "InternalServerError" in cls_name
                or "DeadlineExceeded" in cls_name
                or "timeout" in msg.lower()
                or "connection" in msg.lower()
            ):
                rl_attempts += 1
                if rl_attempts > 6:
                    raise
                wait_s = min(120, 10 * (2 ** (rl_attempts - 1)))  # 10,20,40,80,120,120
                print(f"  [server-busy] sleep {wait_s}s and retry...", flush=True)
                time.sleep(wait_s)
                continue
            raise
    if resp is None:
        raise last_err if last_err else RuntimeError("AI Studio call failed without exception")
    elapsed = time.time() - t0

    text = ""
    try:
        text = resp.text or ""
    except Exception:
        text = ""

    usage = getattr(resp, "usage_metadata", None)
    rec: dict[str, Any] = {
        "model_id": model_id,
        "content": text,
        "prompt_tokens": getattr(usage, "prompt_token_count", 0) if usage else 0,
        "completion_tokens": getattr(usage, "candidates_token_count", 0) if usage else 0,
        "total_tokens": getattr(usage, "total_token_count", 0) if usage else 0,
        "elapsed_s": round(elapsed, 3),
        "cache_key": cache_key,
        "cached": False,
    }

    if rec["content"]:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        tmp.replace(cache_path)
    else:
        rec["empty_response"] = True
        finish_reason = None
        try:
            cands = getattr(resp, "candidates", None) or []
            if cands:
                finish_reason = getattr(cands[0], "finish_reason", None)
                if finish_reason is not None and not isinstance(finish_reason, (str, int)):
                    finish_reason = str(finish_reason)
        except Exception:
            pass
        rec["finish_reason"] = finish_reason
    return rec


# ---------------------------------------------------------------------------
# Amazon Bedrock (Converse API)
# ---------------------------------------------------------------------------
_BEDROCK_CLIENTS: dict[str, Any] = {}
_BEDROCK_LOCK = threading.Lock()
_BEDROCK_IMAGE_FORMATS = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}
# prompt_builder section headers. With `cache_point: true` a cache point is put
# after the activity log, so the conditions that share the log (C2, C3, C4)
# reuse the cached prefix. This changes cost, not the model's output.
_ACTIVITY_LOG_HEADER = "=== USER DAILY ACTIVITY LOG ==="
_AFTER_LOG_HEADERS = ("\n=== USER PROFILE (ENHANCED) ===", "\n\n=== SURVEY QUESTIONS")
_REFUSAL_STOPS = ("refusal", "content_filtered", "guardrail_intervened")
_RETRYABLE_CODES = ("ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException",
                    "InternalServerException", "ModelTimeoutException", "ModelErrorException")


def _bedrock_client(region: str):
    with _BEDROCK_LOCK:
        client = _BEDROCK_CLIENTS.get(region)
        if client is None:
            import boto3  # lazy import: only needed for provider "bedrock"
            from botocore.config import Config
            client = boto3.client(
                "bedrock-runtime", region_name=region,
                config=Config(read_timeout=1500, connect_timeout=60,
                              retries={"max_attempts": 2, "mode": "standard"}),
            )
            _BEDROCK_CLIENTS[region] = client
        return client


def _bedrock_user_blocks(text: str, cache_point: bool, img_paths_resolved: list[Path]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = [{"text": text}]
    if cache_point and _ACTIVITY_LOG_HEADER in text:
        cut = text.find(_AFTER_LOG_HEADERS[0])
        if cut < 0:
            cut = text.find(_AFTER_LOG_HEADERS[1])
        if cut > 20000:
            blocks = [{"text": text[:cut]}, {"cachePoint": {"type": "default"}}, {"text": text[cut:]}]
    for p in img_paths_resolved:
        fmt = _BEDROCK_IMAGE_FORMATS.get(p.suffix.lower().lstrip("."))
        if fmt is None:
            raise ValueError(f"Bedrock accepts jpeg/png/gif/webp images, got {p.name}")
        blocks.append({"image": {"format": fmt, "source": {"bytes": p.read_bytes()}}})
    return blocks


def _call_bedrock(
    *,
    model_id: str,
    system: str,
    user: str,
    max_tokens: int,
    img_paths_resolved: list[Path],
    region: str | None,
    additional_request_fields: dict | None,
    cache_point: bool,
    cache_key: str,
    cache_path: Path,
) -> dict[str, Any]:
    """Call a Bedrock model through the Converse API.

    This is the protocol used for the Bedrock-served models in the paper:
    no temperature is sent (several reasoning models reject it), output is
    capped by `maxTokens`, and the per-model reasoning setting is passed through
    `additionalModelRequestFields`. JSON mode does not exist on Converse; the
    answer parser reads the text. A model that rejects a system prompt gets it
    prepended to the user message. Empty responses are retried, and only
    non-empty responses are cached.

    Returns the same normalized dict shape as the other providers.
    """
    from botocore.exceptions import ClientError

    region = region or "us-west-2"
    client = _bedrock_client(region)
    merge_system = False
    backoff = 10
    for attempt in range(1, 15):
        text = (system + "\n\n" + user) if merge_system else user
        kwargs: dict[str, Any] = {
            "modelId": model_id,
            "messages": [{"role": "user", "content": _bedrock_user_blocks(text, cache_point, img_paths_resolved)}],
            "inferenceConfig": {"maxTokens": max_tokens},
        }
        if not merge_system:
            kwargs["system"] = [{"text": system}]
        if additional_request_fields:
            kwargs["additionalModelRequestFields"] = additional_request_fields
        t0 = time.time()
        try:
            resp = client.converse(**kwargs)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            msg = str(e).lower()
            if code == "ValidationException" and "system" in msg and not merge_system:
                merge_system = True
                continue
            if code == "ValidationException" and cache_point and ("cachepoint" in msg or "cache point" in msg or "caching" in msg):
                cache_point = False
                continue
            if code not in _RETRYABLE_CODES:
                raise
            print(f"  [bedrock {code}] sleep {backoff}s and retry...", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
            continue
        except Exception as e:  # read timeouts, connection resets
            print(f"  [bedrock {type(e).__name__}] sleep {backoff}s and retry...", flush=True)
            time.sleep(backoff)
            backoff = min(backoff * 2, 300)
            continue
        elapsed = time.time() - t0
        blocks = (resp.get("output") or {}).get("message", {}).get("content", []) or []
        content = "".join(b.get("text", "") for b in blocks if "text" in b)
        stop = resp.get("stopReason")
        if not content.strip():
            if stop in _REFUSAL_STOPS and attempt >= 2:
                break
            if stop == "max_tokens" and max_tokens < 32000:
                max_tokens = 32000
                continue
            time.sleep(5)
            continue
        usage = resp.get("usage") or {}
        rec: dict[str, Any] = {
            "model_id": model_id,
            "content": content,
            "prompt_tokens": usage.get("inputTokens", 0),
            "completion_tokens": usage.get("outputTokens", 0),
            "total_tokens": usage.get("totalTokens", 0),
            "cache_read_tokens": usage.get("cacheReadInputTokens", 0),
            "cache_write_tokens": usage.get("cacheWriteInputTokens", 0),
            "elapsed_s": round(elapsed, 3),
            "stop_reason": stop,
            "cache_key": cache_key,
            "cached": False,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(f".tmp.{os.getpid()}.{threading.get_ident()}")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False, indent=2)
        tmp.replace(cache_path)
        return rec
    # Refused or still empty after retries: return an empty, uncached record
    # (the parser then counts every question as missing).
    return {"model_id": model_id, "content": "", "prompt_tokens": 0, "completion_tokens": 0,
            "total_tokens": 0, "elapsed_s": 0.0, "cache_key": cache_key, "cached": False,
            "empty_response": True}
