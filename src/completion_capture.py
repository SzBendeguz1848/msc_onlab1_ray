"""Capture streamed/non-streamed OpenAI chat.completion bodies to JSONL.

Used by router.py and prefix_router.py Proxy replicas. Each replica writes its
own file under CAPTURE_DIR to avoid concurrent-append races. GuideLLM keeps
streaming for TTFT; we tee the body and reassemble SSE into a chat.completion
dict (usage.cached_tokens included when the sim sends it).
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_DEFAULT_CAPTURE_DIR = "/app/results/_capture"
_CHAT_PATHS = frozenset({"/v1/chat/completions", "/v1/completions"})


def capture_enabled_for_path(path: str) -> bool:
    """Return True if this request path should be captured.

    Args:
        path: URL path (e.g. /v1/chat/completions).

    Returns:
        Whether to tee and persist the upstream body.
    """
    return path in _CHAT_PATHS


def capture_dir() -> Path | None:
    """Resolve CAPTURE_DIR from the environment.

    Returns:
        Path to the capture directory, or None if CAPTURE_DIR is empty/unset
        in a way that disables capture. Empty string disables; unset uses default.
    """
    raw = os.environ.get("CAPTURE_DIR", _DEFAULT_CAPTURE_DIR)
    if raw.strip() == "":
        return None
    return Path(raw)


def replica_id_from_backend(backend_url: str) -> str:
    """Derive a filesystem-safe replica id from a backend base URL.

    Args:
        backend_url: e.g. http://sim-3:8000.

    Returns:
        Hostname or a sanitized fallback string.
    """
    host = urlparse(backend_url).hostname
    if host:
        return re.sub(r"[^\w.-]", "_", host)
    return re.sub(r"[^\w.-]", "_", backend_url) or "unknown"


def assemble_chat_completion(raw_body: bytes) -> dict[str, Any]:
    """Turn a raw upstream body into a chat.completion-shaped dict.

    Tries a single JSON object first (stream=false). Otherwise parses OpenAI
    SSE (`data: {...}`) and merges deltas + usage.

    Args:
        raw_body: Bytes buffered while teeing the upstream stream.

    Returns:
        A dict resembling OpenAI chat.completion (may be partial on errors).
    """
    text = raw_body.decode("utf-8", errors="replace").strip()
    if not text:
        return {
            "object": "chat.completion",
            "choices": [],
            "usage": None,
            "_capture_error": "empty_body",
        }
    if not text.startswith("data:"):
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return _assemble_from_sse(text)


def _assemble_from_sse(text: str) -> dict[str, Any]:
    """Merge OpenAI chat.completion SSE chunks into one object.

    Args:
        text: Full SSE payload as text.

    Returns:
        Assembled chat.completion dict.
    """
    content_parts: list[str] = []
    finish_reason: str | None = None
    usage: dict[str, Any] | None = None
    base: dict[str, Any] = {"object": "chat.completion"}
    saw_data = False

    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        saw_data = True
        try:
            obj = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        for key in ("id", "model", "created", "object", "kv_transfer_params"):
            if key in obj:
                base[key] = obj[key]
        if obj.get("usage"):
            usage = obj["usage"]
        for choice in obj.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta")
            if isinstance(delta, dict):
                part = delta.get("content")
                if part:
                    content_parts.append(str(part))
            message = choice.get("message")
            if isinstance(message, dict) and message.get("content"):
                content_parts.append(str(message["content"]))
            text_field = choice.get("text")
            if text_field:
                content_parts.append(str(text_field))

    if not saw_data:
        return {
            "object": "chat.completion",
            "choices": [],
            "usage": None,
            "_capture_error": "unrecognized_body",
            "_raw_preview": text[:500],
        }

    base["choices"] = [
        {
            "index": 0,
            "finish_reason": finish_reason or "stop",
            "message": {
                "role": "assistant",
                "content": "".join(content_parts),
            },
        }
    ]
    if usage is not None:
        base["usage"] = usage
    return base


def append_capture_record(
    *,
    backend_url: str,
    pod: str,
    path: str,
    raw_body: bytes,
) -> None:
    """Append one reassembled completion as a JSONL line for this replica.

    Args:
        backend_url: Proxy's pinned sim base URL.
        pod: Value of x-inference-pod (may be empty).
        path: Request path that was forwarded.
        raw_body: Buffered upstream response body.
    """
    if not capture_enabled_for_path(path):
        return
    directory = capture_dir()
    if directory is None:
        return
    try:
        directory.mkdir(parents=True, exist_ok=True)
        record = {
            "pod": pod,
            "backend": backend_url,
            "path": path,
            "chat_completion": assemble_chat_completion(raw_body),
        }
        replica = replica_id_from_backend(backend_url)
        out = directory / f"{replica}.jsonl"
        with out.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        logger.warning("completion capture failed: %s", exc)
