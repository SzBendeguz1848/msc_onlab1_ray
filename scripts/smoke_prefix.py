"""Smoke the prefix stack: health, models, then the same prompt twice.

Run on the host against an already-up Compose stack. Does not start Docker.

    python scripts/smoke_prefix.py --target http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SMOKE_PROMPT = "PrefixSmoke: the quick brown fox"


def _open(url: str, data: bytes | None = None, timeout: float = 120) -> tuple[int, dict[str, str], bytes]:
    """HTTP GET or POST and return status, headers, body.

    Args:
        url: Absolute URL.
        data: POST body, or None for GET.
        timeout: Socket timeout in seconds.

    Returns:
        Tuple of status code, lowercased header map, and raw body.

    Raises:
        RuntimeError: If the target is unreachable or returns HTTP error.
    """
    headers = {"Content-Type": "application/json"} if data is not None else {}
    req = Request(url, data=data, headers=headers, method="POST" if data else "GET")
    try:
        with urlopen(req, timeout=timeout) as resp:
            status = getattr(resp, "status", None) or resp.getcode()
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            return int(status), hdrs, resp.read()
    except HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} at {url}: {err_body}") from exc
    except URLError as exc:
        raise RuntimeError(f"unreachable {url}: {exc.reason}") from exc


def cached_tokens(resp: dict[str, Any]) -> int:
    """Read cached_tokens from a chat.completion usage object.

    Args:
        resp: Parsed chat.completion JSON.

    Returns:
        cached_tokens, or 0 if the field is missing.
    """
    usage = resp.get("usage") or {}
    detail = (
        usage.get("prompt_tokens_details")
        or usage.get("prompt_tokens_detail")
        or {}
    )
    value = detail.get("cached_tokens")
    return int(value) if value is not None else 0


def chat_once(target: str) -> tuple[str, dict[str, Any], int]:
    """POST the smoke prompt once.

    Args:
        target: OpenAI-compatible base URL.

    Returns:
        Tuple of x-inference-pod (or empty), parsed JSON, and cached_tokens.
    """
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": SMOKE_PROMPT}],
        "max_tokens": 32,
        "stream": False,
    }
    status, hdrs, raw = _open(
        target.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
    )
    if status != 200:
        raise RuntimeError(f"chat completion HTTP {status}: {raw[:500]!r}")
    data = json.loads(raw.decode("utf-8"))
    pod = hdrs.get("x-inference-pod", "")
    return pod, data, cached_tokens(data)


def run(target: str) -> None:
    """Hit health, models, then the same chat prompt twice and assert affinity.

    Args:
        target: OpenAI-compatible base URL (Serve ingress).

    Raises:
        RuntimeError: If any check fails.
    """
    base = target.rstrip("/")
    health_status, _, _ = _open(base + "/health")
    if health_status != 200:
        raise RuntimeError(f"/health HTTP {health_status}")
    models_status, _, models_raw = _open(base + "/v1/models")
    if models_status != 200:
        raise RuntimeError(f"/v1/models HTTP {models_status}")
    logger.info("health=200 models=200 %s", models_raw[:120])

    pod1, r1, cache1 = chat_once(base)
    pod2, r2, cache2 = chat_once(base)
    usage1 = r1.get("usage")
    usage2 = r2.get("usage")
    logger.info("request #1 pod=%s cached_tokens=%s usage=%s", pod1, cache1, usage1)
    logger.info("request #2 pod=%s cached_tokens=%s usage=%s", pod2, cache2, usage2)

    if not pod1 or not pod2:
        raise RuntimeError(
            "missing x-inference-pod (set POD_NAME on each sim)"
        )
    if pod1 != pod2:
        raise RuntimeError(f"prefix affinity failed: {pod1!r} vs {pod2!r}")
    if cache2 <= cache1 and cache2 == 0:
        raise RuntimeError(
            f"cached_tokens did not increase: #1={cache1} #2={cache2}"
        )
    if cache2 < cache1:
        raise RuntimeError(
            f"cached_tokens dropped: #1={cache1} #2={cache2}"
        )
    logger.info("ok: same pod %s, cached_tokens %s -> %s", pod1, cache1, cache2)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: parse --target and run the smoke checks.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        Process exit code (0 success, 1 failure).
    """
    parser = argparse.ArgumentParser(description="Smoke the prefix Serve stack.")
    parser.add_argument("--target", default="http://127.0.0.1:8000")
    ns = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        run(ns.target)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
