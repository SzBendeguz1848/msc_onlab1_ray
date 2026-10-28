"""Ray Serve prefix-affinity proxy in front of llm-d-inference-sim backends.

Chat completions are routed by the native PrefixCacheAffinityRouter. The
Ingress turns the OpenAI JSON body into a SimpleNamespace so the router can
read .messages / .prompt. Health and /v1/models bypass the prefix router and
go to the first sim.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, AsyncIterator

import httpx
import ray
from ray import serve
from ray.serve.config import RequestRouterConfig
from ray.serve.handle import DeploymentHandle
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import StreamingResponse

from completion_capture import append_capture_record, capture_enabled_for_path

# Public ray.serve.llm.request_router imports vLLM/SGLang. The internal
# module is enough for PrefixCacheAffinityRouter on the Proxy→sim path.
try:
    from ray.serve.llm.request_router import PrefixCacheAffinityRouter
except ImportError:
    from ray.llm._internal.serve.routing_policies.prefix_aware.prefix_aware_router import (
        PrefixCacheAffinityRouter,
    )


logger = logging.getLogger(__name__)

_DROP_REQUEST_HEADERS = frozenset({"host", "content-length"})
_DROP_RESPONSE_HEADERS = frozenset(
    {"content-length", "transfer-encoding", "connection"}
)
_CHAT_PATHS = frozenset({"/v1/chat/completions", "/v1/completions"})


@dataclass(frozen=True)
class ServeSettings:
    """Runtime settings loaded from environment variables."""

    backend_urls: list[str]
    num_replicas: int
    imbalanced_threshold: int
    match_rate_threshold: float


def load_settings() -> ServeSettings:
    """Parse Serve settings from the process environment.

    Returns:
        ServeSettings: Backend URLs, replica count, and prefix-router kwargs.

    Raises:
        KeyError: If BACKEND_URLS is missing.
        ValueError: If BACKEND_URLS is empty or a numeric env is invalid.
    """
    raw_urls = os.environ["BACKEND_URLS"]
    backend_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]
    if not backend_urls:
        raise ValueError("BACKEND_URLS must contain at least one URL")
    return ServeSettings(
        backend_urls=backend_urls,
        num_replicas=int(os.environ.get("NUM_REPLICAS", str(len(backend_urls)))),
        imbalanced_threshold=int(os.environ.get("PREFIX_IMBALANCED_THRESHOLD", "5")),
        match_rate_threshold=float(
            os.environ.get("PREFIX_MATCH_RATE_THRESHOLD", "0.15")
        ),
    )


SETTINGS = load_settings()


def routing_payload(body: bytes) -> SimpleNamespace | None:
    """Build the prefix-router arg from an OpenAI JSON body.

    Args:
        body: Raw request body bytes.

    Returns:
        SimpleNamespace with the JSON fields if messages or prompt is set,
        otherwise None so the caller can skip the prefix router.
    """
    if not body:
        return None
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    if data.get("messages") is None and data.get("prompt") is None:
        return None
    return SimpleNamespace(**data)


def request_headers(request: Request) -> dict[str, str]:
    """Copy inbound headers, dropping hop-by-hop fields.

    Args:
        request: Incoming Starlette request.

    Returns:
        Headers safe to forward to a sim.
    """
    return {
        key: value
        for key, value in request.headers.items()
        if key.lower() not in _DROP_REQUEST_HEADERS
    }


def filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    """Drop hop-by-hop response headers.

    Args:
        headers: Upstream response headers.

    Returns:
        Headers safe to send to the client.
    """
    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in _DROP_RESPONSE_HEADERS
    }


def build_router_config() -> RequestRouterConfig:
    """Configure the native PrefixCacheAffinityRouter.

    Returns:
        RequestRouterConfig pointing at PrefixCacheAffinityRouter.
    """
    return RequestRouterConfig(
        request_router_class=PrefixCacheAffinityRouter,
        request_router_kwargs={
            "imbalanced_threshold": SETTINGS.imbalanced_threshold,
            "match_rate_threshold": SETTINGS.match_rate_threshold,
            "do_eviction": True,
        },
    )


@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0})
class BackendAssigner:
    """Assigns each Proxy replica a unique sim backend URL (round-robin claim)."""

    def __init__(self) -> None:
        """Initialize the claim counter over the configured backend URLs."""
        self._next_index = 0
        self._urls = SETTINGS.backend_urls

    def claim(self) -> str:
        """Claim the next backend URL for a Proxy replica.

        Returns:
            str: Absolute backend base URL (e.g. http://sim-0:8000).
        """
        url = self._urls[self._next_index % len(self._urls)]
        self._next_index += 1
        logger.info(f"claimed backend {url}")
        return url


@serve.deployment(
    num_replicas=SETTINGS.num_replicas,
    max_ongoing_requests=32,
    request_router_config=build_router_config(),
)
class Proxy:
    """Forwards a chat request to one pinned sim; selected by PrefixCache."""

    async def __init__(self, assigner: DeploymentHandle) -> None:
        """Claim a backend URL and open a persistent async HTTP client.

        Args:
            assigner: Handle to the BackendAssigner deployment.
        """
        self.backend_url: str = await assigner.claim.remote()
        self._client = httpx.AsyncClient(base_url=self.backend_url, timeout=None)
        logger.info(f"proxy pinned to {self.backend_url}")

    async def forward(
        self, _routing: Any, fwd: dict[str, Any]
    ) -> AsyncIterator[Any]:
        """Stream upstream response: metadata dict, then raw body chunks.

        Args:
            _routing: SimpleNamespace with .messages / .prompt for the router.
            fwd: method, path, query, headers, body for the sim.

        Yields:
            First a dict with status_code and headers, then body byte chunks.
        """
        upstream_url = httpx.URL(path=fwd["path"], query=fwd["query"])
        upstream = await self._client.send(
            self._client.build_request(
                fwd["method"],
                upstream_url,
                headers=fwd["headers"],
                content=fwd["body"],
            ),
            stream=True,
        )
        path = str(fwd.get("path") or "")
        filtered = filter_response_headers(upstream.headers)
        pod = ""
        for key, value in upstream.headers.items():
            if key.lower() == "x-inference-pod":
                pod = value
                break
        backend_url = self.backend_url
        try:
            yield {
                "status_code": upstream.status_code,
                "headers": filtered,
            }
            buf = bytearray()
            async for chunk in upstream.aiter_raw():
                if capture_enabled_for_path(path):
                    buf.extend(chunk)
                yield chunk
            if capture_enabled_for_path(path):
                append_capture_record(
                    backend_url=backend_url,
                    pod=pod,
                    path=path,
                    raw_body=bytes(buf),
                )
        finally:
            await upstream.aclose()


@serve.deployment(
    num_replicas=1,
    max_ongoing_requests=64,
    ray_actor_options={"num_cpus": 0},
)
class Ingress:
    """HTTP entry: prefix-route chat, otherwise forward to the first sim."""

    def __init__(self, proxy: DeploymentHandle) -> None:
        """Store a streaming handle to Proxy and a client for non-chat paths.

        Args:
            proxy: Handle to the multi-replica Proxy deployment.
        """
        self._proxy = proxy.options(stream=True, method_name="forward")
        self._direct = httpx.AsyncClient(
            base_url=SETTINGS.backend_urls[0], timeout=None
        )

    async def __call__(self, request: Request) -> StreamingResponse:
        """Route chat via PrefixCache, other paths straight to sim-0.

        Args:
            request: Incoming Starlette/ASGI request.

        Returns:
            StreamingResponse: Upstream body streamed for GuideLLM TTFT.
        """
        body = await request.body()
        payload = routing_payload(body)
        headers = request_headers(request)
        if payload is None or request.url.path not in _CHAT_PATHS:
            return await self._direct_forward(request, headers, body)
        fwd = {
            "method": request.method,
            "path": request.url.path,
            "query": request.url.query.encode("utf-8"),
            "headers": headers,
            "body": body,
        }
        gen = self._proxy.remote(payload, fwd)
        meta = await gen.__anext__()
        status_code = int(meta["status_code"])
        response_headers = dict(meta["headers"])

        async def _body() -> AsyncIterator[bytes]:
            async for chunk in gen:
                yield chunk

        async def _cleanup() -> None:
            if not gen.cancelled():
                gen.cancel()

        return StreamingResponse(
            _body(),
            status_code=status_code,
            headers=response_headers,
            background=BackgroundTask(_cleanup),
        )

    async def _direct_forward(
        self,
        request: Request,
        headers: dict[str, str],
        body: bytes,
    ) -> StreamingResponse:
        """Proxy a non-chat request to the first backend.

        Args:
            request: Incoming request (path and query reused).
            headers: Filtered request headers.
            body: Raw request body.

        Returns:
            StreamingResponse from the first sim.
        """
        upstream_url = httpx.URL(
            path=request.url.path,
            query=request.url.query.encode("utf-8"),
        )
        upstream = await self._direct.send(
            self._direct.build_request(
                request.method, upstream_url, headers=headers, content=body
            ),
            stream=True,
        )
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers=filter_response_headers(upstream.headers),
            background=BackgroundTask(upstream.aclose),
        )


def main() -> None:
    """Start Ray Serve with Ingress → PrefixCache Proxy and block."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info(
        "PrefixCacheAffinityRouter on Ingress→Proxy→sim "
        f"(imbalanced_threshold={SETTINGS.imbalanced_threshold}, "
        f"match_rate_threshold={SETTINGS.match_rate_threshold})"
    )
    if not ray.is_initialized():
        ray.init(
            dashboard_host="0.0.0.0",
            include_dashboard=True,
            _metrics_export_port=8080,
        )
    logger.info("Ray Dashboard: http://localhost:8265")
    serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})
    serve.run(
        Ingress.bind(Proxy.bind(BackendAssigner.bind())),
        route_prefix="/",
        blocking=True,
    )


if __name__ == "__main__":
    main()
