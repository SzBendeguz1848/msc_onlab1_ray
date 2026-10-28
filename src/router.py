"""Ray Serve HTTP proxy that routes requests to llm-d-inference-sim backends.
Target: Ray 2.58.0. Replica selection is controlled by the ROUTER env var
(p2c | round_robin | consistent_hash). Each Proxy replica pins to one sim URL
via BackendAssigner at init time.
"""
from __future__ import annotations
import logging
import os
from collections.abc import AsyncIterator
from dataclasses import dataclass
import httpx
import ray
from ray import serve
from ray.serve.config import RequestRouterConfig
from ray.serve.handle import DeploymentHandle
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import StreamingResponse

from ray.serve.experimental.round_robin_router import RoundRobinRouter
from ray.serve.experimental.consistent_hash_router import ConsistentHashRouter
from ray.serve._private.request_router.pow_2_router import PowerOfTwoChoicesRequestRouter

from completion_capture import append_capture_record, capture_enabled_for_path

logger = logging.getLogger(__name__)


# Hop-by-hop headers must not be forwarded (httpx recomputes Content-Length).
_DROP_REQUEST_HEADERS = frozenset({"host", "content-length"})
_DROP_RESPONSE_HEADERS = frozenset(
    {"content-length", "transfer-encoding", "connection"}
)


@dataclass(frozen=True)
class ServeSettings:
    """Runtime settings loaded from environment variables."""
    router: str
    backend_urls: list[str]
    num_replicas: int

def load_settings() -> ServeSettings:
    """Parse Serve settings from the process environment.
    Returns:
        ServeSettings: Router name, backend URL list, and replica count.
    Raises:
        KeyError: If BACKEND_URLS is missing.
        ValueError: If BACKEND_URLS is empty or NUM_REPLICAS is not an int.
    """
    raw_urls = os.environ["BACKEND_URLS"]
    backend_urls = [u.strip() for u in raw_urls.split(",") if u.strip()]
    if not backend_urls:
        raise ValueError("BACKEND_URLS must contain at least one URL")
    num_replicas = int(os.environ.get("NUM_REPLICAS", str(len(backend_urls))))
    return ServeSettings(
        router=os.environ.get("ROUTER", "p2c"),
        backend_urls=backend_urls,
        num_replicas=num_replicas,
    )

SETTINGS = load_settings()

def build_router_config(router_name: str) -> RequestRouterConfig | None:
    """Map ROUTER env value to a Ray Serve RequestRouterConfig.
    Args:
        router_name: One of p2c, round_robin, consistent_hash.
    Returns:
        RequestRouterConfig for non-default routers, or None for P2C
        (Serve's built-in PowerOfTwoChoicesRequestRouter).
    Raises:
        ValueError: If router_name is not a known policy.
    """
    if router_name == "p2c":
        return None
    if router_name == "round_robin":
        return RequestRouterConfig(
            request_router_class=(
                "ray.serve.experimental.round_robin_router:RoundRobinRouter"
            ),
        )
    if router_name == "consistent_hash":
        return RequestRouterConfig(
            request_router_class=(
                "ray.serve.experimental.consistent_hash_router:"
                "ConsistentHashRouter"
            ),
            request_router_kwargs={
                "num_virtual_nodes": 100,
                "num_fallback_replicas": 2,
            },
        )
    raise ValueError(f"unknown ROUTER={router_name!r}")

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

_router_cfg = build_router_config(SETTINGS.router)
_proxy_kwargs: dict[str, object] = {"num_replicas": SETTINGS.num_replicas}
if _router_cfg is not None:
    _proxy_kwargs["request_router_config"] = _router_cfg

@serve.deployment(**_proxy_kwargs)
class Proxy:
    """Streaming HTTP reverse-proxy from Serve to one pinned sim backend."""
    async def __init__(self, assigner: DeploymentHandle) -> None:
        """Claim a backend URL and open a persistent async HTTP client.
        Args:
            assigner: Handle to the BackendAssigner deployment.
        Notes:
            __call__ is async, so Serve initializes replicas on the asyncio
            loop; await the remote claim (do not use .result()).
        """
        self.backend_url: str = await assigner.claim.remote()
        self._client = httpx.AsyncClient(base_url=self.backend_url, timeout=None)
        logger.info(f"proxy pinned to {self.backend_url}")

    async def __call__(self, request: Request) -> StreamingResponse:
        """Forward the incoming request to the pinned sim and stream the reply.
        Args:
            request: Incoming Starlette/ASGI request (OpenAI-compatible paths).
        Returns:
            StreamingResponse: Upstream body streamed so GuideLLM can measure
            TTFT; connection closed via BackgroundTask.
        """
        # Preserve path + query so /v1/models and chat completions both work.
        upstream_url = httpx.URL(
            path=request.url.path,
            query=request.url.query.encode("utf-8"),
        )
        body = await request.body()
        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in _DROP_REQUEST_HEADERS
        }
        # x-session-id is kept for ConsistentHash when the client sends it.
        upstream = await self._client.send(
            self._client.build_request(
                request.method, upstream_url, headers=headers, content=body
            ),
            stream=True,
        )
        response_headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in _DROP_RESPONSE_HEADERS
        }
        path = request.url.path
        pod = ""
        for key, value in upstream.headers.items():
            if key.lower() == "x-inference-pod":
                pod = value
                break
        backend_url = self.backend_url

        async def _tee() -> AsyncIterator[bytes]:
            buf = bytearray()
            try:
                async for chunk in upstream.aiter_raw():
                    if capture_enabled_for_path(path):
                        buf.extend(chunk)
                    yield chunk
            finally:
                if capture_enabled_for_path(path):
                    append_capture_record(
                        backend_url=backend_url,
                        pod=pod,
                        path=path,
                        raw_body=bytes(buf),
                    )
                await upstream.aclose()

        return StreamingResponse(
            _tee(),
            status_code=upstream.status_code,
            headers=response_headers,
        )


def main() -> None:
    """Start Ray Serve with dashboard on 0.0.0.0 and block on the app."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # serve.start() would bind the dashboard to 127.0.0.1; Docker port-publish
    # only reaches eth0, so init Ray with an all-interfaces dashboard first.
    # Pin metrics export so Prometheus can scrape ray-serve:8080 statically.
    if not ray.is_initialized():
        ray.init(
            dashboard_host="0.0.0.0",
            include_dashboard=True,
            _metrics_export_port=8080,
        )
    logger.info("Ray Dashboard: http://localhost:8265")
    logger.info("Prometheus: http://localhost:9090")
    logger.info("Grafana: http://localhost:3000 (admin/admin)")
    serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})
    serve.run(
        Proxy.bind(BackendAssigner.bind()),
        route_prefix="/",
        blocking=True,
    )

if __name__ == "__main__":
    main()
