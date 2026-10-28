---
name: Ray Router Benchmark Architecture
overview: GuideLLM on host; Ray Serve and 6–8 llm-d-inference-sim backends in Docker. Phase 1 benchmarks three built-in routers (P2C, RoundRobin, ConsistentHash) under identical ShareGPT traffic. Custom Prefix/KV-aware routers deferred.
todos:
  - id: pyproject
    content: "Step 1: pyproject.toml — host + serve optional-deps, package metadata"
    status: pending
  - id: backends
    content: "Step 2: Dockerfile + docker-compose.yml (ray-serve + sim-0..7, default-v1 flags)"
    status: pending
  - id: serve-proxy
    content: "Step 3–4: src/router.py — Proxy + BackendAssigner + ROUTER→RequestRouterConfig"
    status: pending
  - id: builtin-routers
    content: "Step 3–4: covered in router.py (p2c / round_robin / consistent_hash)"
    status: cancelled
  - id: guidellm
    content: "Step 5: scripts/run_benchmark.py (prepare | run) → results/.../guidellm.json"
    status: pending
  - id: evaluate
    content: "Step 6: evaluate.ipynb — primary metrics tables + inline plots"
    status: pending
isProject: false
---

# Ray Serve Router Benchmark

## Goal

Compare three built-in Ray Serve request routers under identical load against mock LLM backends, without KubeRay, GPUs, or real model weights.

Flow: generator → router selects a replica → replica forwards to its backend. Run artifacts land in results; evaluate reads them.

## Repository layout

```
msc_onlab1_ray/
  Dockerfile                 # Ray Serve image (.[serve])
  docker-compose.yml         # ray-serve + sim-0..7
  pyproject.toml
  README.md
  data/
    sharegpt_500.jsonl
  scripts/
    run_benchmark.py         # prepare | run (GuideLLM → guidellm.json)
  src/
    router.py
  evaluate.ipynb             # tables + inline plots from guidellm.json
  results/
    <run-id>/
      guidellm.json
    plots/                   # optional PNG exports from the notebook
```

## Components

### create `[pyproject.toml](pyproject.toml)`

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "msc-onlab1-ray"
version = "0.1.0"
description = "Ray Serve request-router benchmark against llm-d-inference-sim"
requires-python = ">=3.10"
dependencies = []

[project.optional-dependencies]
host = [
  "guidellm[recommended]",
  "huggingface_hub",
  "pandas",
  "matplotlib",
  "jupyter",                # evaluate.ipynb
]
serve = [
  "ray[serve]==2.58.0",  # pin latest stable (PyPI); RoundRobin/ConsistentHash experimental APIs
  "httpx",
]

[tool.setuptools.packages.find]
where = ["src"]
```

### Traffic generator — GuideLLM

One script: [`scripts/run_benchmark.py`](scripts/run_benchmark.py).

| Command | Meaning |
|---|---|
| `python scripts/run_benchmark.py --prepare` | Build `data/sharegpt_500.jsonl` (once / refresh) |
| `python scripts/run_benchmark.py` | Run GuideLLM (auto-prepare if jsonl missing) |
| `python scripts/run_benchmark.py --run-id rr` | Same, write `results/rr/guidellm.json` |
| `python scripts/run_benchmark.py --session` | Add `x-session-id` (ConsistentHash) |

```python
# scripts/run_benchmark.py
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "sharegpt_500.jsonl"
HF_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
HF_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"
HF_REVISION = "main"  # pin to commit hash after first successful prepare
N_PROMPTS = 500


def prepare() -> None:
    path = hf_hub_download(
        repo_id=HF_REPO, filename=HF_FILE, repo_type="dataset", revision=HF_REVISION
    )
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    DATA.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with DATA.open("w", encoding="utf-8") as out:
        for row in rows:
            for turn in row.get("conversations") or []:
                role = (turn.get("from") or turn.get("role") or "").lower()
                if role not in ("human", "user"):
                    continue
                text = turn.get("value") or turn.get("content")
                if not text:
                    break
                out.write(json.dumps({"prompt": text}, ensure_ascii=False) + "\n")
                n += 1
                break
            if n >= N_PROMPTS:
                break
    if n < N_PROMPTS:
        raise SystemExit(f"only {n} prompts")
    print(f"wrote {n} -> {DATA}")


def run(args: argparse.Namespace) -> None:
    if not DATA.is_file():
        prepare()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "results" / run_id
    out.mkdir(parents=True, exist_ok=True)
    backend = f"kind=openai_http,target={args.target}"
    if args.session:
        backend += ",extras.headers.x-session-id=bench-session"
    subprocess.check_call([
        "guidellm", "run",
        "--backend", backend,
        "--data", f"kind=json_file,path={DATA}",
        "--data-column-mapper",
        "kind=generative_column_mapper,column_mappings.text_column=prompt",
        "--data-loader", "kind=pytorch,samples=500,shuffle=false",
        "--profile", f"kind=constant,rate={args.rate}",
        "--constraint", "kind=max_requests,count=500",
        "--output", f"kind=json,path={out / 'guidellm.json'}",
    ], cwd=ROOT)
    print(f"wrote {out / 'guidellm.json'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--prepare", action="store_true", help="only rebuild jsonl")
    p.add_argument("--target", default="http://127.0.0.1:8000")
    p.add_argument("--rate", type=float, default=5.0)
    p.add_argument("--run-id", default=None)
    p.add_argument("--session", action="store_true")
    args = p.parse_args()
    if args.prepare:
        prepare()
        return
    run(args)


if __name__ == "__main__":
    main()
```

### Ray Serve — Docker image + Compose

Topology in Docker: one `ray-serve` service + eight identical `sim-*` services on one bridge network. GuideLLM on the host hits `localhost:8000`.

#### `[Dockerfile](Dockerfile)` (Ray Serve image)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[serve]"
ENV PYTHONUNBUFFERED=1
# router.py starts Ray + serve.run; binds Serve HTTP on 8000
CMD ["python", "src/router.py"]
```

- Installs only `.[serve]` (`ray[serve]`, `httpx`) — not GuideLLM.
- `src/router.py` (next step) reads `ROUTER`, `NUM_REPLICAS`, `BACKEND_URLS` from env.

#### `[docker-compose.yml](docker-compose.yml)`

Default **N = 8**. Shared sim baseline = `backend_profile: default-v1`. YAML anchor avoids copying flags eight times.

```yaml
x-sim: &sim
  image: ghcr.io/llm-d/llm-d-inference-sim:v0.9.0
  command:
    - --port
    - "8000"
    - --model
    - fake-model
    - --time-to-first-token
    - "100"          # ms — non-zero so routers differ under load
    - --inter-token-latency
    - "20"           # ms
    - --max-num-seqs
    - "4"            # concurrency cap → waiting queue can form
    - --max-waiting-queue-length
    - "1000"
  restart: unless-stopped

services:
  ray-serve:
    build: .
    ports:
      - "8000:8000"
    environment:
      ROUTER: ${ROUTER:-p2c}           # p2c| round_robin | consistent_hash
      NUM_REPLICAS: "8"
      BACKEND_URLS: "http://sim-0:8000,http://sim-1:8000,http://sim-2:8000,http://sim-3:8000,http://sim-4:8000,http://sim-5:8000,http://sim-6:8000,http://sim-7:8000"
    depends_on:
      - sim-0
      - sim-1
      - sim-2
      - sim-3
      - sim-4
      - sim-5
      - sim-6
      - sim-7

  sim-0: { <<: *sim, ports: ["8001:8000"] }
  sim-1: { <<: *sim, ports: ["8002:8000"] }
  sim-2: { <<: *sim, ports: ["8003:8000"] }
  sim-3: { <<: *sim, ports: ["8004:8000"] }
  sim-4: { <<: *sim, ports: ["8005:8000"] }
  sim-5: { <<: *sim, ports: ["8006:8000"] }
  sim-6: { <<: *sim, ports: ["8007:8000"] }
  sim-7: { <<: *sim, ports: ["8008:8000"] }
```

**Serve app — `[src/router.py](src/router.py)`** (Ray `**2.58.0**`)

Docs (same release): [Custom request router](https://docs.ray.io/en/releases-2.58.0/serve/advanced-guides/custom-request-router.html), [HTTP / FastAPI](https://docs.ray.io/en/releases-2.58.0/serve/http-guide.html). Replica selection ≠ Serve LLM ingress ([routing policies](https://docs.ray.io/en/latest/serve/llm/architecture/routing-policies.html)).

Pinned class paths from 2.58:


| `ROUTER`          | `request_router_class`                                                                                                                                |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `p2c`            | omit config → default `PowerOfTwoChoicesRequestRouter`                                                                                                |
| `round_robin`     | `ray.serve.experimental.round_robin_router:RoundRobinRouter`                                                                                          |
| `consistent_hash` | `ray.serve.experimental.consistent_hash_router:ConsistentHashRouter` + `request_router_kwargs={"num_virtual_nodes": 100, "num_fallback_replicas": 2}` |


```python
# src/router.py — target Ray 2.58.0
import os
import ray
import httpx
from starlette.requests import Request
from starlette.responses import StreamingResponse, Response
from ray import serve
from ray.serve.config import RequestRouterConfig
from ray.serve.handle import DeploymentHandle

ROUTER = os.environ.get("ROUTER", "p2c")
URLS = [u.strip() for u in os.environ["BACKEND_URLS"].split(",") if u.strip()]
N = int(os.environ.get("NUM_REPLICAS", str(len(URLS))))

def _router_config():
    if ROUTER == "p2c":
        return None
    if ROUTER == "round_robin":
        return RequestRouterConfig(
            request_router_class="ray.serve.experimental.round_robin_router:RoundRobinRouter",
        )
    if ROUTER == "consistent_hash":
        return RequestRouterConfig(
            request_router_class="ray.serve.experimental.consistent_hash_router:ConsistentHashRouter",
            request_router_kwargs={"num_virtual_nodes": 100, "num_fallback_replicas": 2},
        )
    raise ValueError(f"unknown ROUTER={ROUTER}")

@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0})
class BackendAssigner:
    def __init__(self):
        self._i = 0
        self._urls = URLS
    def claim(self) -> str:
        url = self._urls[self._i % len(self._urls)]
        self._i += 1
        return url

_cfg = _router_config()
_proxy_kwargs = {"num_replicas": N}
if _cfg is not None:
    _proxy_kwargs["request_router_config"] = _cfg

@serve.deployment(**_proxy_kwargs)
class Proxy:
    def __init__(self, assigner: DeploymentHandle):
        # 2.58 DeploymentHandle: sync result() on remote call
        self.backend_url: str = assigner.claim.remote().result()
        self._client = httpx.AsyncClient(base_url=self.backend_url, timeout=None)

    async def __call__(self, request: Request):
        # 1) Build upstream URL: same path + query as the incoming request
        url = httpx.URL(
            path=request.url.path,
            query=request.url.query.encode("utf-8"),
        )
        # 2) Read body (GuideLLM POST /v1/chat/completions)
        body = await request.body()
        # 3) Copy headers; drop hop-by-hop (host, content-length recomputed by httpx)
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length")
        }
        # keep x-session-id for ConsistentHash
        # 4) Stream from sim so TTFT is visible to GuideLLM
        upstream = await self._client.send(
            self._client.build_request(
                request.method, url, headers=headers, content=body
            ),
            stream=True,
        )
        return StreamingResponse(
            upstream.aiter_raw(),
            status_code=upstream.status_code,
            headers={
                k: v for k, v in upstream.headers.items()
                if k.lower() not in ("content-length", "transfer-encoding", "connection")
            },
            background=None,  # on response close: await upstream.aclose()
        )

def main():
    serve.start(detached=True, http_options={"host": "0.0.0.0", "port": 8000})
    serve.run(Proxy.bind(BackendAssigner.bind()), route_prefix="/", blocking=True)

if __name__ == "__main__":
    main()
```


| Concern        | Choice (2.58)                                                                                           |
| -------------- | ------------------------------------------------------------------------------------------------------- |
| Replica ↔ sim  | `BackendAssigner` (1 replica); each `Proxy` calls `assigner.claim.remote().result()` once in `__init__` |
| HTTP           | Deployment `__call__(Request)` — no FastAPI required for a full-path OpenAI proxy                       |
| Router switch  | `ROUTER` env + recreate container                                                                       |
| ConsistentHash | Client sends `x-session-id` (GuideLLM `extras.headers`); header name default per 2.58 docs              |
| Streaming      | `StreamingResponse` passthrough from sim                                                                |


**Acceptance:** Compose up → `curl localhost:8000/v1/models` through Serve to a sim; `ROUTER=round_robin|consistent_hash|p2c` + recreate changes only replica selection.

### Mock backends — llm-d-inference-sim

Defined entirely by the Compose `x-sim` block above ([llm-d-inference-sim](https://github.com/llm-d/llm-d-inference-sim), image `ghcr.io/llm-d/llm-d-inference-sim:v0.9.0`).

- Identical flags on every sim for all router runs (`default-v1`)
- OpenAI `/v1/chat/completions` built-in

### Evaluation — [`evaluate.ipynb`](evaluate.ipynb)

Single code cell: load `results/<run-id>/guidellm.json` for each router (`p2c`, `round_robin`, `consistent_hash`), build one comparison table + the plots, all shown inline right after that cell. Optional `savefig` → `results/plots/` for the thesis PDF.

| Metric | Report |
|---|---|
| TTFT | mean + p50 / p90 / p99 |
| E2E latency | mean + p50 / p90 / p99 |
| Throughput | RPS, tokens/s |
| Error / timeout rate | failed / incomplete share |
| TPOT / ITL | for the write-up |

## Deferred (later phase)


| Item                              | When / why deferred                                                           |
| --------------------------------- | ----------------------------------------------------------------------------- |
| **Custom Prefix-affinity router** | Needs prompt-aware routing + shared-prefix workload; after baseline pipeline. |
| **Custom KV-aware router**        | Needs ZMQ KV ingest + scoring; after baseline.                                |


## Out of scope (design decisions)


| Item                                                       | Why excluded                                                                                                                                                                                        |
| ---------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **KubeRay / Kubernetes**                                   | Cluster ops do not change router semantics; single-node Compose is enough to compare policies.                                                                                                      |
| **Real LLM weights / GPU**                                 | Supervisor constraint and reproducibility; sim models KV/queue/latency/failures on CPU.                                                                                                             |
| **Native `PrefixCacheAffinityRouter` / `KVAwareRouter`**   | Bound to Serve LLM + GPU engines; not drop-in for HTTP proxy replicas in front of external sims.                                                                                                    |
| `**build_openai_app` / Ray Serve LLM**                     | Couples engine and router; reintroduces hardware dependency.                                                                                                                                        |
| **Prometheus / Grafana / Serve+sim scrape**                | Phase 1 ranks routers from GuideLLM primary metrics only; offline Serve/sim metrics deferred.                                                                                                       |
| `**scenarios/` YAML pack**                                 | Phase 1 uses CLI/env; large sweeps later.                                                                                                                                                           |
| `**DeploymentHandle` / `ray.serve.handle**`                | Used for Python-to-Python calls between Serve deployments (composition). Phase 1 exposes a single HTTP deployment to GuideLLM; replica choice is `request_router_config`, not handle-based fan-out. |
| **llmock**                                                 | No KV/queue/prefix-overlap serving model.                                                                                                                                                           |
| **GuideLLM + `synthetic_text` + `--seed`**                 | Rejected as content lock: seed-dependent prompt generation is not trusted; identical traffic must come from pinned ShareGPT texts.                                                                  |
| **GuideLLM + hand-written / Gemini-generated `json_file`** | Stronger local lock than HF ShareGPT, but extra prep; ShareGPT first-500 slice is enough and already fixed by dataset content.                                                                      |
| **Custom traffic generator + own JSON**                    | Would reimplement tokenization, OpenAI HTTP client, and TTFT/latency logging that GuideLLM already provides; no dependency win once ShareGPT is accepted.                                           |


