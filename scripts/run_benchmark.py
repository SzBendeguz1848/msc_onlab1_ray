"""Prepare ShareGPT prompts and run GuideLLM against the Serve proxy."""
from __future__ import annotations
import argparse
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SHAREGPT_DATA = DATA_DIR / "sharegpt_500.jsonl"
PREFIX_SHARED_DATA = DATA_DIR / "prefix_shared_500.jsonl"
# Back-compat alias used by prepare().
DATA = SHAREGPT_DATA
RESULTS_DIR = ROOT / "results"
SAMPLE_COMPLETION = RESULTS_DIR / "sample_chat_completion.json"
# Must match CAPTURE_DIR in docker-compose (ray-serve + guidellm both mount ./results).
CAPTURE_DIR = Path(os.environ.get("CAPTURE_DIR", str(RESULTS_DIR / "_capture")))
ASSEMBLED_NAME = "assembled_chat_completions.json"

HF_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
HF_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"
# Pin to a commit hash after the first successful prepare for reproducibility.
HF_REVISION = "main"
N_PROMPTS = 500

DATASETS = {
    "sharegpt": SHAREGPT_DATA,
    "prefix_shared": PREFIX_SHARED_DATA,
}

@dataclass(frozen=True)
class BenchmarkArgs:
    """CLI options for a GuideLLM benchmark run."""
    target: str
    rate: float
    run_id: str | None
    use_session: bool
    dataset: str


def resolve_data_path(dataset: str) -> Path:
    """Map --dataset name to a JSONL path under data/.

    Args:
        dataset: One of sharegpt, prefix_shared.

    Returns:
        Absolute path to the JSONL file.

    Raises:
        ValueError: If dataset is unknown.
    """
    try:
        return DATASETS[dataset]
    except KeyError as exc:
        known = ", ".join(sorted(DATASETS))
        raise ValueError(f"unknown dataset {dataset!r}; choose from: {known}") from exc


def _first_user_text(row: dict[str, Any]) -> str | None:
    """Extract the first human/user turn text from a ShareGPT conversation row.
    Args:
        row: One ShareGPT JSON object with a conversations list.
    Returns:
        The first user message text, or None if none is found.
    """
    for turn in row.get("conversations") or []:
        role = (turn.get("from") or turn.get("role") or "").lower()
        if role not in ("human", "user"):
            continue
        text = turn.get("value") or turn.get("content")
        if text:
            return text
        return None
    return None

def prepare() -> Path:
    """Download ShareGPT and write the first N_PROMPTS user turns as JSONL.
    Returns:
        Path: Path to the written sharegpt_500.jsonl file.
    Raises:
        RuntimeError: If fewer than N_PROMPTS prompts can be extracted.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=HF_REPO,
        filename=HF_FILE,
        repo_type="dataset",
        revision=HF_REVISION,
    )
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    DATA.parent.mkdir(parents=True, exist_ok=True)
    prompts: list[str] = []
    for row in rows:
        text = _first_user_text(row)
        if text is None:
            continue
        prompts.append(text)
        if len(prompts) >= N_PROMPTS:
            break
    if len(prompts) < N_PROMPTS:
        raise RuntimeError(f"only {len(prompts)} prompts (need {N_PROMPTS})")
    with DATA.open("w", encoding="utf-8") as out:
        for text in prompts:
            out.write(json.dumps({"prompt": text}, ensure_ascii=False) + "\n")
    logger.info(f"wrote {len(prompts)} prompts -> {DATA}")
    return DATA


def clear_capture_dir(directory: Path = CAPTURE_DIR) -> None:
    """Create capture dir and delete prior JSONL shards from Proxy tees.

    Args:
        directory: Directory where ray-serve Proxy replicas append *.jsonl.
    """
    directory.mkdir(parents=True, exist_ok=True)
    removed = 0
    for path in directory.glob("*.jsonl"):
        path.unlink(missing_ok=True)
        removed += 1
    logger.info(f"cleared capture dir {directory} ({removed} jsonl files)")


def assemble_captured_completions(
    out_path: Path,
    directory: Path = CAPTURE_DIR,
) -> Path:
    """Merge Proxy JSONL shards into one assembled_chat_completions.json.

    Args:
        out_path: Destination JSON path (typically results/<run-id>/...).
        directory: Capture directory with per-replica *.jsonl files.

    Returns:
        Path to the written assembled file (empty completions if nothing captured).
    """
    completions: list[dict[str, Any]] = []
    if directory.is_dir():
        for shard in sorted(directory.glob("*.jsonl")):
            for line_no, line in enumerate(
                shard.read_text(encoding="utf-8").splitlines(), start=1
            ):
                line = line.strip()
                if not line:
                    continue
                try:
                    completions.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"skip bad JSONL {shard.name}:{line_no}")
    payload = {
        "count": len(completions),
        "completions": completions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(f"wrote {out_path} (count={len(completions)})")
    return out_path


def run(args: BenchmarkArgs) -> Path:
    """Run GuideLLM against the Serve target and write guidellm.json.
    Args:
        args: Target URL, request rate, run id, session-header flag, dataset.
    Returns:
        Path: Path to the written results/<run-id>/guidellm.json file.
    Raises:
        FileNotFoundError: If the selected dataset JSONL is missing.
        subprocess.CalledProcessError: If the guidellm CLI exits non-zero.
    """
    data_path = resolve_data_path(args.dataset)
    if args.dataset == "sharegpt" and not data_path.is_file():
        prepare()
    if not data_path.is_file():
        hint = (
            "run: python scripts/prefix_prompt_gen.py"
            if args.dataset == "prefix_shared"
            else "run: python scripts/run_benchmark.py --prepare"
        )
        raise FileNotFoundError(f"missing {data_path}; {hint}")
    run_id = args.run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = RESULTS_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "guidellm.json"
    assembled_path = out_dir / ASSEMBLED_NAME
    clear_capture_dir()
    backend = f"kind=openai_http,target={args.target}"
    # Static session id pins all traffic to one ConsistentHash replica; omit
    # for fair comparison against P2C / RoundRobin (uniform fallback IDs).
    if args.use_session:
        backend += ",extras.headers.x-session-id=bench-session"
    # load_kwargs.split=train works around guidellm#1002 (DatasetDict vs Dataset).
    # Inside the guidellm container the repo is /app; use path relative to cwd.
    data_arg = data_path
    try:
        data_arg = data_path.relative_to(ROOT)
    except ValueError:
        pass
    cmd = [
        "guidellm", "run",
        "--backend", backend,
        "--data", f'kind=json_file,path={data_arg.as_posix()},load_kwargs={{"split": "train"}}',
        "--data-column-mapper",
        "kind=generative_column_mapper,column_mappings.text_column=prompt",
        "--data-loader",
        "kind=pytorch,samples=500,shuffle=false",
        "--profile",
        f"kind=constant,rate={args.rate}",
        "--constraint", "kind=max_requests,count=500",
        "--metrics", "kind=generative,sample_size=0",
        "--output", f"kind=json,path={output_path}",
    ]
    logger.info(f"dataset={args.dataset} path={data_arg}")
    logger.info(f"running: {' '.join(cmd)}")
    try:
        subprocess.check_call(cmd, cwd=ROOT)
    finally:
        # Always assemble whatever the Proxies captured (even on GuideLLM failure).
        assemble_captured_completions(assembled_path)
        clear_capture_dir()
    logger.info(f"wrote {output_path}")
    return output_path


def probe_completion(target: str) -> Path:
    """POST one non-streaming chat completion and save the JSON body.
    Args:
        target: OpenAI-compatible base URL (Serve proxy or a single sim).
    Returns:
        Path: Path to results/sample_chat_completion.json.
    Raises:
        URLError / HTTPError: If the target is unreachable or returns an error.
    """
    url = target.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 32,
        "stream": False,
    }
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    logger.info(f"probing {url}")
    try:
        with urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            status = getattr(resp, "status", None) or resp.getcode()
    except HTTPError as exc:
        err_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"chat completion failed HTTP {exc.code} at {url}: {err_body}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"chat completion unreachable at {url}: {exc.reason}"
        ) from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"non-JSON response from {url} (HTTP {status}): {raw[:500]}"
        ) from exc
    SAMPLE_COMPLETION.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_COMPLETION.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info(f"wrote {SAMPLE_COMPLETION}")
    return SAMPLE_COMPLETION

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments for prepare / run / probe modes.
    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).
    Returns:
        argparse.Namespace: Parsed flags.
    """
    parser = argparse.ArgumentParser(
        description="Prepare ShareGPT JSONL or run a GuideLLM benchmark."
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="only rebuild data/sharegpt_500.jsonl",
    )
    parser.add_argument(
        "--probe-completion",
        action="store_true",
        help="POST one chat completion; write results/sample_chat_completion.json",
    )
    parser.add_argument("--target", default="http://127.0.0.1:8000")
    # prefix_shared + per-token sims (docker-compose.yml): prefer --rate 3
    # (longer prompts). run_router_bench.ps1 defaults to rate 3.
    parser.add_argument("--rate", type=float, default=7.0)
    parser.add_argument(
        "--run-id",
        default=None,
        help="results/<run-id>/guidellm.json (use router name: prefix, p2c, …)",
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="sharegpt",
        help="JSONL under data/: sharegpt or prefix_shared (from prefix_prompt_gen.py)",
    )
    parser.add_argument(
        "--session",
        action="store_true",
        help="send a static x-session-id (ConsistentHash sticky demo only)",
    )
    return parser.parse_args(argv)

def main(argv: list[str] | None = None) -> None:
    """Entry point: prepare dataset, probe a sample completion, or run GuideLLM.
    Args:
        argv: Optional CLI argv override for tests.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ns = parse_args(argv)
    if ns.prepare:
        prepare()
        return
    if ns.probe_completion:
        probe_completion(ns.target)
        return
    run(
        BenchmarkArgs(
            target=ns.target,
            rate=ns.rate,
            run_id=ns.run_id,
            use_session=ns.session,
            dataset=ns.dataset,
        )
    )

if __name__ == "__main__":
    main()
