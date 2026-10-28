"""Generate a shared-prefix JSONL for GuideLLM (prefix-router benchmarks).

Writes data/prefix_shared_500.jsonl by default: long synthetic topic prefixes
plus unique short suffixes, shuffled with a fixed seed. No HuggingFace / ShareGPT.

    python scripts/prefix_prompt_gen.py
    python scripts/prefix_prompt_gen.py --n 500 --prefixes 8 --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "prefix_shared_500.jsonl"

# Repeated filler so each topic prefix spans many KV blocks (block-size 16).
_FILLER = (
    "This is a shared system context block used only for prefix-cache routing "
    "benchmarks. It describes policies, constraints, and background facts that "
    "every request under the same topic reuses. Keep this text identical across "
    "prompts that share a topic so the simulator can hit cached token blocks. "
)

_TOPIC_NAMES = (
    "networking",
    "databases",
    "compilers",
    "security",
    "distributed_systems",
    "operating_systems",
    "machine_learning",
    "storage",
)


def build_prefix(topic_index: int, topic_name: str, repeats: int) -> str:
    """Build one long shared prefix for a topic.

    Args:
        topic_index: Numeric topic id (0-based).
        topic_name: Short topic label embedded in the text.
        repeats: How many times to repeat the filler paragraph.

    Returns:
        Prefix string ending with a topic marker line.
    """
    body = _FILLER * repeats
    return (
        f"TOPIC_{topic_index} ({topic_name}) SHARED PREFIX START\n"
        f"{body}"
        f"TOPIC_{topic_index} SHARED PREFIX END\n"
    )


def build_suffix(topic_name: str, question_index: int) -> str:
    """Build a short unique user question for a topic.

    Args:
        topic_name: Topic label.
        question_index: Per-topic question counter.

    Returns:
        Suffix string unique for (topic_name, question_index).
    """
    return (
        f"User question {question_index} about {topic_name}: "
        f"briefly explain item {question_index}."
    )


def generate_prompts(
    n: int,
    num_prefixes: int,
    prefix_repeats: int,
    seed: int,
) -> list[str]:
    """Generate n shuffled prompts with shared topic prefixes.

    Args:
        n: Total number of prompts.
        num_prefixes: Number of distinct topic prefixes.
        prefix_repeats: Filler repetitions per prefix (length control).
        seed: RNG seed for shuffling.

    Returns:
        List of full prompt strings (prefix + suffix).
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if num_prefixes < 1:
        raise ValueError("prefixes must be >= 1")
    if num_prefixes > len(_TOPIC_NAMES):
        raise ValueError(
            f"prefixes must be <= {len(_TOPIC_NAMES)} (named topics available)"
        )

    prefixes = [
        build_prefix(i, _TOPIC_NAMES[i], prefix_repeats)
        for i in range(num_prefixes)
    ]
    per_prefix = math.ceil(n / num_prefixes)
    prompts: list[str] = []
    for topic_i, prefix in enumerate(prefixes):
        topic_name = _TOPIC_NAMES[topic_i]
        for q in range(per_prefix):
            if len(prompts) >= n:
                break
            prompts.append(prefix + build_suffix(topic_name, q))
        if len(prompts) >= n:
            break

    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts


def write_jsonl(path: Path, prompts: list[str]) -> None:
    """Write GuideLLM-compatible JSONL ({"prompt": ...} per line).

    Args:
        path: Output file path.
        prompts: Prompt strings to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for text in prompts:
            out.write(json.dumps({"prompt": text}, ensure_ascii=False) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI flags for the prefix prompt generator.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed namespace.
    """
    parser = argparse.ArgumentParser(
        description="Generate data/prefix_shared_*.jsonl for prefix-router benches."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"output JSONL path (default: {DEFAULT_OUT})",
    )
    parser.add_argument("--n", type=int, default=500, help="number of prompts")
    parser.add_argument(
        "--prefixes",
        type=int,
        default=8,
        help="number of distinct shared prefixes / topics",
    )
    parser.add_argument(
        "--prefix-repeats",
        type=int,
        default=12,
        help="filler paragraph repeats per prefix (controls shared token length)",
    )
    parser.add_argument("--seed", type=int, default=42, help="shuffle seed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Generate and write the shared-prefix JSONL."""
    ns = parse_args(argv)
    prompts = generate_prompts(
        n=ns.n,
        num_prefixes=ns.prefixes,
        prefix_repeats=ns.prefix_repeats,
        seed=ns.seed,
    )
    write_jsonl(ns.out, prompts)
    sample_len = len(prompts[0]) if prompts else 0
    print(
        f"wrote {len(prompts)} prompts -> {ns.out} "
        f"(prefixes={ns.prefixes}, seed={ns.seed}, sample_chars={sample_len})"
    )


if __name__ == "__main__":
    main()
