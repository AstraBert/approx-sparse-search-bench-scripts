#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "transformers>=4.50,<6",
#   "huggingface-hub",
#   "pyarrow==26.0.0",
#   "numpy>=2,<3",
# ]
#
# [tool.uv]
# exclude-newer = "2026-11-02T00:00:00Z"
# ///
import argparse
import json
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, PreTrainedTokenizerBase

BEIR_REPO = "BeIR/nq"
BEIR_QUERIES = "queries/queries-00000-of-00001.parquet"


class Query(NamedTuple):
    text: str
    dims: np.ndarray  # dimension ids, sorted by weight, largest first
    weights: np.ndarray
    original: np.ndarray  # True where the dimension is a word of the query text


class CutStats(NamedTuple):
    cut_dims: int
    cut_original: int
    kept_dims: int
    kept_original: int
    cut_weight: float
    total_weight: float
    original_total: int
    original_cut: int


def query_texts(ids: list[str]) -> list[str]:
    path = hf_hub_download(BEIR_REPO, BEIR_QUERIES, repo_type="dataset")
    table = pq.read_table(path, columns=["_id", "text"]).to_pydict()
    text_of = dict(zip(table["_id"], table["text"], strict=True))
    return [text_of[query_id] for query_id in ids]


def normalize(token: str) -> str:
    """A token without its word-start marker and case, so `ĠFire` equals `fire`."""
    return token.lstrip("Ġ▁").removeprefix("##").lower()


def query_terms(text: str, tokenizer: PreTrainedTokenizerBase) -> set[str]:
    """Whole words of the text plus the pieces the tokenizer splits them into."""
    pieces = tokenizer.convert_ids_to_tokens(
        tokenizer(text, add_special_tokens=False)["input_ids"]
    )
    return set(re.findall(r"\w+", text.lower())) | {normalize(p) for p in pieces}


def read_vectors(path: Path) -> tuple[list[str], list[list[int]], list[list[float]]]:
    """Ids, dimensions and weights from the encoder parquet or the Seismic json."""
    if path.suffix == ".json":
        vectors = json.loads(path.read_text())["vectors"]
        return (
            [v["id"] for v in vectors],
            [v["coordinates"] for v in vectors],
            [v["values"] for v in vectors],
        )
    table = pq.read_table(path).to_pydict()
    return table["id"], table["indices"], table["weights"]


def load_queries(
    path: Path, tokenizer: PreTrainedTokenizerBase, limit: int
) -> list[Query]:
    ids, all_dims, all_weights = read_vectors(path)
    texts = query_texts(ids[:limit])
    zipped = zip(texts, all_dims, all_weights, strict=False)
    queries = []
    for text, dims, weights in zipped:
        terms = query_terms(text, tokenizer)
        order = np.argsort(-np.asarray(weights, dtype=np.float32), kind="stable")
        dims = np.asarray(dims)[order]
        tokens = tokenizer.convert_ids_to_tokens(dims.tolist())
        queries.append(
            Query(
                text,
                dims,
                np.asarray(weights, dtype=np.float32)[order],
                np.array([normalize(t) in terms for t in tokens]),
            )
        )
    return queries


def cut_stats(queries: list[Query], cut: int) -> CutStats:
    totals = np.zeros(8)
    for q in queries:
        kept, dropped = q.original[:cut], q.original[cut:]
        totals += [
            dropped.size,
            dropped.sum(),
            kept.size,
            kept.sum(),
            q.weights[cut:].sum(),
            q.weights.sum(),
            q.original.sum(),
            dropped.sum(),
        ]
    return CutStats(*map(float, totals))  # ty: ignore[missing-argument]


def percent(part: float, whole: float) -> str:
    return f"{100 * part / whole:5.1f}%" if whole else "    - "


def print_overview(queries: list[Query]) -> None:
    sizes = np.array([q.dims.size for q in queries])
    original = np.array([q.original.sum() for q in queries])
    weights = np.concatenate([q.weights for q in queries])
    flags = np.concatenate([q.original for q in queries])
    print(
        f"Queries: {len(queries)}, non-zero dims per query: median {np.median(sizes):.0f}"
    )
    print(f"Original terms per query: median {np.median(original):.0f}")
    print(f"Share of dims that are original terms: {percent(flags.sum(), flags.size)}")
    print(
        f"Mean weight, original {weights[flags].mean():.3f}, "
        f"expansion {weights[~flags].mean():.3f}"
    )


def print_cuts(queries: list[Query], cuts: list[int]) -> None:
    print(
        f"\n{'cut':>5} {'cut dims':>9} {'of them original':>17} "
        f"{'kept: original':>15} {'weight cut':>11} {'originals lost':>15}"
    )
    for cut in cuts:
        s = cut_stats(queries, cut)
        print(
            f"{cut:>5} {int(s.cut_dims):>9} {percent(s.cut_original, s.cut_dims):>17} "
            f"{percent(s.kept_original, s.kept_dims):>15} "
            f"{percent(s.cut_weight, s.total_weight):>11} "
            f"{percent(s.original_cut, s.original_total):>15}"
        )


def print_examples(
    queries: list[Query], tokenizer: PreTrainedTokenizerBase, cut: int, count: int
) -> None:
    print(f"\nExamples at cut {cut}. * marks a term from the query text.")
    for q in queries[:count]:
        tokens = tokenizer.convert_ids_to_tokens(q.dims.tolist())
        labeled = [
            f"{'*' if q.original[i] else ''}{token}:{q.weights[i]:.2f}"
            for i, token in enumerate(tokens)
        ]
        print(f"\n{q.text}")
        print(f"  kept: {' '.join(labeled[:cut])}")
        print(f"  cut:  {' '.join(labeled[cut:])}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show which query terms query_cut removes: original or expansion."
    )
    parser.add_argument("model", help="the model that encoded the queries")
    parser.add_argument(
        "queries",
        type=Path,
        help="queries.parquet from the encoder, or the Seismic queries json",
    )
    parser.add_argument("--cuts", type=int, nargs="+", default=[10, 20, 40])
    parser.add_argument("--num-queries", type=int, default=1_000)
    parser.add_argument("--examples", type=int, default=3)
    parser.add_argument("--example-cut", type=int, default=20)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    queries = load_queries(args.queries, tokenizer, args.num_queries)
    print_overview(queries)
    print_cuts(queries, args.cuts)
    print_examples(queries, tokenizer, args.example_cut, args.examples)


if __name__ == "__main__":
    main()
