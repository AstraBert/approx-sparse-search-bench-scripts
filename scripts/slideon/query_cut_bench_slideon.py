#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "httpx==0.28.1",
#   "huggingface-hub==2.4.0",
#   "numpy==2.5.4",
#   "pyarrow==26.0.0",
#   "scipy==1.18.1",
# ]
# ///
import argparse
import asyncio
import math
import random
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import httpx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from scipy import sparse

REPO_ID = "lightonai/SlideOn-sparse-vectors"
CONFIG = "beir-v1.0.0-nq"
QUERIES_FILE = f"{CONFIG}/queries/test/part-00000.parquet"
VECTOR_NAME = "slideon"
DEFAULT_MAX_DOCS = 1_000_000
VOCAB_SIZE = 65_536  # indices are uint16, so no dimension can be above this
TOP = 10
INTERLEAVE_SEED = 0
READ_BATCH_SIZE = 20_000  # rows converted to a numpy matrix at once
GROUND_TRUTH_CHUNK = 25  # queries scored against the whole corpus at once
UPLOAD_CONCURRENCY = 4

SparseJson = dict[str, list[Any]]


@dataclass(frozen=True)
class Corpus:
    doc_ids: list[str]
    docs: sparse.csr_matrix
    queries: sparse.csr_matrix


class Hits(NamedTuple):
    ids: list[str]
    server_ms: float


class Config(NamedTuple):
    query_cut: (
        float | None
    )  # None: the baseline, full query. Else the weight fraction kept
    oversampling: int | None  # None: no rescoring, the cut results are final

    @property
    def label(self) -> str:
        if self.query_cut is None:
            return "baseline"
        if self.oversampling is None:
            return f"query_cut={self.query_cut:g}"
        return f"cut={self.query_cut:g} os={self.oversampling}"


@dataclass(frozen=True)
class RunResult:
    label: str
    client_ms: np.ndarray
    server_ms: np.ndarray
    recall: float
    mrr: float
    ndcg: float


def corpus_files() -> list[str]:
    """Repo paths of the corpus parquet parts, in order."""
    files = HfApi().list_repo_files(REPO_ID, repo_type="dataset")
    prefix = f"{CONFIG}/corpus/"
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))


def download(repo_path: str) -> Path:
    print(f"Fetching {repo_path}")
    return Path(hf_hub_download(REPO_ID, repo_path, repo_type="dataset"))


def batch_to_csr(batch: pa.RecordBatch) -> sparse.csr_matrix:
    """Rebuild sparse vectors from the `indices` and `weights` list columns."""
    indices_col = batch.column("indices")
    weights_col = batch.column("weights")
    # Arrow stores a list column as one flat array of all the values plus
    # `offsets`: row r owns the values from offsets[r] to offsets[r + 1].
    # That is exactly the CSR `indptr`, once shifted to start at 0
    # (a batch can be a slice of a bigger array, so offsets[0] may not be 0).
    offsets = indices_col.offsets.to_numpy()
    indptr = offsets - offsets[0]
    # flatten() joins all rows into one array. It respects the slicing, so it
    # holds only this batch's values. scipy wants int32 indices and float32
    # data here, while the file has uint16 and float16.
    indices = indices_col.flatten().to_numpy().astype(np.int32)
    data = weights_col.flatten().to_numpy(zero_copy_only=False).astype(np.float32)
    assert len(indices) == len(data) == indptr[-1], "indices and weights do not match"
    matrix = sparse.csr_matrix((data, indices, indptr), shape=(len(batch), VOCAB_SIZE))
    # The file lists each vector's dimensions by weight, not by index.
    # sum_duplicates() sorts the indices of every row and merges repeats.
    matrix.sum_duplicates()
    return matrix


def read_parquet(path: Path) -> Iterator[tuple[list[str], sparse.csr_matrix]]:
    """Yield (ids, CSR matrix) for consecutive slices of one parquet file."""
    # The corpus parts hold very large row groups, so reading one needs a few GB of RAM.
    for batch in pq.ParquetFile(path).iter_batches(batch_size=READ_BATCH_SIZE):
        yield batch.column("id").to_pylist(), batch_to_csr(batch)


def load_rows(paths: Iterator[Path], limit: int) -> tuple[list[str], sparse.csr_matrix]:
    """Read up to `limit` vectors. Return their ids and a CSR matrix."""
    ids: list[str] = []
    chunks: list[sparse.csr_matrix] = []
    for path in paths:
        for batch_ids, matrix in read_parquet(path):
            # a CSR matrix can be cut by rows like a list: keep only what is still needed
            keep = min(len(batch_ids), limit - len(ids))
            ids.extend(batch_ids[:keep])
            chunks.append(matrix[:keep])
            print(f"\r  {len(ids)} vectors", end="", flush=True)
            if len(ids) == limit:
                break
        if len(ids) == limit:
            break
    print()
    assert len(ids) == limit, f"Only {len(ids)} vectors found, wanted {limit}"
    # vstack glues the chunks on top of each other into one matrix
    matrix = sparse.vstack(chunks, format="csr")
    matrix.eliminate_zeros()  # drop explicit zero weights, if any
    return ids, matrix


def load_corpus(
    max_docs: int | None, num_queries: int, local_dir: Path | None
) -> Corpus:
    if local_dir is None:
        # the corpus parts are downloaded lazily: only as many as `max_docs` needs
        parts = (download(path) for path in corpus_files())
        query_files = iter([download(QUERIES_FILE)])
        doc_limit = max_docs or DEFAULT_MAX_DOCS
    else:
        # files written by tools/encode_nq_sparse.py, in the same layout
        doc_files = sorted(local_dir.glob("docs-*.parquet"))
        assert doc_files, f"No docs-*.parquet in {local_dir}"
        parts = iter(doc_files)
        query_files = iter([local_dir / "queries.parquet"])
        available = sum(pq.ParquetFile(path).metadata.num_rows for path in doc_files)
        doc_limit = max_docs or available
    doc_ids, docs = load_rows(parts, doc_limit)
    _, queries = load_rows(query_files, num_queries)
    return Corpus(doc_ids=doc_ids, docs=docs, queries=queries)


def top_k_per_row(scores: np.ndarray) -> np.ndarray:
    """For each row of a (queries x docs) score matrix, return the column ids of
    the TOP highest scores, best first."""
    # argpartition is a cheap partial sort: the first TOP entries of its result
    # are the TOP largest scores, in no particular order. Scores are negated
    # because argpartition puts the smallest values first.
    candidates = np.argpartition(-scores, TOP, axis=1)[:, :TOP]
    # take_along_axis picks, in every row, the scores at that row's candidates
    candidate_scores = np.take_along_axis(scores, candidates, axis=1)
    # argsort fully sorts only these TOP candidates per row (negated: best first)
    order = np.argsort(-candidate_scores, axis=1)
    # reorder the candidate doc ids with that ranking
    return np.take_along_axis(candidates, order, axis=1)


def ground_truth(corpus: Corpus) -> list[list[str]]:
    """Exact top TOP doc ids per query, by brute-force dot product."""
    # .T flips the (docs x dims) matrix to (dims x docs). tocsr() makes the flip
    # physical, so the products below read contiguous memory.
    docs_t = corpus.docs.T.tocsr()
    expected: list[list[str]] = []
    for start in range(0, corpus.queries.shape[0], GROUND_TRUTH_CHUNK):
        # slicing a CSR matrix by rows gives a smaller CSR matrix
        chunk = corpus.queries[start : start + GROUND_TRUTH_CHUNK]
        # `@` is a sparse matrix product: (chunk x dims) @ (dims x docs) gives
        # every query's dot product with every doc. toarray() makes it a dense
        # numpy array, which is small enough with only a few queries at a time.
        scores = (chunk @ docs_t).toarray()
        top = top_k_per_row(scores)
        expected.extend([corpus.doc_ids[doc] for doc in row] for row in top)
        print(f"\r  {len(expected)}/{corpus.queries.shape[0]}", end="", flush=True)
    print()
    return expected


def row_to_json(matrix: sparse.csr_matrix, row: int) -> SparseJson:
    # indptr[row]:indptr[row + 1] is this row's slice of the flat CSR arrays
    start, end = matrix.indptr[row], matrix.indptr[row + 1]
    return {
        "indices": matrix.indices[start:end].tolist(),
        "values": matrix.data[start:end].tolist(),
    }


def score_query(returned: list[str], expected: list[str]) -> tuple[float, float, float]:
    """Recall, MRR and nDCG at TOP, with the ground truth docs as the relevant set."""
    relevant = set(expected)
    hit_ranks = [rank for rank, doc in enumerate(returned) if doc in relevant]
    recall = len(hit_ranks) / TOP
    mrr = 1 / (hit_ranks[0] + 1) if hit_ranks else 0.0
    dcg = sum(1 / math.log2(rank + 2) for rank in hit_ranks)
    ideal_dcg = sum(1 / math.log2(rank + 2) for rank in range(TOP))
    return recall, mrr, dcg / ideal_dcg


async def recreate_collection(client: httpx.AsyncClient, name: str) -> None:
    (await client.delete(f"/collections/{name}")).raise_for_status()
    body = {"vectors": {}, "sparse_vectors": {VECTOR_NAME: {}}}
    (await client.put(f"/collections/{name}", json=body)).raise_for_status()


async def upload(
    client: httpx.AsyncClient, name: str, corpus: Corpus, batch_size: int
) -> None:
    semaphore = asyncio.Semaphore(UPLOAD_CONCURRENCY)
    total = len(corpus.doc_ids)
    done = 0

    async def send(start: int) -> None:
        nonlocal done
        async with semaphore:
            end = min(start + batch_size, total)
            points = [
                {
                    "id": row,
                    "vector": {VECTOR_NAME: row_to_json(corpus.docs, row)},
                    "payload": {"_id": corpus.doc_ids[row]},
                }
                for row in range(start, end)
            ]
            response = await client.put(
                f"/collections/{name}/points",
                params={"wait": "true"},
                json={"points": points},
            )
            response.raise_for_status()
            done += end - start
            print(f"\rUploaded {done}/{total}", end="", flush=True)

    await asyncio.gather(*(send(start) for start in range(0, total, batch_size)))
    print()


async def collection_info(client: httpx.AsyncClient, name: str) -> dict[str, Any]:
    response = await client.get(f"/collections/{name}")
    response.raise_for_status()
    return response.json()["result"]


async def wait_until_green(client: httpx.AsyncClient, name: str) -> None:
    consecutive = 0
    while consecutive < 3:
        info = await collection_info(client, name)
        consecutive = consecutive + 1 if info["status"] == "green" else 0
        await asyncio.sleep(2)


def query_body(vector: SparseJson, config: Config) -> dict[str, Any]:
    body: dict[str, Any] = {
        "query": vector,
        "using": VECTOR_NAME,
        "limit": TOP,
        "with_payload": ["_id"],
    }
    if config.query_cut is None:
        return body
    cut_params = {"keep_query_weight": config.query_cut}
    if config.oversampling is None:
        body["params"] = cut_params
        return body
    # First stage: cheap search with the cut query, `oversampling` times more candidates.
    # The outer query has no `query_cut`, so it rescores only those candidates in full.
    body["prefetch"] = {
        "query": vector,
        "using": VECTOR_NAME,
        "limit": TOP * config.oversampling,
        "params": cut_params,
    }
    return body


async def search(
    client: httpx.AsyncClient, name: str, vector: SparseJson, config: Config
) -> Hits:
    response = await client.post(
        f"/collections/{name}/points/query", json=query_body(vector, config)
    )
    response.raise_for_status()
    body = response.json()
    ids = [point["payload"]["_id"] for point in body["result"]["points"]]
    return Hits(ids, body["time"] * 1000)


async def assert_query_cut_supported(
    client: httpx.AsyncClient, name: str, vector: SparseJson
) -> None:
    """A server that knows `keep_query_weight` rejects 0.0. An older one ignores the field."""
    response = await client.post(
        f"/collections/{name}/points/query",
        json=query_body(vector, Config(query_cut=0.0, oversampling=None)),
    )
    if response.status_code not in (400, 422):
        raise SystemExit(
            f"Server accepted keep_query_weight=0.0 (HTTP {response.status_code}): "
            "it was not built with keep_query_weight, results would be meaningless."
        )


async def run_interleaved(
    client: httpx.AsyncClient,
    name: str,
    vectors: list[SparseJson],
    expected: list[list[str]],
    configs: list[Config],
    warmup: int,
) -> list[RunResult]:
    """Run every config on every query, in a random order per query.

    Slow drift on the machine (heat, background load) then hits all configs alike.
    """
    rng = random.Random(INTERLEAVE_SEED)
    print(f"Running {len(configs)} configs, interleaved")

    def shuffled() -> list[int]:
        return rng.sample(range(len(configs)), len(configs))

    for vector in vectors[:warmup]:
        for index in shuffled():
            await search(client, name, vector, configs[index])

    client_ms: list[list[float]] = [[] for _ in configs]
    server_ms: list[list[float]] = [[] for _ in configs]
    scores: list[list[tuple[float, float, float]]] = [[] for _ in configs]
    for number, (vector, truth) in enumerate(zip(vectors, expected, strict=True)):
        for index in shuffled():
            start = time.perf_counter()
            hits = await search(client, name, vector, configs[index])
            client_ms[index].append((time.perf_counter() - start) * 1000)
            server_ms[index].append(hits.server_ms)
            scores[index].append(score_query(hits.ids, truth))
        if (number + 1) % 100 == 0:
            print(f"  {number + 1}/{len(vectors)} queries")

    results = []
    for index, config in enumerate(configs):
        # scores is (queries x 3): axis=0 averages down the rows, one mean per metric
        recall, mrr, ndcg = np.mean(scores[index], axis=0)
        results.append(
            RunResult(
                config.label,
                np.array(client_ms[index]),
                np.array(server_ms[index]),
                recall,
                mrr,
                ndcg,
            )
        )
    return results


def print_corpus_stats(corpus: Corpus) -> None:
    # indptr holds row offsets, so the difference of neighbours is the number of
    # non-zero dimensions in each row
    doc_nnz = np.diff(corpus.docs.indptr)
    query_nnz = np.diff(corpus.queries.indptr)
    print(f"Docs: {len(doc_nnz)}, non-zero dims per doc: mean {doc_nnz.mean():.1f}")
    print(
        f"Queries: {len(query_nnz)}, non-zero dims per query: "
        f"min {query_nnz.min()}, median {np.median(query_nnz):.0f}, max {query_nnz.max()}"
    )


def print_latency_table(title: str, series: list[tuple[str, np.ndarray]]) -> None:
    # percentile(x, q) is the value below which q% of the samples fall
    base_p50 = np.percentile(series[0][1], 50)
    header = (
        f"{'config':<16}{'min ms':>9}{'p50 ms':>9}{'p95 ms':>9}{'max ms':>9}"
        f"{'p50 x':>8}{'paired x':>10}"
    )
    print(f"\n{title}\n")
    print(header)
    print("-" * len(header))
    for label, latencies in series:
        p50, p95 = np.percentile(latencies, [50, 95])  # both in one pass
        # every config saw the same query at about the same time, so compare per query
        paired = np.median(series[0][1] / latencies)
        print(
            f"{label:<16}{latencies.min():>9.2f}{p50:>9.2f}{p95:>9.2f}"
            f"{latencies.max():>9.2f}{base_p50 / p50:>8.2f}{paired:>10.2f}"
        )


def print_quality_table(results: list[RunResult]) -> None:
    header = f"{'config':<16}{'recall':>9}{'mrr':>9}{'ndcg':>9}"
    print(f"\nQuality at {TOP}, ground truth is the exact top {TOP}\n")
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.label:<16}{result.recall:>9.4f}{result.mrr:>9.4f}{result.ndcg:>9.4f}"
        )


def print_report(results: list[RunResult]) -> None:
    print_latency_table(
        "Latency, server side (`time` in the response)",
        [(result.label, result.server_ms) for result in results],
    )
    print_latency_table(
        "Latency, client side (includes HTTP)",
        [(result.label, result.client_ms) for result in results],
    )
    print_quality_table(results)


async def run(
    args: argparse.Namespace, corpus: Corpus, expected: list[list[str]]
) -> None:
    headers = {"api-key": args.api_key} if args.api_key else {}
    timeout = httpx.Timeout(300)
    async with httpx.AsyncClient(
        base_url=args.url, headers=headers, timeout=timeout
    ) as client:
        if not args.skip_upload:
            await recreate_collection(client, args.collection)
            await upload(client, args.collection, corpus, args.batch_size)
            await wait_until_green(client, args.collection)
        info = await collection_info(client, args.collection)
        print(
            f"Collection: {info['points_count']} points, {info['segments_count']} segments"
        )

        vectors = [
            row_to_json(corpus.queries, row) for row in range(corpus.queries.shape[0])
        ]
        await assert_query_cut_supported(client, args.collection, vectors[0])
        results = await run_interleaved(
            client,
            args.collection,
            vectors,
            expected,
            configs(args.query_cut, args.oversampling),
            args.warmup,
        )
    print_report(results)


def configs(query_cuts: list[float], oversamplings: list[int]) -> list[Config]:
    """The baseline, then every cut alone, then every cut with each oversampling."""
    plain = [Config(cut, None) for cut in query_cuts]
    rescored = [Config(cut, os) for cut in query_cuts for os in oversamplings]
    return [Config(None, None), *plain, *rescored]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare sparse search with and without query_cut on SlideOn NQ."
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=None,
        help="read vectors written by encode_nq_sparse.py instead of SlideOn from the hub",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help=f"default: {DEFAULT_MAX_DOCS} from the hub, every document in --local-dir",
    )
    parser.add_argument("--num-queries", type=int, default=1_000)
    parser.add_argument("--url", default="http://localhost:6333")
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--collection",
        default=None,
        help="default: query_cut_bench_slideon, or query_cut_bench_<local dir name>",
    )
    parser.add_argument(
        "--query-cut",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9],
        help="fractions of the query weight to keep, each above 0 and at most 1",
    )
    parser.add_argument(
        "--oversampling",
        type=int,
        nargs="*",
        default=[],
        help="Also run each cut as a first stage that fetches this many times more "
        f"than the top {TOP}, then rescore those with the full query. e.g. 2 5 10",
    )
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument(
        "--skip-upload", action="store_true", help="Reuse the existing collection"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.collection is None:
        suffix = args.local_dir.name if args.local_dir else "slideon"
        args.collection = f"query_cut_bench_{suffix}"
    corpus = load_corpus(args.max_docs, args.num_queries, args.local_dir)
    print_corpus_stats(corpus)
    print("Computing ground truth")
    expected = ground_truth(corpus)
    asyncio.run(run(args, corpus, expected))


if __name__ == "__main__":
    main()
