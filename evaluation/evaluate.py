"""nDCG@10 via pytrec_eval, and a random-ranking floor.

pytrec_eval breaks score ties by document id, descending; blinded ids make that deterministic.
"""

import random
import statistics

import pytrec_eval

Run = dict[str, dict[str, float]]  # query id -> doc id -> score
Qrels = dict[str, dict[str, int]]  # query id -> doc id -> relevance

K = 10
RANDOM_SEEDS = 100


def evaluate(run: Run, qrels: Qrels, k: int = K) -> dict[str, dict[str, float]]:
    """Per-query scores, shaped like pytrec_eval's own output."""
    measure = f"ndcg_cut_{k}"
    scores = pytrec_eval.RelevanceEvaluator(qrels, {measure}).evaluate(run)
    return {
        query_id: {measure: scores.get(query_id, {}).get(measure, 0.0)}
        for query_id in qrels
    }


def mean(results: dict[str, dict[str, float]]) -> dict[str, float]:
    measures = next(iter(results.values()))
    return {m: sum(r[m] for r in results.values()) / len(results) for m in measures}


def random_run(qrels: Qrels, seed: int = 0) -> Run:
    """Random scores for every judged pair; relevance values are not read."""
    rng = random.Random(seed)
    return {q: {d: rng.random() for d in judgments} for q, judgments in qrels.items()}


def random_floor(
    qrels: Qrels, seeds: int = RANDOM_SEEDS, k: int = K
) -> dict[str, dict[str, float]]:
    """Mean and SD of each measure over ``seeds`` random runs."""
    runs = [mean(evaluate(random_run(qrels, seed), qrels, k)) for seed in range(seeds)]
    return {
        m: {
            "mean": statistics.mean(r[m] for r in runs),
            "sd": statistics.stdev(r[m] for r in runs),
        }
        for m in runs[0]
    }
