"""Grounds the corpus (cached in cache/items.json) and scores the method, its
ungraded and concepts-only variants, BM25 and the random floor by nDCG@10.
Writes cache/results.json."""

import argparse
import hashlib
import json
from dataclasses import astuple
from pathlib import Path

from evaluation import blind, corpus, count, evaluate, importance, qrels, qualified
from evaluation.baselines import bm25
from ranking_method import link, policy, rank, resource

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache" / "items.json"
DEVICE = "mps"
BATCHES = ("oneitem", "twoitem", "threeitem")
GOLDS = {"qualified": qualified, "count": count, "weighted": importance}
SYSTEMS = ("random floor", "bm25", "concepts only", "concepts + ordinal, flat",
           "concepts + ordinal")


def _meta(texts, device, vocab):
    digest = hashlib.sha256()
    for text in sorted(texts):
        digest.update(rank.text_key(text).encode())
    return {
        "format": 2,
        "release": resource.release(),
        "model": link.MODEL,
        "device": device,
        "semantic_threshold": link.SEMANTIC_THRESHOLD,
        "lexical_threshold": link.LEXICAL_SUPPORT_THRESHOLD,
        "vocabulary": len(vocab),
        "ordinal": len(policy.ordinal_elements()),
        "corpus": digest.hexdigest(),
    }


def _dump(items):
    return [[i.element_id, i.category, [astuple(s) for s in i.support]] for i in items]


def _load(rows):
    return tuple(
        link.GroundedItem(e, c, tuple(link.Support(*s) for s in support))
        for e, c, support in rows
    )


def grounding(texts, device=DEVICE, cache=CACHE, vocab=None):
    """{text_key: grounded items} for every text, cached across runs."""
    vocab = vocab if vocab is not None else link.vocabulary()
    meta = _meta(texts, device, vocab)
    if cache is not None and cache.exists():
        stored = json.loads(cache.read_text())
        if stored.get("meta") == meta:
            return {k: _load(v) for k, v in stored["items"].items()}
    grounded = rank.ground(texts, vocab=vocab, enc=link.encoder(device=device))
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(
            {"meta": meta, "items": {k: _dump(v) for k, v in grounded.items()}}))
    return grounded


def report(device=DEVICE, vocab=None, cache=CACHE):
    records = corpus.load_corpus()
    blinded = blind.blind(records)
    grounded = grounding([r.profile_text for r in records], device, cache, vocab)

    by_batch = {b: [r for r in records if r.batch_name == b] for b in BATCHES}
    runs = {}
    for batch, rs in by_batch.items():
        view = blind.select(blinded, rs)
        runs[batch] = {
            "concepts + ordinal": rank.rank(view.queries, view.documents, grounded=grounded,
                                          key=rank.LEXICOGRAPHIC),
            # the method without requirement grades
            "concepts + ordinal, flat": rank.rank(
                view.queries, view.documents, grounded=grounded,
                ladder=policy.CONCEPTS_ORDINAL_FLAT, key=rank.LEXICOGRAPHIC),
            "concepts only": rank.rank(view.queries, view.documents, grounded=grounded,
                                       ladder=policy.CONCEPTS, key=rank.LEXICOGRAPHIC),
            "bm25": bm25.rank(view.queries, view.documents),
        }

    out = {}
    for name, gold in GOLDS.items():
        judged = {b: qrels.blind_qrels(qrels.build_qrels(rs, gold.scorer), blinded)
                  for b, rs in by_batch.items()}
        rows = {
            system: [evaluate.mean(evaluate.evaluate(runs[b][system], judged[b]))["ndcg_cut_10"]
                     for b in BATCHES]
            for system in SYSTEMS if system != "random floor"
        }
        floor = [evaluate.random_floor(judged[b])["ndcg_cut_10"] for b in BATCHES]
        rows["random floor"] = [f["mean"] for f in floor]
        rows["random floor sd"] = [f["sd"] for f in floor]
        out[name] = rows
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    results = report(device=args.device)
    order = SYSTEMS + ("random floor sd",)
    width = max(len(s) for s in order)
    print(f"O*NET release: {resource.release()}   model: {link.MODEL}")
    for gold, rows in results.items():
        print(f"\n{gold}   ({' / '.join(BATCHES)})")
        for system in order:
            print(f"  {system:<{width}}  " + " / ".join(f"{v:.4f}" for v in rows[system]))
    (ROOT / "cache" / "results.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
