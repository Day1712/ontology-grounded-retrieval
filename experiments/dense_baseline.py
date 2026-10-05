"""Qwen3 dense baseline scored by nDCG@10 on the three golds (cross-check only).
Writes cache/dense_results.json."""

import argparse
import json
from pathlib import Path

import torch

from evaluation import blind, corpus, count, evaluate, importance, qrels, qualified
from evaluation.baselines import dense

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "cache" / "dense_results.json"
DEVICE = "mps"
DTYPE = torch.float32
BATCHES = ("oneitem", "twoitem", "threeitem")
GOLDS = {"qualified": qualified, "count": count, "weighted": importance}


def report(device=DEVICE, dtype=DTYPE):
    records = corpus.load_corpus()
    blinded = blind.blind(records)
    by_batch = {b: [r for r in records if r.batch_name == b] for b in BATCHES}

    loaded = dense.load(dtype=dtype, device=device)
    runs = {}
    for batch, rs in by_batch.items():
        view = blind.select(blinded, rs)
        runs[batch] = dense.rank(view.queries, view.documents, loaded=loaded)

    out = {}
    for name, gold in GOLDS.items():
        judged = {b: qrels.blind_qrels(qrels.build_qrels(rs, gold.scorer), blinded)
                  for b, rs in by_batch.items()}
        out[name] = {"dense": [
            evaluate.mean(evaluate.evaluate(runs[b], judged[b]))["ndcg_cut_10"]
            for b in BATCHES
        ]}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=DEVICE)
    args = parser.parse_args()

    results = report(device=args.device)
    existing = json.loads((ROOT / "cache" / "results.json").read_text())
    print(f"model: {dense.MODEL}   revision: {dense.REVISION[:12]}   dtype: {DTYPE}")
    for gold, rows in results.items():
        rows = {**existing[gold], **rows}
        width = max(len(s) for s in rows)
        print(f"\n{gold}   ({' / '.join(BATCHES)})")
        for system in ("random floor", "bm25", "dense", "concepts only",
                       "concepts + ordinal"):
            print(f"  {system:<{width}}  " + " / ".join(f"{v:.4f}" for v in rows[system]))
    RESULTS.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
