"""The flat structured-matching rows of the ablation table and their paired 95%
intervals; rows already in cache/ablation.json must reproduce it exactly.
Writes cache/ablation_flat.json."""

import json
import statistics
from pathlib import Path

from evaluation import blind, corpus, evaluate, qrels
from ranking_method import rank

from experiments import ablation, benchmark

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "cache" / "ablation_flat.json"
ROWS = ("both", "both, flat requirements", "both, flat, ordinal identity",
        "both, ordinal identity", "both, no ordinal")
CONTRASTS = (
    ("full vs flat", "both", "both, flat, ordinal identity"),
    ("ordering, ungraded", "both, flat requirements", "both, flat, ordinal identity"),
    ("categories, ungraded", "both, flat, ordinal identity", "both, no ordinal"),
    ("grading, unordered", "both, ordinal identity", "both, flat, ordinal identity"),
)


def main():
    records = corpus.load_corpus()
    cached = benchmark.grounding([r.profile_text for r in records])
    blinded = blind.blind(records)
    by_batch = {b: [r for r in records if r.batch_name == b] for b in ablation.BATCHES}
    judged = {g: {b: qrels.blind_qrels(qrels.build_qrels(rs, m.scorer), blinded)
                  for b, rs in by_batch.items()} for g, m in ablation.GOLDS.items()}
    built = ablation.groundings(records, cached)["lexical+semantic"]

    per_query, means = {}, {}
    for name in ROWS:
        ladder = ablation.LADDERS[name][1]
        for batch, rs in by_batch.items():
            view = blind.select(blinded, rs)
            run = rank.rank(view.queries, view.documents, grounded=built,
                            ladder=ladder, key=rank.LEXICOGRAPHIC)
            for gold in ablation.GOLDS:
                scored = evaluate.evaluate(run, judged[gold][batch])
                pq = {q: v["ndcg_cut_10"] for q, v in scored.items()}
                per_query[(name, gold, batch)] = pq
                means.setdefault(gold, {}).setdefault(name, {})[batch] = statistics.mean(pq.values())

    known = json.loads((ROOT / "cache" / "ablation.json").read_text())["ladder_means"]
    for gold in ablation.GOLDS:
        for name in ROWS:
            if name in known[gold]:
                for batch in ablation.BATCHES:
                    a, b = means[gold][name][batch], known[gold][name][batch]
                    assert abs(a - b) < 1e-9, f"{name}/{gold}/{batch}: {a} != {b}"

    contrasts = {}
    for label, a_name, b_name in CONTRASTS:
        for gold in ablation.GOLDS:
            for batch in ablation.BATCHES:
                a, b = per_query[(a_name, gold, batch)], per_query[(b_name, gold, batch)]
                mean, half = ablation._interval([a[q] - b[q] for q in sorted(a)])
                contrasts.setdefault(label, {}).setdefault(gold, {})[batch] = {
                    "mean": mean, "half": half, "distinguishable": abs(mean) > half}

    for gold in ablation.GOLDS:
        print(f"\n{gold}   ({' / '.join(ablation.BATCHES)})")
        for name in ROWS:
            print(f"  {name:30s}" + " / ".join(f"{means[gold][name][b]:.4f}"
                                                for b in ablation.BATCHES))
    print("\npaired differences, 95% interval")
    for label, rows in contrasts.items():
        print(f"\n{label}")
        for gold in ablation.GOLDS:
            print(f"  {gold:10s}" + "  ".join(
                f"{c['mean']:+.4f}±{c['half']:.4f}{' ' if c['distinguishable'] else '~'}"
                for c in (rows[gold][b] for b in ablation.BATCHES)))
    RESULTS.write_text(json.dumps({"means": means, "contrasts": contrasts}, indent=1))


if __name__ == "__main__":
    main()

