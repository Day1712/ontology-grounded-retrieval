"""Effect of the graded requirement relations: grading coverage, nDCG@10 with and
without grading, top-10 displacement and sensitivity to the ungraded weight.
Writes cache/grading_quality.json."""

import argparse
import collections
import json
import statistics
from pathlib import Path

from evaluation import blind, corpus, count, evaluate, importance, qrels, qualified
from ranking_method import grade, policy, rank, represent, resource

from experiments import benchmark

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "cache" / "grading_quality.json"
BATCHES = benchmark.BATCHES
GOLDS = {"count": count, "weighted": importance, "qualified": qualified}
SHORT = {
    represent.REQUIRES_ESSENTIAL: "essential",
    represent.REQUIRES_IMPORTANT: "important",
    represent.REQUIRES_SOMEWHAT: "somewhat",
}
KNOWN = ROOT / "cache" / "results.json"


def read_grades(records, grounded):
    """``{custom_id: {element id: relation}}`` over the job descriptions."""
    return {
        r.custom_id: grade.read(r.profile_text, grounded[rank.text_key(r.profile_text)])
        for r in records if r.profile_type == "jd"
    }


def distribution(records, grounded, graded):
    """How the reading grades the corpus, and each group's mean O*NET importance."""
    counts = collections.Counter()
    values = collections.defaultdict(list)
    for record in records:
        if record.profile_type != "jd":
            continue
        found = graded[record.custom_id]
        known = {k: v["importance"] for k, v in record.profile_dict.items()
                 if "importance" in v}
        for item in grounded[rank.text_key(record.profile_text)]:
            if item.category is not None:
                continue
            if resource.branch(item.element_id) not in grade.gradeable():
                continue
            label = SHORT.get(found.get(item.element_id), "generic")
            counts[label] += 1
            if item.element_id in known:
                values[label].append(known[item.element_id])
    total = sum(counts.values())
    return {
        "gradeable requirements": total,
        "graded": 1 - counts["generic"] / total if total else 0.0,
        "groups": {
            label: {
                "n": counts[label],
                "share": counts[label] / total if total else 0.0,
                "mean importance": statistics.mean(values[label]) if values[label] else None,
                "sd importance": (statistics.stdev(values[label])
                                  if len(values[label]) > 1 else None),
                "rated": len(values[label]),
            }
            for label in ("essential", "important", "somewhat", "generic")
        },
    }


def _order(row, k=10):
    """The top ``k`` document ids, highest score first, ties broken by id."""
    return tuple(d for d, _ in sorted(row.items(), key=lambda kv: (-kv[1], kv[0]))[:k])


def displaced(flat, graded, k=10):
    """Share of queries whose top-``k`` ordering the grading changed, and share
    whose top-``k`` membership it changed."""
    reordered = rewritten = 0
    for query, row in flat.items():
        a, b = _order(row, k), _order(graded[query], k)
        reordered += a != b
        rewritten += set(a) != set(b)
    n = len(flat)
    return {"queries": n,
            "reordered": reordered / n if n else 0.0,
            "membership changed": rewritten / n if n else 0.0}


def report(device=benchmark.DEVICE, cache=benchmark.CACHE):
    records = corpus.load_corpus()
    grounded = benchmark.grounding([r.profile_text for r in records], device, cache)
    blinded = blind.blind(records)
    by_batch = {b: [r for r in records if r.batch_name == b] for b in BATCHES}
    judged = {
        gold: {b: qrels.blind_qrels(qrels.build_qrels(rs, mod.scorer), blinded)
               for b, rs in by_batch.items()}
        for gold, mod in GOLDS.items()
    }

    def runs(ladder):
        out = {}
        for batch, rs in by_batch.items():
            view = blind.select(blinded, rs)
            out[batch] = rank.rank(view.queries, view.documents, grounded=grounded,
                                   ladder=ladder, key=rank.LEXICOGRAPHIC)
        return out

    def ndcg(run):
        return {gold: {b: statistics.mean(v["ndcg_cut_10"] for v in
                                          evaluate.evaluate(run[b], judged[gold][b]).values())
                       for b in BATCHES} for gold in GOLDS}

    flat, graded_runs = runs(policy.CONCEPTS_ORDINAL_FLAT), runs(policy.CONCEPTS_GRADED)

    # the flat rung must reproduce benchmark's flat row
    known = json.loads(KNOWN.read_text())
    flat_ndcg = ndcg(flat)
    for i, batch in enumerate(BATCHES):
        for gold in GOLDS:
            a, b = flat_ndcg[gold][batch], known[gold]["concepts + ordinal, flat"][i]
            assert abs(a - b) < 1e-12, f"flat rung moved: {gold}/{batch} {a} != {b}"

    out = {
        "reading": distribution(records, grounded, read_grades(records, grounded)),
        "ndcg": {"flat": flat_ndcg, "graded": ndcg(graded_runs)},
        "displacement": {b: displaced(flat[b], graded_runs[b]) for b in BATCHES},
    }

    # sensitivity to the weight of ungraded requirements
    stated = policy.UNGRADED_WEIGHT
    sensitivity = {}
    try:
        for w in (1.0, 2.0, 3.0):
            policy.UNGRADED_WEIGHT = w
            run = runs(policy.CONCEPTS_GRADED)
            sensitivity[str(w)] = {
                "ndcg": ndcg(run),
                "displacement": {b: displaced(flat[b], run[b]) for b in BATCHES},
            }
    finally:
        policy.UNGRADED_WEIGHT = stated
    out["ungraded weight sensitivity"] = sensitivity
    out["stated ungraded weight"] = stated
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=benchmark.DEVICE)
    args = parser.parse_args()
    found = report(device=args.device)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(found, indent=1) + "\n")

    r = found["reading"]
    print(f"gradeable requirements {r['gradeable requirements']:,}   "
          f"graded {r['graded']*100:.1f}%")
    print(f"\n{'group':10s} {'n':>7s} {'share':>7s} {'mean imp':>9s} {'sd':>6s}")
    for label, g in r["groups"].items():
        mi = f"{g['mean importance']:9.2f}" if g["mean importance"] is not None else " " * 9
        sd = f"{g['sd importance']:6.2f}" if g["sd importance"] is not None else " " * 6
        print(f"{label:10s} {g['n']:7,d} {g['share']*100:6.1f}% {mi} {sd}")
    print(f"\n{'gold':10s} {'batch':10s} {'flat':>8s} {'graded':>8s} {'delta':>8s}")
    for gold in GOLDS:
        for b in BATCHES:
            a = found["ndcg"]["flat"][gold][b]
            c = found["ndcg"]["graded"][gold][b]
            print(f"{gold:10s} {b:10s} {a:8.4f} {c:8.4f} {c-a:+8.4f}")
    print(f"\n{'batch':10s} {'queries':>8s} {'top-10 reordered':>18s} {'membership changed':>20s}")
    for b in BATCHES:
        d = found["displacement"][b]
        print(f"{b:10s} {d['queries']:8d} {d['reordered']*100:17.1f}% "
              f"{d['membership changed']*100:19.1f}%")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
