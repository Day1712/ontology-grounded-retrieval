"""Occupation-held-out selection of the grounding setting: the 54-setting grid is
scored on half the occupations and the choice evaluated on the other half.
Writes cache/heldout/ (table "occupation-held-out ranking").

    grid --case <label>   per-record grounding counts for 18 settings (~100 s)
    splits                selection over random splits (seconds)
    rank                  ranking on held-out pools for the first splits
"""

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluation import blind, corpus, evaluate, qrels
from evaluation.baselines import bm25
from ranking_method import link, rank

from experiments import ablation, benchmark, thresholds

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "cache" / "heldout"
BR = thresholds.BRANCHES
SETTINGS = [(c, k, s) for c in thresholds.CASE for k in thresholds.KS
            for s in thresholds.SEMANTIC]
SHIPPED = ("skills+knowledge", 1, link.SEMANTIC_THRESHOLD)
N_SPLITS = 100
N_RANKED = 2


def _load():
    records = corpus.load_corpus()
    texts = [r.profile_text for r in records]
    _, scored = thresholds.similarity(texts, benchmark.DEVICE, thresholds.CACHE,
                                      link.vocabulary())
    return records, texts, [scored[rank.text_key(t)] for t in texts]


def _annotate(texts, order, setting):
    case, k, sem = setting
    return link.annotate(texts, vocab=thresholds.vocabulary(thresholds.CASE[case]),
                         scored=order, semantic_threshold=sem,
                         lexical_threshold=link.LEXICAL_SUPPORT_THRESHOLD, k=k)


def grid(case):
    records, texts, order = _load()
    gold = thresholds.intended(records)
    mine = [s for s in SETTINGS if s[0] == case]
    counts = np.zeros((len(mine), len(records), len(BR), 3), dtype=np.int32)
    for si, setting in enumerate(mine):
        items = _annotate(texts, order, setting)
        for ri, (want, row) in enumerate(zip(gold, items)):
            got = {x.element_id for x in row if x.category is None}
            got = {i for i in got if link.resource.branch(i) in BR}
            for i in got & want:
                counts[si, ri, BR.index(link.resource.branch(i)), 0] += 1
            for i in got - want:
                counts[si, ri, BR.index(link.resource.branch(i)), 1] += 1
            for i in want - got:
                counts[si, ri, BR.index(link.resource.branch(i)), 2] += 1
        print(f"  {setting}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / f"grid_{case}.npz", counts=counts,
             settings=np.array([json.dumps(s) for s in mine]))


def _counts():
    blocks, names = [], []
    for case in thresholds.CASE:
        z = np.load(OUT / f"grid_{case}.npz")
        blocks.append(z["counts"])
        names += [tuple(json.loads(s)) for s in z["settings"]]
    return np.concatenate(blocks), names


def macro_f1(counts, mask):
    """Macro-F1 over branches of every setting, on the records in ``mask``."""
    summed = counts[:, mask].sum(axis=1).astype(float)  # (settings, branches, 3)
    tp, fp, fn = (summed[..., j] for j in range(3))
    p = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    r = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f = np.divide(2 * p * r, p + r, out=np.zeros_like(tp), where=(p + r) > 0)
    return f.mean(axis=1)


def split(records, seed):
    """Two halves of the occupations, stratified by SOC major group."""
    rng = random.Random(seed)
    by_major = defaultdict(list)
    for occ in sorted({r.occ_code for r in records}):
        by_major[occ.split("-")[0]].append(occ)
    a, b, extra = set(), set(), 0
    for major in sorted(by_major):
        occs = by_major[major][:]
        rng.shuffle(occs)
        half = len(occs) // 2
        if len(occs) % 2:
            half += extra
            extra = 1 - extra
        a |= set(occs[:half])
        b |= set(occs[half:])
    return a, b


def splits():
    records = corpus.load_corpus()
    counts, names = _counts()
    shipped = names.index(tuple(json.loads(json.dumps(SHIPPED))))
    full = macro_f1(counts, np.ones(len(records), dtype=bool))
    assert int(np.argmax(full)) == shipped, "full-corpus grid does not select the shipped setting"
    occ = np.array([r.occ_code for r in records])
    rows = []
    for seed in range(N_SPLITS):
        a, b = split(records, seed)
        for sel, test in ((a, b), (b, a)):
            ms, mt = np.isin(occ, list(sel)), np.isin(occ, list(test))
            fs, ft = macro_f1(counts, ms), macro_f1(counts, mt)
            chosen = int(np.argmax(fs))
            rows.append({
                "seed": seed, "selection": sorted(sel), "chosen": names[chosen],
                "is_shipped": chosen == shipped,
                "heldout_f1_chosen": float(ft[chosen]),
                "heldout_f1_shipped": float(ft[shipped]),
                "heldout_f1_best": float(ft.max()),
                "heldout_best": names[int(np.argmax(ft))],
                "selection_f1_chosen": float(fs[chosen]),
                "heldout_margin": float(ft[chosen] - np.delete(ft, chosen).max()),
            })
    tally = defaultdict(int)
    for r in rows:
        tally[json.dumps(r["chosen"])] += 1
    gap_best = [r["heldout_f1_best"] - r["heldout_f1_chosen"] for r in rows]
    gap_ship = [r["heldout_f1_shipped"] - r["heldout_f1_chosen"] for r in rows]
    gen = [abs(r["selection_f1_chosen"] - r["heldout_f1_chosen"]) for r in rows]
    summary = {
        "selections": len(rows),
        "shipped_chosen": sum(r["is_shipped"] for r in rows),
        "chosen_settings": dict(tally),
        "full_corpus_f1_shipped": float(full[shipped]),
        "heldout_f1_chosen_mean": statistics.mean(r["heldout_f1_chosen"] for r in rows),
        "heldout_f1_chosen_range": [min(r["heldout_f1_chosen"] for r in rows),
                                    max(r["heldout_f1_chosen"] for r in rows)],
        "regret_vs_heldout_best_mean": statistics.mean(gap_best),
        "regret_vs_heldout_best_max": max(gap_best),
        "shipped_minus_chosen_max": max(gap_ship),
        "abs_selection_minus_heldout_f1_mean": statistics.mean(gen),
        "heldout_margin_over_runner_up_min": min(r["heldout_margin"] for r in rows),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "splits.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


def ranked():
    records, texts, order = _load()
    rows = json.loads((OUT / "splits.json").read_text())["rows"][:2 * N_RANKED]
    blinded = blind.blind(records)
    cached = benchmark.grounding(texts)
    out = []
    for row in rows:
        test = {r.occ_code for r in records} - set(row["selection"])
        setting = tuple(row["chosen"])
        if setting == SHIPPED:
            grounded = cached
        else:
            items = _annotate(texts, order, setting)
            grounded = {rank.text_key(t): i for t, i in zip(texts, items)}
        result = {"seed": row["seed"], "chosen": row["chosen"], "ndcg10": {}}
        for batch in ablation.BATCHES:
            rs = [r for r in records if r.batch_name == batch and r.occ_code in test]
            view = blind.select(blinded, rs)
            runs = {
                "method": rank.rank(view.queries, view.documents, grounded=grounded,
                                    key=rank.LEXICOGRAPHIC),
                "bm25": bm25.rank(view.queries, view.documents),
            }
            for gold, mod in ablation.GOLDS.items():
                judged = qrels.blind_qrels(qrels.build_qrels(rs, mod.scorer), blinded)
                for system, run in runs.items():
                    result["ndcg10"].setdefault(gold, {}).setdefault(system, {})[batch] = \
                        evaluate.mean(evaluate.evaluate(run, judged))["ndcg_cut_10"]
        out.append(result)
        print(json.dumps(result), flush=True)
    (OUT / "ranked.json").write_text(json.dumps(out, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("grid", "splits", "rank"))
    parser.add_argument("--case", choices=tuple(thresholds.CASE))
    args = parser.parse_args()
    if args.step == "grid":
        grid(args.case)
    elif args.step == "splits":
        splits()
    else:
        ranked()


if __name__ == "__main__":
    main()
