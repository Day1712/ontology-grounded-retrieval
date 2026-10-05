"""BM25 over grounded concept IRIs instead of text, beside text BM25 and flat
structured matching, with nDCG@10, C_q and paired 95% intervals.
Writes cache/concept_bm25.json (concept-BM25 table)."""

import json
import statistics
import sys
from pathlib import Path

import bm25s

from evaluation import blind, corpus, evaluate, qrels
from evaluation.baselines import bm25 as text_bm25
from ranking_method import rank, represent
from evaluation.concordance import concordance

from experiments import ablation, benchmark

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "cache" / "concept_bm25.json"
FLAT = "both, flat, ordinal identity"


def concept_tokens(item_id, text, subject, grounded, ladder, requiring):
    found = rank.stated(item_id, text, subject, grounded, ladder, requiring)
    return sorted({str(a.object) for a in found})


def concept_run(queries, documents, grounded, ladder):
    docs = [concept_tokens(d.id, d.text, represent.person, grounded, ladder, False)
            for d in documents]
    retriever = bm25s.BM25(k1=text_bm25.K1, b=text_bm25.B, method=text_bm25.METHOD)
    retriever.index(docs, show_progress=False)
    run, empty = {}, 0
    for q in queries:
        toks = concept_tokens(q.id, q.text, represent.role, grounded, ladder, True)
        known = [t for t in toks if t in retriever.vocab_dict]
        if not known:
            empty += 1
            run[q.id] = {d.id: 0.0 for d in documents}
            continue
        scores = retriever.get_scores(known)
        run[q.id] = {d.id: float(s) for d, s in zip(documents, scores, strict=True)}
    return run, empty, [len(t) for t in docs]


def main(batches=("threeitem",)):
    records = corpus.load_corpus()
    cached = benchmark.grounding([r.profile_text for r in records])
    blinded = blind.blind(records)
    by_batch = {b: [r for r in records if r.batch_name == b] for b in batches}
    built = ablation.groundings(records, cached)["lexical+semantic"]
    ladder = ablation.LADDERS[FLAT][1]
    out = {}
    for batch, rs in by_batch.items():
        view = blind.select(blinded, rs)
        runs = {
            "text BM25": text_bm25.rank(view.queries, view.documents),
            "flat structured matching": rank.rank(view.queries, view.documents,
                                                  grounded=built, ladder=ladder,
                                                  key=rank.LEXICOGRAPHIC),
        }
        crun, empty, lens = concept_run(view.queries, view.documents, built, ladder)
        runs["concept BM25"] = crun
        print(f"[{batch}] queries without known concept tokens: {empty}; "
              f"CV concept-set size median {statistics.median(lens)}", file=sys.stderr)
        for gold, module in ablation.GOLDS.items():
            judged = qrels.blind_qrels(qrels.build_qrels(rs, module.scorer), blinded)
            judged = {q: v for q, v in judged.items() if q in runs["text BM25"]}
            pq = {}
            for name, run in runs.items():
                scored = evaluate.evaluate(run, judged)
                pq[name] = {q: v["ndcg_cut_10"] for q, v in scored.items()}
                out.setdefault(batch, {}).setdefault(gold, {})[name] = {
                    "ndcg10": statistics.mean(pq[name].values()),
                    "cq": concordance(run, judged)["cq"],
                }
            for a, b in (("concept BM25", "text BM25"),
                         ("flat structured matching", "concept BM25")):
                mean, half = ablation._interval([pq[a][q] - pq[b][q] for q in sorted(pq[a])])
                out[batch][gold][f"{a} - {b}"] = {"mean": mean, "half": half}
    OUT.write_text(json.dumps(out, indent=2))
    for batch, golds in out.items():
        print(f"\n== {batch}")
        for gold, rows in golds.items():
            print(f"  {gold}")
            for name, v in rows.items():
                if "ndcg10" in v:
                    print(f"    {name:26s} nDCG@10 {v['ndcg10']:.3f}  C_q {v['cq']:.3f}")
                else:
                    print(f"    {name:48s} {v['mean']:+.3f} ± {v['half']:.3f}")


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("threeitem",))
