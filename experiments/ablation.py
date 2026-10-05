"""Ablation of grounding (none/lexical/semantic/both) and ordinal reading, plus the
ladder rows at full grounding, with paired 95% intervals over queries.
Writes cache/ablation.json (ablation table)."""

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from evaluation import blind, corpus, count, evaluate, importance, qrels, qualified
from ranking_method import link, ordinal, policy, rank, represent, resource

from experiments import benchmark

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "cache" / "ablation.json"
BATCHES = ("oneitem", "twoitem", "threeitem")
GOLDS = {"count": count, "weighted": importance, "qualified": qualified}
GROUNDINGS = ("none", "lexical", "semantic", "lexical+semantic")

#: Ladder rows at full grounding: the method with one thing removed or added.
LADDERS = {
    "lexical only": ("lexical", policy.CONCEPTS_ORDINAL),
    "semantic only": ("semantic", policy.CONCEPTS_ORDINAL),
    "both": ("lexical+semantic", policy.CONCEPTS_ORDINAL),
    "both, flat requirements": ("lexical+semantic", policy.CONCEPTS_ORDINAL_FLAT),
    "both, ordinal identity": ("lexical+semantic", policy.ORDINAL_IDENTITY),
    "both, flat, ordinal identity": ("lexical+semantic", policy.FLAT_IDENTITY),
    "both, no ordinal, graded": ("lexical+semantic", policy.CONCEPTS_ONLY_GRADED),
    "both, no ordinal": ("lexical+semantic", policy.CONCEPTS),
    "both + hierarchy": ("lexical+semantic", policy.Ladder(hierarchy=True, graded=True)),
}

#: Each pair differs in one thing.
LADDER_CONTRASTS = (
    ("does the ontology's requirement grading change the ranking?",
     "both", "both, flat requirements"),
    ("do the ordered scales help?", "both", "both, no ordinal, graded"),
    ("do the published categories help?",
     "both, ordinal identity", "both, no ordinal, graded"),
    ("does O*NET's ordering of the scales matter?", "both", "both, ordinal identity"),
    ("does the hierarchy change the ranking?", "both + hierarchy", "both"),
    ("what do grading and ordering add together?", "both", "both, flat, ordinal identity"),
    ("does ordering help without grading?", "both, flat requirements", "both, flat, ordinal identity"),
    ("do the categories help without grading?", "both, flat, ordinal identity", "both, no ordinal"),
    ("does the ordinal reading help?", "both, flat requirements", "both, no ordinal"),
)

#: Contrasts on the grounding x ordinal grid.
CONTRASTS = (
    ("does ordinal grounding help?", ("lexical+semantic", True), ("lexical+semantic", False)),
    ("does concept grounding help?", ("lexical+semantic", True), ("none", True)),
    ("does semantic linking help?", ("lexical+semantic", True), ("lexical", True)),
    ("do embeddings help at all?", ("lexical+semantic", True), ("lexical", True)),
)


def _ordinal_items(text):
    """The ordered-scale readings (no model involved)."""
    return [
        link.GroundedItem(element_id, category)
        for element_id in policy.ordinal_elements()
        for category in (ordinal.level(element_id, text),)
        if category is not None
    ]


def groundings(records, cached):
    """{grounding name: {text key: items}}, ordinal items always included;
    whether they are used is set by the ladder."""
    out = {name: {} for name in GROUNDINGS}
    for record in records:
        text = record.profile_text
        key = rank.text_key(text)
        ordinals = _ordinal_items(text)

        by_name = {"none": []}
        concepts = [i for i in cached[key] if i.category is None]
        for name, wanted in (("lexical", {link.LEXICAL}),
                             ("semantic", {link.SEMANTIC}),
                             ("lexical+semantic", {link.LEXICAL, link.SEMANTIC})):
            kept = []
            for item in concepts:
                support = tuple(s for s in item.support if s.mechanism in wanted)
                if support:
                    kept.append(link.GroundedItem(item.element_id, None, support))
            by_name[name] = kept

        for name, items in by_name.items():
            out[name][key] = tuple(items) + tuple(ordinals)
    return out


def _requirement_branch(assertion):
    """Which branch a stated requirement belongs to; both ordered scales map to
    "ordinal"."""
    if assertion.predicate == represent.ONET_CATEGORY:
        return "ordinal"
    return resource.branch(resource.by_iri()[assertion.object])


def status_share(grounded, blinded, by_batch, ladder):
    """How every stated requirement resolved, per branch, over all scored pairs."""
    counts = defaultdict(lambda: defaultdict(int))
    for batch, rs in by_batch.items():
        view = blind.select(blinded, rs)
        held = [represent.index(represent.assertions(
                    represent.person(d.id), represent.document(d.id),
                    grounded[rank.text_key(d.text)]))
                for d in view.documents]
        for query in view.queries:
            required = represent.assertions(
                represent.role(query.id), represent.document(query.id),
                grounded[rank.text_key(query.text)], requiring=True)
            for evidence in held:
                for assertion, verdict in policy.explain(required, evidence, ladder):
                    counts[_requirement_branch(assertion)][verdict.status] += 1
    return {b: dict(v) for b, v in counts.items()}


def _interval(differences):
    """Mean paired difference and a 95% normal-approximation interval."""
    n = len(differences)
    mean = statistics.mean(differences)
    if n < 2:
        return mean, 0.0
    half = 1.96 * statistics.stdev(differences) / math.sqrt(n)
    return mean, half


def report(device=benchmark.DEVICE, cache=benchmark.CACHE):
    records = corpus.load_corpus()
    cached = benchmark.grounding([r.profile_text for r in records], device, cache)

    blinded = blind.blind(records)
    by_batch = {b: [r for r in records if r.batch_name == b] for b in BATCHES}
    judged = {
        gold: {b: qrels.blind_qrels(qrels.build_qrels(rs, mod.scorer), blinded)
               for b, rs in by_batch.items()}
        for gold, mod in GOLDS.items()
    }

    built = groundings(records, cached)

    # per query scores, keyed (grounding, ordinal, gold, batch) -> {query: ndcg}
    per_query = {}
    for name, grounded in built.items():
        for use_ordinal in (True, False):
            if name == "none" and not use_ordinal:
                continue
            for batch, rs in by_batch.items():
                view = blind.select(blinded, rs)
                run = rank.rank(view.queries, view.documents, grounded=grounded,
                                ladder=policy.Ladder(ordinal=use_ordinal),
                                key=rank.LEXICOGRAPHIC)
                for gold in GOLDS:
                    scored = evaluate.evaluate(run, judged[gold][batch])
                    per_query[(name, use_ordinal, gold, batch)] = {
                        q: v["ndcg_cut_10"] for q, v in scored.items()
                    }

    means = defaultdict(dict)
    for (name, use_ordinal, gold, batch), scores in per_query.items():
        label = f"{name} {'+ordinal' if use_ordinal else '-ordinal'}"
        means[gold].setdefault(label, {})[batch] = statistics.mean(scores.values())

    # the grid's rows are ungraded, so they must reproduce benchmark's flat rows
    known = json.loads((ROOT / "cache" / "results.json").read_text())
    for label, system in (("lexical+semantic +ordinal", "concepts + ordinal, flat"),
                          ("lexical+semantic -ordinal", "concepts only")):
        for i, batch in enumerate(BATCHES):
            for gold in GOLDS:
                a = means[gold][label][batch]
                b = known[gold][system][i]
                assert abs(a - b) < 1e-9, f"{label}/{gold}/{batch}: {a} != {b}"

    contrasts = {}
    for question, (an, ao), (bn, bo) in CONTRASTS:
        rows = {}
        for gold in GOLDS:
            for batch in BATCHES:
                a = per_query[(an, ao, gold, batch)]
                b = per_query[(bn, bo, gold, batch)]
                diff = [a[q] - b[q] for q in sorted(a)]
                mean, half = _interval(diff)
                rows.setdefault(gold, {})[batch] = {
                    "mean": mean, "half": half, "n": len(diff),
                    "distinguishable": abs(mean) > half,
                }
        contrasts[question] = rows

    # the ladder rows
    ladder_per_query, ladder_means = {}, defaultdict(dict)
    for name, (grounding, ladder) in LADDERS.items():
        for batch, rs in by_batch.items():
            view = blind.select(blinded, rs)
            run = rank.rank(view.queries, view.documents, grounded=built[grounding],
                            ladder=ladder, key=rank.LEXICOGRAPHIC)
            for gold in GOLDS:
                scored = evaluate.evaluate(run, judged[gold][batch])
                ladder_per_query[(name, gold, batch)] = {
                    q: v["ndcg_cut_10"] for q, v in scored.items()
                }
                ladder_means[gold].setdefault(name, {})[batch] = statistics.mean(
                    v["ndcg_cut_10"] for v in scored.values())

    # 'both' and 'both, no ordinal' must reproduce benchmark's rows
    for i, batch in enumerate(BATCHES):
        for gold in GOLDS:
            for name, system in (("both", "concepts + ordinal"),
                                 ("both, no ordinal", "concepts only")):
                a = ladder_means[gold][name][batch]
                b = known[gold][system][i]
                assert abs(a - b) < 1e-9, f"{name}/{gold}/{batch}: {a} != {b}"

    ladder_contrasts = {}
    for question, a_name, b_name in LADDER_CONTRASTS:
        rows = {}
        for gold in GOLDS:
            for batch in BATCHES:
                a = ladder_per_query[(a_name, gold, batch)]
                b = ladder_per_query[(b_name, gold, batch)]
                diff = [a[q] - b[q] for q in sorted(a)]
                mean, half = _interval(diff)
                rows.setdefault(gold, {})[batch] = {
                    "mean": mean, "half": half, "n": len(diff),
                    "distinguishable": abs(mean) > half,
                }
        ladder_contrasts[question] = rows

    return {
        "means": dict(means),
        "contrasts": contrasts,
        "ladder_means": {g: dict(v) for g, v in ladder_means.items()},
        "ladder_contrasts": ladder_contrasts,
        "status_share": status_share(built["lexical+semantic"], blinded, by_batch,
                                     policy.HIERARCHY),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=benchmark.DEVICE)
    args = parser.parse_args()
    out = report(device=args.device)

    labels = [f"{n} {o}" for n in GROUNDINGS for o in ("+ordinal", "-ordinal")
              if not (n == "none" and o == "-ordinal")]
    width = max(len(x) for x in labels)
    for gold in GOLDS:
        print(f"\n{gold}   ({' / '.join(BATCHES)})")
        for label in labels:
            row = out["means"][gold][label]
            print(f"  {label:<{width}}  " + " / ".join(f"{row[b]:.4f}" for b in BATCHES))

    print("\n\npaired differences, 95% interval over queries")
    for question, rows in out["contrasts"].items():
        print(f"\n{question}")
        for gold in GOLDS:
            cells = []
            for batch in BATCHES:
                c = rows[gold][batch]
                mark = " " if c["distinguishable"] else "~"
                cells.append(f"{c['mean']:+.4f}±{c['half']:.4f}{mark}")
            print(f"  {gold:<10} " + "  ".join(cells))
    print("\n~ marks a difference whose interval includes zero.")

    width = max(len(x) for x in LADDERS)
    print("\n\nthe four rows, at full grounding")
    for gold in GOLDS:
        print(f"\n{gold}   ({' / '.join(BATCHES)})")
        for name in LADDERS:
            row = out["ladder_means"][gold][name]
            print(f"  {name:<{width}}  " + " / ".join(f"{row[b]:.4f}" for b in BATCHES))

    print("\n\nrow contrasts, 95% interval over queries")
    for question, rows in out["ladder_contrasts"].items():
        print(f"\n{question}")
        for gold in GOLDS:
            cells = []
            for batch in BATCHES:
                c = rows[gold][batch]
                mark = " " if c["distinguishable"] else "~"
                cells.append(f"{c['mean']:+.4f}±{c['half']:.4f}{mark}")
            print(f"  {gold:<10} " + "  ".join(cells))

    print("\n\nrequirement status by branch, hierarchy on, over every scored pair")
    share = out["status_share"]
    order = (policy.EXACT, policy.RELATED, policy.UNMATCHED,
             policy.MET, policy.SHORT, policy.UNSTATED)
    print(f"  {'branch':8s}" + "".join(f"{s:>12s}" for s in order))
    for branch in sorted(share):
        total = sum(share[branch].values())
        print(f"  {branch:8s}" + "".join(
            f"{share[branch].get(s, 0) / total:11.1%} " for s in order))

    RESULTS.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
