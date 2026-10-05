"""Checks the PROV-O decision record against the ranker and the assertion graph.

Recorded scores must equal the ranker's exactly, verdicts must match the stated
requirements and add up to the recorded weights and coverage, and every
``ext:satisfiedBy`` target must be an annotation in ``graph.ttl``.
"""

import argparse
import json
from pathlib import Path

import rdflib
from rdflib.namespace import RDF

from evaluation import blind, corpus
from ranking_method import policy, rank, represent

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "cache" / "record_checks.json"
EXT = rdflib.Namespace(represent.EXT)
OA = rdflib.Namespace(represent.OA)


def recorded(graph):
    """``{(role, candidate): {...}}`` -- every score the record states."""
    out = {}
    for node in graph.subjects(RDF.type, EXT.Score):
        role = str(graph.value(node, EXT.forRole)).rsplit("/", 1)[1]
        candidate = str(graph.value(node, EXT.forCandidate)).rsplit("/", 1)[1]
        out[(role, candidate)] = {
            "node": node,
            "score": float(graph.value(node, EXT.rankingScore)),
            "coverage": float(graph.value(node, EXT.coverage)),
            "satisfied": float(graph.value(node, EXT.satisfiedWeight)),
            "total": float(graph.value(node, EXT.totalWeight)),
            "rank": int(graph.value(node, EXT.rank)),
            "verdicts": list(graph.objects(node, rdflib.URIRef(
                represent.PROV + "wasDerivedFrom"))),
        }
    return out


def scores_agree(record, graph, grounded, blinded, records, ladder):
    """Recompute every recorded pair with the ranker and compare exactly."""
    wanted = {}
    for (role, candidate) in record:
        wanted.setdefault(role, set()).add(candidate)
    by_batch = {}
    for r in records:
        by_batch.setdefault(r.batch_name, []).append(r)
    fresh = {}
    for batch, rs in by_batch.items():
        view = blind.select(blinded, rs)
        queries = [q for q in view.queries if q.id in wanted]
        if not queries:
            continue
        run = rank.rank(queries, view.documents, grounded=grounded,
                        ladder=ladder, key=rank.LEXICOGRAPHIC)
        for role, row in run.items():
            for candidate in wanted[role]:
                fresh[(role, candidate)] = row[candidate]
    bad = []
    for key, found in record.items():
        if key not in fresh:
            bad.append((key, "pair not reproducible from the corpus"))
        elif fresh[key] != found["score"]:
            bad.append((key, f"recorded {found['score']!r}, ranker {fresh[key]!r}"))
    return bad, len(fresh)


def verdicts_agree(record, graph, grounded, blinded, records, ladder):
    """Verdict count, weights and coverage against the requirements stated."""
    stated_count = {}
    by_batch = {}
    for r in records:
        by_batch.setdefault(r.batch_name, []).append(r)
    for batch, rs in by_batch.items():
        view = blind.select(blinded, rs)
        for query in view.queries:
            stated_count[query.id] = len(rank.stated(
                query.id, query.text, represent.role, grounded, ladder,
                requiring=True))
    bad = []
    for (role, candidate), found in record.items():
        if len(found["verdicts"]) != stated_count.get(role):
            bad.append(((role, candidate),
                        f"{len(found['verdicts'])} verdicts recorded, "
                        f"{stated_count.get(role)} requirements stated"))
            continue
        weights = met = 0.0
        for verdict in found["verdicts"]:
            weight = graph.value(verdict, EXT.weight)
            if weight is None:
                bad.append(((role, candidate), f"{verdict} has no ext:weight"))
                break
            weights += float(weight)
            if str(graph.value(verdict, EXT.outcome)) in (policy.EXACT, policy.MET):
                met += float(weight)
        else:
            if abs(weights - found["total"]) > 1e-9:
                bad.append(((role, candidate),
                            f"verdict weights {weights} != totalWeight "
                            f"{found['total']}"))
            elif abs(met - found["satisfied"]) > 1e-9:
                bad.append(((role, candidate),
                            f"met weights {met} != satisfiedWeight "
                            f"{found['satisfied']}"))
            elif found["total"] and abs(
                    found["satisfied"] / found["total"] - found["coverage"]) > 1e-12:
                bad.append(((role, candidate), "coverage is not satisfied/total"))
    return bad


def pointers_resolve(graph, assertions):
    """Every ``ext:satisfiedBy`` target must be an annotation the graph states."""
    known = set(assertions.subjects(RDF.type, OA.Annotation))
    targets = set(graph.objects(None, EXT.satisfiedBy))
    missing = sorted(str(t) for t in targets - known)
    return missing, len(targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("record", type=Path, help="a written decision record")
    parser.add_argument("--graph", type=Path, default=ROOT / "cache" / "graph.ttl",
                        help="the assertion graph its pointers name")
    parser.add_argument("--device", default="mps")
    args = parser.parse_args()

    from experiments import benchmark

    graph = rdflib.Graph().parse(args.record, format="turtle")
    print(f"{len(graph):,} triples in the record")
    record = recorded(graph)
    print(f"{len(record):,} scored pairs recorded")

    records = corpus.load_corpus()
    grounded = benchmark.grounding([r.profile_text for r in records], args.device)
    blinded = blind.blind(records)
    ladder = policy.CONCEPTS_ORDINAL

    wrong_scores, reproduced = scores_agree(
        record, graph, grounded, blinded, records, ladder)
    print(f"scores recomputed with the ranker: {reproduced:,}; "
          f"{len(wrong_scores)} disagree")
    wrong_verdicts = verdicts_agree(
        record, graph, grounded, blinded, records, ladder)
    print(f"verdict counts, weights and coverage: {len(wrong_verdicts)} disagree")

    assertions = rdflib.Graph().parse(args.graph, format="turtle")
    missing, targets = pointers_resolve(graph, assertions)
    print(f"ext:satisfiedBy targets: {targets:,}; {len(missing)} not in "
          f"{args.graph.name}")

    for where, why in (wrong_scores + wrong_verdicts)[:10]:
        print(f"  FAIL {where} -- {why}")
    for iri in missing[:10]:
        print(f"  FAIL dangling {iri}")

    failures = len(wrong_scores) + len(wrong_verdicts) + len(missing)
    RESULTS.write_text(json.dumps({
        "record": str(args.record),
        "graph": str(args.graph),
        "triples": len(graph),
        "pairs": len(record),
        "scores_reproduced": reproduced,
        "score_disagreements": [[list(w), why] for w, why in wrong_scores],
        "verdict_disagreements": [[list(w), why] for w, why in wrong_verdicts],
        "satisfied_by_targets": targets,
        "dangling": missing,
    }, indent=1) + "\n")
    print(f"wrote {RESULTS}")
    print("the record is what was decided" if not failures
          else f"{failures} failures")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
