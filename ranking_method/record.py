"""Writes PROV-O records of ranking decisions (cache/decisions.ttl): per role,
the policy, one outcome per requirement linked to the satisfying CV annotation,
and the resulting score, for the top candidates (``--top``) or one ``--pair``.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation import blind, corpus

from . import emit, grade, link, policy, rank, represent, resource

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "cache" / "decisions.ttl"
BATCHES = ("oneitem", "twoitem", "threeitem")
PROV = "http://www.w3.org/ns/prov#"
XSD = represent.XSD

PREFIXES = (
    "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
    "@prefix schema: <https://schema.org/> .",
    "@prefix oa: <http://www.w3.org/ns/oa#> .",
    f"@prefix prov: <{PROV}> .",
    f"@prefix xsd: <{XSD}> .",
    f"@prefix cnlo: <{represent.CNLO}> .",
    f"@prefix ext: <{represent.EXT}> .",
    "",
)


def _num(value):
    """A number as a round-trippable Turtle literal (unrounded ``repr``)."""
    if isinstance(value, bool) or isinstance(value, int):
        return str(value)
    return repr(float(value))


def plan(ladder, vocab):
    """``(iri, lines)`` for the policy a ranking ran under, identified by a
    digest of every parameter that decides an outcome."""
    payload = {
        "ladder": {"ordinal": ladder.ordinal, "exceeding": ladder.exceeding,
                   "hierarchy": ladder.hierarchy, "graded": ladder.graded},
        "weights": {k: v for k, v in sorted(policy.WEIGHTS.items())},
        "ungraded_weight": policy.UNGRADED_WEIGHT,
        "semantic_threshold": link.SEMANTIC_THRESHOLD,
        "lexical_threshold": str(link.LEXICAL_SUPPORT_THRESHOLD),
        "vocabulary": len(vocab),
        "ordinal_elements": list(policy.ordinal_elements()),
        "release": resource.release(),
        "cues": [list(c) for c in grade.cues()],
        "alignment": {b: list(r) for b, r in sorted(represent.alignment().items())},
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    iri = f"{represent.EXT}policy/{digest[:16]}"
    rung = (f"ordinal={ladder.ordinal}, exceeding={ladder.exceeding}, "
            f"hierarchy={ladder.hierarchy}, graded={ladder.graded}")
    lines = [
        f"<{iri}> a ext:RankingPolicy, prov:Plan ;",
        f'  ext:policyDigest "{digest}" ;',
        f'  ext:ladder "{rung}" ;',
        f"  ext:ungradedWeight {_num(policy.UNGRADED_WEIGHT)} ;",
        f'  rdfs:label "{resource.release()}, {len(vocab)} groundable elements, '
        f'{len(grade.cues())} grading cues" .',
        "",
    ]
    return iri, lines


def _element_of(assertion):
    """The O*NET element a stated requirement is about."""
    if assertion.predicate == represent.ONET_CATEGORY:
        return resource.category_of(assertion.object)[0]
    return resource.by_iri()[assertion.object]


def _satisfied_by(item_id, assertion):
    """The candidate's annotation IRIs for one of their assertions, as in
    the assertion graph."""
    return [emit.annotation_iri(item_id, assertion.object, e.start, e.end)
            for e in assertion.evidence]


def satisfactions(job_id, candidate_id, required, held, ladder, activity):
    """One pair's outcomes: ``(lines, satisfied weight, related weight, total)``."""
    lines, met_mass, related_mass, total = [], 0.0, 0.0, 0.0
    role, person = represent.role(job_id), represent.person(candidate_id)
    for requirement, verdict in policy.explain(required, held, ladder):
        element = _element_of(requirement)
        contributes = policy.weight(requirement, ladder)
        total += contributes
        pairs = [
            ("a", "ext:Satisfaction"),
            ("prov:wasGeneratedBy", f"<{activity}>"),
            ("prov:wasDerivedFrom", f"<{role}>, <{person}>"),
            ("schema:about", f"<{requirement.object}>"),
            ("ext:outcome", f'"{verdict.status}"'),
            ("ext:weight", _num(contributes)),
        ]
        supplying = None
        if verdict.status in (policy.EXACT, policy.MET):
            met_mass += contributes
            # Same key as represent.index: (predicate, element) for ordinals.
            supplying = held.get(
                (requirement.predicate, element)
                if requirement.predicate == represent.ONET_CATEGORY
                else requirement.object)
        elif verdict.status == policy.RELATED:
            related_mass += contributes
            known = resource.elements()
            supplying = held.get(known[verdict.via].iri)
            pairs.append(("ext:via", f"<{known[verdict.via].iri}>"))
            pairs.append(("ext:sharedConcept", f"<{known[verdict.shared].iri}>"))
        if supplying is not None:
            for annotation in _satisfied_by(candidate_id, supplying):
                pairs.append(("ext:satisfiedBy", f"<{annotation}>"))
        node = f"{represent.EXT}satisfaction/{job_id}/{candidate_id}/{element}"
        lines.append(f"<{node}>")
        lines.append(" ;\n".join(f"  {p} {o}" for p, o in pairs) + " .")
        lines.append("")
    return lines, met_mass, related_mass, total


def score(job_id, candidate_id, required, held, ladder, activity, position):
    """One pair's outcomes and the score derived from them.

    ``ext:rankingScore`` (the lexicographic score that ordered candidates) is
    written beside ``ext:coverage`` because they can differ.
    """
    lines, met_mass, related_mass, total = satisfactions(
        job_id, candidate_id, required, held, ladder, activity)
    covered = policy.Coverage(
        met_mass / total if total else 0.0,
        related_mass / total if total else 0.0,
        len(required), total)
    node = f"{represent.EXT}score/{job_id}/{candidate_id}"
    body = [
        f"  ext:forRole <{represent.role(job_id)}>",
        f"  ext:forCandidate <{represent.person(candidate_id)}>",
        f"  ext:satisfiedWeight {_num(met_mass)}",
        f"  ext:totalWeight {_num(total)}",
        f"  ext:coverage {_num(covered.exact)}",
        f"  ext:relatedCoverage {_num(covered.related)}",
        f"  ext:rankingScore {_num(covered.lexicographic())}",
        f"  ext:rank {position}",
        f"  prov:wasGeneratedBy <{activity}>",
    ]
    # No requirements: omit prov:wasDerivedFrom rather than write it empty.
    if required:
        body.append("  prov:wasDerivedFrom " + ", ".join(
            f"<{represent.EXT}satisfaction/{job_id}/{candidate_id}/{_element_of(r)}>"
            for r in required))
    lines += [f"<{node}> a ext:Score ;", " ;\n".join(body) + " .", ""]
    return lines, covered


def ranking(batch, job_id, order, considered, policy_iri, when):
    """The activity: what was ranked, under which plan (via
    ``prov:qualifiedAssociation``), and how wide the pool was."""
    activity = f"{represent.EXT}ranking/{batch}/{job_id}"
    used = ", ".join([f"<{represent.role(job_id)}>"]
                     + [f"<{represent.person(d)}>" for d in order])
    return activity, [
        f"<{activity}> a prov:Activity ;",
        f'  rdfs:label "ranking of {batch} candidates for one role" ;',
        f"  prov:used {used} ;",
        "  prov:qualifiedAssociation [ a prov:Association ;",
        f"    prov:hadPlan <{policy_iri}> ] ;",
        f"  ext:candidatesConsidered {considered} ;",
        f'  prov:startedAtTime "{when}"^^xsd:dateTime .',
        "",
    ]


def order_of(scores, top=None):
    """Candidate ids, best first, ties broken by identifier."""
    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return [d for d, _ in (ranked[:top] if top else ranked)]


def write(handle, records, grounded, blinded, ladder, when, top=None,
          batches=BATCHES, pair=None):
    """Write the record. ``pair`` is ``(job id, candidate id)`` for one pair."""
    vocab = link.vocabulary()
    policy_iri, lines = plan(ladder, vocab)
    handle.write("\n".join(PREFIXES) + "\n")
    handle.write("\n".join(lines) + "\n")
    written = 0
    for batch in batches:
        rs = [r for r in records if r.batch_name == batch]
        if not rs:
            continue
        view = blind.select(blinded, rs)
        queries = [q for q in view.queries if pair is None or q.id == pair[0]]
        if not queries:
            continue
        held = {d.id: represent.index(rank.stated(
            d.id, d.text, represent.person, grounded, ladder)) for d in view.documents}
        by_id = {d.id: d for d in view.documents}
        for query in queries:
            required = rank.stated(query.id, query.text, represent.role,
                                   grounded, ladder, requiring=True)
            scored = {d: policy.coverage(required, held[d], ladder).lexicographic()
                      for d in by_id}
            ordering = order_of(scored)
            keep = ([pair[1]] if pair else ordering[:top])
            activity, activity_lines = ranking(
                batch, query.id, keep, len(scored), policy_iri, when)
            handle.write("\n".join(activity_lines) + "\n")
            for candidate in keep:
                position = ordering.index(candidate) + 1
                lines, _ = score(query.id, candidate, required, held[candidate],
                                 ladder, activity, position)
                handle.write("\n".join(lines) + "\n")
                written += 1
    return written


def _resolve(blinded, wanted):
    """A blind id or a corpus custom_id, as a blind id."""
    if wanted in blinded.source:
        return wanted
    found = [b for b, custom in blinded.source.items() if custom == wanted]
    if not found:
        raise SystemExit(f"no record with id {wanted!r}")
    return found[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--top", type=int, default=10,
                        help="record this many candidates per role (default 10)")
    parser.add_argument("--batch", choices=BATCHES, action="append",
                        help="restrict to a batch; repeatable")
    parser.add_argument("--pair", nargs=2, metavar=("ROLE", "CANDIDATE"),
                        help="record one pair, by blind id or corpus custom_id")
    parser.add_argument("--time", help="pin prov:startedAtTime, for a "
                                       "byte-identical rebuild")
    args = parser.parse_args()

    # Imported here so the module does not depend on the experiments package.
    from experiments import benchmark

    records = corpus.load_corpus()
    grounded = benchmark.grounding([r.profile_text for r in records], args.device)
    blinded = blind.blind(records)
    ladder = policy.CONCEPTS_ORDINAL
    when = args.time or datetime.now(timezone.utc).replace(
        microsecond=0).isoformat().replace("+00:00", "Z")
    pair = None
    if args.pair:
        pair = (_resolve(blinded, args.pair[0]), _resolve(blinded, args.pair[1]))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        written = write(handle, records, grounded, blinded, ladder, when,
                        top=args.top, batches=tuple(args.batch or BATCHES),
                        pair=pair)
    size = args.out.stat().st_size
    print(f"wrote {args.out}: {written:,} scored pairs, {size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
