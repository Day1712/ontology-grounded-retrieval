"""Closed-world domain/range check on an emitted graph (thesis: RDF graph conformance).

Every object of a relation whose declared range is a CNL-O or Core-O class must carry
an explicitly emitted type under that range; the relations checked are read from the schemas.
"""

import argparse
import json
from pathlib import Path

import rdflib
from rdflib.namespace import RDF, RDFS

from ranking_method import represent, resource

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "cache" / "conformance.json"
SCHEMAS = ("cnlo.ttl", "coreo.ttl", "ext.ttl")
CNLO = rdflib.Namespace(represent.CNLO)
COREO = rdflib.Namespace("http://purl.org/coreo#")
EXT = rdflib.Namespace(represent.EXT)
SCHEMA = rdflib.Namespace(represent.SCHEMA)

#: Where a domain or a range may be declared: RDFS, or Schema.org terms in ext.ttl.
DOMAINS = (RDFS.domain, SCHEMA.domainIncludes)
RANGES = (RDFS.range, SCHEMA.rangeIncludes)
#: A range this check can test: a class the emitted graph types elements with.
CHECKABLE = (str(CNLO), str(COREO))


def schema_graph(root=ROOT):
    g = rdflib.Graph()
    for name in SCHEMAS:
        g.parse(root / "data" / "ontology" / name, format="turtle")
    return g


def _under(schema, cls, root):
    """True where ``cls`` is ``root`` or a declared subclass of it."""
    if cls == root:
        return True
    return root in set(schema.transitive_objects(cls, RDFS.subClassOf))


def claims(schema, graph):
    """``((relation, ranges, subject must be a role), ...)`` for every relation the
    graph uses whose declared range is a CNL-O or Core-O class."""
    out = []
    for relation in sorted(set(graph.predicates()), key=str):
        ranges = {o for p in RANGES for o in schema.objects(relation, p)
                  if isinstance(o, rdflib.URIRef)}
        wanted = {r for r in ranges if str(r).startswith(CHECKABLE)}
        if not wanted:
            continue
        domains = {o for p in DOMAINS for o in schema.objects(relation, p)}
        out.append((relation, frozenset(wanted), CNLO.Occupation in domains))
    return tuple(out)


def unconstrained(schema, graph):
    """Relations the graph uses that no checkable range constrains."""
    checked = {r for r, _, _ in claims(schema, graph)}
    return tuple(sorted((str(p) for p in set(graph.predicates()) if p not in checked)))


def violations(graph, schema):
    """Every emitted assertion that fails a claim."""
    by_iri = resource.by_iri()
    out = []
    for relation, wanted, is_role in claims(schema, graph):
        for subject, _, obj in graph.triples((None, relation, None)):
            if is_role and (subject, RDF.type, CNLO.Occupation) not in graph:
                out.append((subject, "subject is not typed cnlo:Occupation"))
            if str(obj) not in by_iri:
                out.append((obj, "not a published O*NET element"))
                continue
            types = list(graph.objects(obj, RDF.type))
            if not types:
                out.append((obj, f"{relation}: no explicitly emitted type"))
                continue
            if not any(_under(schema, t, w) for t in types for w in wanted):
                out.append((obj, f"typed {[str(t) for t in types]}, "
                                f"not under {sorted(str(w) for w in wanted)}"))
    return out


#: (relation, declared range, flipped range) for the falsification run.
FALSIFY = (EXT.requiresTechnology, COREO.ResourceType, CNLO.HumanAspectType)


def falsification(graph, schema):
    """Violations the same graph shows once one declared range is flipped (in memory)."""
    relation, declared, flipped = FALSIFY
    altered = rdflib.Graph()
    for triple in schema:
        altered.add(triple)
    altered.remove((relation, SCHEMA.rangeIncludes, declared))
    altered.add((relation, SCHEMA.rangeIncludes, flipped))
    bad = violations(graph, altered)
    return {
        "relation": str(relation),
        "declared_range": str(declared),
        "flipped_to": str(flipped),
        "violations": len(bad),
        "distinct_objects": len({str(s) for s, _ in bad}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("turtle", type=Path, nargs="+",
                        help="emitted graphs, checked as their union")
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()
    graph = rdflib.Graph()
    for path in args.turtle:
        graph.parse(path, format="turtle")
    schema = schema_graph()
    tested = claims(schema, graph)
    skipped = unconstrained(schema, graph)
    bad = violations(graph, schema)
    print(f"{len(graph)} triples checked against {len(schema)} schema triples")
    print(f"{len(tested)} relations constrained by a CNL-O or Core-O range:")
    for relation, wanted, is_role in tested:
        short = str(relation).split("#")[-1].split("/")[-1]
        print(f"  {short:22s} -> {sorted(str(w).split('#')[-1] for w in wanted)}"
              f"{'  (role)' if is_role else ''}")
    print(f"{len(skipped)} used but unconstrained, so untested here:")
    for relation in skipped:
        print(f"  {relation}")
    for subject, why in bad:
        print(f"  FAIL {subject} -- {why}")
    flipped = falsification(graph, schema)
    print(f"falsification: with {flipped['relation'].split('#')[-1]} ranging over "
          f"{flipped['flipped_to'].split('#')[-1]} instead of "
          f"{flipped['declared_range'].split('#')[-1]}, the same graph fails "
          f"{flipped['violations']:,} assertions over {flipped['distinct_objects']} objects")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "graphs": [str(p) for p in args.turtle],
        "triples": len(graph),
        "schema_triples": len(schema),
        "claims": [[str(r), sorted(str(w) for w in wanted), is_role]
                   for r, wanted, is_role in tested],
        "unconstrained": list(skipped),
        "violations": [[str(s), why] for s, why in bad],
        "falsification": flipped,
    }, indent=1) + "\n")
    print(f"wrote {args.out}")
    print("conforms" if not bad else f"{len(bad)} violations")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
