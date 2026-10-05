"""Writes the assertions as an RDF graph (cache/graph.ttl), using blinded ids.

Also writes the blind-id to custom_id map that evaluation.labelled uses to join
the occupation back on.
"""

import argparse
import json
from pathlib import Path

import rdflib
from rdflib.namespace import RDF

from evaluation import blind, corpus
from experiments import benchmark

from . import grade, link, rank, represent, resource

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "cache" / "graph.ttl"
#: ``{blind id: custom_id}``, read by evaluation.labelled.
SOURCE_MAP = ROOT / "cache" / "graph_source.json"
OCCUPATION = rdflib.URIRef(represent.OCCUPATION)
OA = rdflib.Namespace(represent.OA)


def annotation_iri(item_id, body, start, end):
    """The IRI of one annotation, keyed by document, body and offsets.

    Shared with :mod:`ranking_method.record`, which points at these annotations.
    """
    return (f"{represent.EXT}annotation/{item_id}/"
            f"{body.rsplit('/', 1)[-1]}/{start}-{end}")


def annotation(out, body, evidence, item_id):
    """One OA annotation per supporting span: what was found, and where."""
    node = rdflib.URIRef(annotation_iri(item_id, body, evidence.start, evidence.end))
    target, selector = rdflib.BNode(), rdflib.BNode()
    out.add((node, RDF.type, OA.Annotation))
    out.add((node, OA.hasBody, rdflib.URIRef(body)))
    out.add((node, OA.hasTarget, target))
    out.add((target, OA.hasSource, rdflib.URIRef(evidence.source)))
    out.add((target, OA.hasSelector, selector))
    out.add((selector, RDF.type, OA.TextPositionSelector))
    out.add((selector, OA.start, rdflib.Literal(evidence.start)))
    out.add((selector, OA.end, rdflib.Literal(evidence.end)))


def graph(records, grounded, blinded=None):
    """Every record's assertions, each element object typed by its branch rule.

    Requirements carry the graded CNL-O relation where the posting states one.
    """
    rules = represent.alignment()
    by_iri = resource.by_iri()
    blinded = blinded or blind.blind(records)
    item_of = {custom_id: item for item, custom_id in blinded.source.items()}
    out = rdflib.Graph()
    for record in records:
        is_jd = record.profile_type == "jd"
        item_id = item_of[record.custom_id]
        subject = (represent.role if is_jd else represent.person)(item_id)
        document = represent.document(item_id)
        out.add((rdflib.URIRef(document), rdflib.URIRef(represent.ABOUT),
                 rdflib.URIRef(subject)))
        if is_jd:
            out.add((rdflib.URIRef(subject), RDF.type, OCCUPATION))
        items = grounded[rank.text_key(record.profile_text)]
        grades = grade.read(record.profile_text, items) if is_jd else None
        for assertion in represent.assertions(
                subject, document, items, requiring=is_jd, grades=grades):
            out.add((rdflib.URIRef(assertion.subject),
                     rdflib.URIRef(assertion.predicate),
                     rdflib.URIRef(assertion.object)))
            element = by_iri.get(assertion.object)
            if element is not None:
                out.add((rdflib.URIRef(assertion.object), RDF.type,
                         rdflib.URIRef(rules[resource.branch(element)][0])))
            for evidence in assertion.evidence:
                annotation(out, assertion.object, evidence, item_id)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=benchmark.DEVICE)
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--map", type=Path, default=SOURCE_MAP)
    args = parser.parse_args()

    records = corpus.load_corpus()
    grounded = benchmark.grounding([r.profile_text for r in records],
                                   device=args.device, vocab=link.vocabulary())
    blinded = blind.blind(records)
    written = graph(records, grounded, blinded)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    written.serialize(args.out, format="turtle")
    args.map.write_text(json.dumps(blinded.source, indent=1, sort_keys=True) + "\n")
    print(f"wrote {args.out}: {len(written):,} triples over {len(records):,} records")
    print(f"wrote {args.map}: {len(blinded.source):,} blind ids")


if __name__ == "__main__":
    main()
