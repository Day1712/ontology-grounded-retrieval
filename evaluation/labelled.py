"""The emitted graph with corpus identifiers and occupations joined back on.

Derived from ``graph.ttl`` and its blind-id map; never used for ranking.
"""

import argparse
import json
from pathlib import Path

import rdflib
from rdflib.namespace import RDF, RDFS

from ranking_method import emit, represent

from .corpus import load_corpus

OCCUPATIONAL_CATEGORY = rdflib.URIRef(represent.OCCUPATIONAL_CATEGORY)
OCCUPATION = rdflib.URIRef(represent.OCCUPATION)
#: The IRI stems a blind id can appear in.
STEMS = ("role/", "person/", "document/", "annotation/")


def source_map(path: Path = emit.SOURCE_MAP) -> dict[str, str]:
    """``{blind id: custom_id}``, as :mod:`ranking_method.emit` wrote it."""
    return json.loads(Path(path).read_text())


def _relabel(iri: str, mapping: dict[str, str]) -> str:
    """``iri`` with the blind id after its stem replaced, or unchanged if it carries none."""
    if not iri.startswith(represent.EXT):
        return iri
    rest = iri[len(represent.EXT):]
    for stem in STEMS:
        if not rest.startswith(stem):
            continue
        tail = rest[len(stem):]
        item_id, sep, remainder = tail.partition("/")
        custom_id = mapping.get(item_id)
        if custom_id is None:
            return iri
        return f"{represent.EXT}{stem}{custom_id}{sep}{remainder}"
    return iri


def labelled(graph_path: Path = emit.RESULTS, map_path: Path = emit.SOURCE_MAP,
             records=None) -> rdflib.Graph:
    """``graph.ttl`` with corpus identifiers restored and occupations from ``records`` (default: corpus)."""
    mapping = source_map(map_path)
    occ = {r.custom_id: r.occ_code for r in (records or load_corpus())}
    source = rdflib.Graph()
    source.parse(graph_path, format="turtle")

    out = rdflib.Graph()
    for prefix, namespace in source.namespaces():
        out.bind(prefix, namespace)
    for s, p, o in source:
        out.add((rdflib.URIRef(_relabel(str(s), mapping)) if isinstance(s, rdflib.URIRef) else s,
                 p,
                 rdflib.URIRef(_relabel(str(o), mapping)) if isinstance(o, rdflib.URIRef) else o))

    for subject in set(out.subjects(RDF.type, OCCUPATION)):
        custom_id = str(subject)[len(represent.EXT) + len("role/"):]
        code = occ.get(custom_id)
        if code is None:
            continue
        out.add((subject, OCCUPATIONAL_CATEGORY, rdflib.Literal(code)))
        out.add((subject, RDFS.seeAlso,
                 rdflib.URIRef(represent.ONET_OCCUPATION + code)))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, default=emit.RESULTS)
    parser.add_argument("--map", type=Path, default=emit.SOURCE_MAP)
    parser.add_argument("--out", type=Path,
                        default=emit.RESULTS.with_name("graph_labelled.ttl"))
    args = parser.parse_args()

    out = labelled(args.graph, args.map)
    out.serialize(args.out, format="turtle")
    print(f"wrote {args.out}: {len(out):,} triples")


if __name__ == "__main__":
    main()
