"""Reads CompetentNL's graded requirement relations (essential, important,
somewhat) out of posting text, using the cues in data/ontology/grading.ttl.
Ambiguous cases keep the generic ``cnlo:requires``.
"""

import re
from functools import lru_cache
from pathlib import Path

import rdflib
from rdflib.namespace import RDFS

from . import represent, resource

GRADING = Path(__file__).resolve().parent.parent / "data" / "ontology" / "grading.ttl"
SCHEMA = Path(__file__).resolve().parent.parent / "data" / "ontology" / "cnlo.ttl"
CUE = represent.EXT + "cue"
HUMAN_ASPECT_TYPE = represent.CNLO + "HumanAspectType"

#: Same sentence pattern as ordinal.py.
SENTENCE = re.compile(r"[^.!?\n]+")


@lru_cache(maxsize=None)
def cues(path=None) -> tuple[tuple[str, str], ...]:
    """``((cue, relation), ...)``, longest cue first ("very important" before
    "important")."""
    graph = rdflib.Graph()
    graph.parse(path or GRADING, format="turtle")
    found = [(str(o).lower(), str(s))
             for s, o in graph.subject_objects(rdflib.URIRef(CUE))]
    if not found:
        raise ValueError(f"{path or GRADING} declares no cues")
    return tuple(sorted(found, key=lambda pair: (-len(pair[0]), pair[0])))


@lru_cache(maxsize=None)
def gradeable(schema=None, alignment=None) -> frozenset[str]:
    """Branches whose aligned class is ``cnlo:HumanAspectType`` or a subclass."""
    graph = rdflib.Graph()
    graph.parse(schema or SCHEMA, format="turtle")
    root = rdflib.URIRef(HUMAN_ASPECT_TYPE)
    under = {str(root)} | {
        str(s) for s in graph.transitive_subjects(RDFS.subClassOf, root)
    }
    rules = represent.alignment(alignment)
    return frozenset(b for b, rule in rules.items() if rule[0] in under)


def sentences(text: str) -> tuple[tuple[int, int, str], ...]:
    """``((start, end, text), ...)``. Offsets are into the text as given."""
    return tuple((m.start(), m.end(), m.group()) for m in SENTENCE.finditer(text))


def _stated(sentence: str) -> set[str]:
    """The relations a sentence's cues state; cues matched longest first,
    without overlap."""
    low, taken, found = sentence.lower(), [], set()
    for cue, relation in cues():
        for match in re.finditer(re.escape(cue), low):
            if any(match.start() < end and start < match.end()
                   for start, end in taken):
                continue
            taken.append((match.start(), match.end()))
            found.add(relation)
    return found


def read(text: str, items) -> dict[str, str]:
    """``{element id: relation}`` for the requirements a grade is stated for.

    Graded only where all support falls in one sentence, that sentence holds no
    other gradeable requirement, and it states exactly one relation.
    """
    if not text:
        return {}
    spans = sentences(text)
    if not spans:
        return {}
    branches, located = gradeable(), {}
    for item in items:
        if item.category is not None or not item.support:
            continue
        if resource.branch(item.element_id) not in branches:
            continue
        where = {n for n, (start, end, _) in enumerate(spans)
                 for s in item.support if start <= s.start < end}
        if where:
            located[item.element_id] = where
    # A requirement whose support crosses sentences occupies each of them.
    holding: dict[int, int] = {}
    for where in located.values():
        for n in where:
            holding[n] = holding.get(n, 0) + 1
    out = {}
    for element_id, where in located.items():
        if len(where) != 1:
            continue
        n = next(iter(where))
        if holding[n] != 1:
            continue
        found = _stated(spans[n][2])
        if len(found) == 1:
            out[element_id] = next(iter(found))
    return out
