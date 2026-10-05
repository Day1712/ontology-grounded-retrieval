"""Grounded items as RDF assertions about a role (requires) or a candidate
(holds), each carrying its source spans. Relations per branch come from
data/ontology/alignment.ttl.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import rdflib

from . import resource

SCHEMA = "https://schema.org/"
OA = "http://www.w3.org/ns/oa#"
PROV = "http://www.w3.org/ns/prov#"
XSD = "http://www.w3.org/2001/XMLSchema#"
#: This method's own namespace.
EXT = "https://example.org/onet-retrieval#"

ONET_CATEGORY = EXT + "onetCategory"

CNLO = "https://linkeddata.competentnl.nl/def/competentnl#"
OCCUPATION = CNLO + "Occupation"

#: CNL-O's graded subproperties of cnlo:requires (see :mod:`ranking_method.grade`).
REQUIRES_ESSENTIAL = CNLO + "requiresHATEssential"
REQUIRES_IMPORTANT = CNLO + "requiresHATImportant"
REQUIRES_SOMEWHAT = CNLO + "requiresHATSomewhat"
ABOUT = SCHEMA + "about"
#: A role's occupation; added by evaluation.labelled, never a ranking input.
OCCUPATIONAL_CATEGORY = SCHEMA + "occupationalCategory"
#: O*NET's own page for an occupation code.
ONET_OCCUPATION = "https://www.onetonline.org/link/summary/"
ALIGNMENT = Path(__file__).resolve().parent.parent / "data" / "ontology" / "alignment.ttl"
BRANCH_TYPE = EXT + "branchType"
BRANCH_RELATION = EXT + "branchRelation"
BRANCH_HOLDING = EXT + "branchHolding"


@lru_cache(maxsize=None)
def alignment(path=None):
    """``{branch: (type, requires, holds)}``, read from ``alignment.ttl``."""
    graph = rdflib.Graph()
    graph.parse(path or ALIGNMENT, format="turtle")
    by_iri = {e.iri: branch for branch, e in resource.elements().items()}
    out = {}
    for subject in set(graph.subjects(rdflib.URIRef(BRANCH_TYPE), None)):
        branch = by_iri.get(str(subject))
        if branch is None:
            raise ValueError(f"alignment names {subject}, which is not an O*NET element")
        out[branch] = (
            str(graph.value(subject, rdflib.URIRef(BRANCH_TYPE))),
            str(graph.value(subject, rdflib.URIRef(BRANCH_RELATION))),
            str(graph.value(subject, rdflib.URIRef(BRANCH_HOLDING))),
        )
    return out


@dataclass(frozen=True)
class Evidence:
    """A span of a source document. Start is included, end excluded."""

    source: str
    start: int
    end: int
    mechanism: str
    score: float | None


@dataclass(frozen=True)
class Assertion:
    subject: str
    predicate: str
    object: str
    evidence: tuple = ()


def person(item_id):
    return f"{EXT}person/{item_id}"


def document(item_id):
    return f"{EXT}document/{item_id}"


def role(item_id):
    """The role a job posting is about (distinct from the posting document)."""
    return f"{EXT}role/{item_id}"


def assertions(subject, source, items, requiring=False, grades=None):
    """One assertion per grounded item, with its evidence attached.

    ``requiring`` picks the branch rule's requires (role) or holds (candidate)
    relation; ``grades`` (from :func:`ranking_method.grade.read`) refines the
    requires relation. Ordinal items take ``onetCategory`` on both sides.
    """
    rules = alignment()
    out = []
    for item in items:
        if item.category is not None:
            predicate = ONET_CATEGORY
            target = resource.category_iri(item.element_id, item.category)
        else:
            branch = resource.branch(item.element_id)
            target = resource.elements()[item.element_id].iri
            predicate = rules[branch][1 if requiring else 2]
            if requiring and grades:
                predicate = grades.get(item.element_id, predicate)
        out.append(Assertion(subject, predicate, target, tuple(
            Evidence(source, s.start, s.end, s.mechanism, s.score)
            for s in item.support
        )))
    return tuple(out)


def index(items):
    """Assertions keyed by what a requirement has to match against: the O*NET
    resource, or ``(predicate, element)`` for an ordinal assertion."""
    out = {}
    for assertion in items:
        if assertion.predicate == ONET_CATEGORY:
            out[(assertion.predicate, resource.category_of(assertion.object)[0])] = assertion
        else:
            out[assertion.object] = assertion
    return out

