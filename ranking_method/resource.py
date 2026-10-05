"""O*NET as the source of truth: elements, hierarchy, workplace examples, task
mappings and the ordered requirement categories, read from the published files.
"""

import csv
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import rdflib
from rdflib import Namespace

ONET = Namespace("https://www.onetcenter.org/rdf/schema/onet/")
RDF_DIR = Path(__file__).resolve().parent.parent / "data" / "onet_rdf"


@dataclass(frozen=True)
class Element:
    """A published O*NET Content Model element."""

    iri: str
    element_id: str
    name: str
    description: str
    parent: str | None  # element_id of its memberOf target


@lru_cache(maxsize=None)
def release() -> str:
    """The release these files were distributed as."""
    head = (RDF_DIR / "RELEASE.txt").read_text(encoding="utf-8").splitlines()
    return next(line.strip() for line in head if line.strip().startswith("O*NET"))


@lru_cache(maxsize=None)
def _graph(name):
    g = rdflib.Graph()
    g.parse(RDF_DIR / name, format="turtle")
    return g


@lru_cache(maxsize=None)
def elements() -> dict[str, Element]:
    """Every published Element, keyed by its elementID."""
    g = _graph("Element.ttl")
    by_iri = {s: str(o) for s, o in g.subject_objects(ONET.elementID)}
    return {
        element_id: Element(
            iri=str(subject),
            element_id=element_id,
            name=str(g.value(subject, ONET.elementName)),
            description=str(g.value(subject, ONET.description)),
            parent=by_iri.get(g.value(subject, ONET.memberOf)),
        )
        for subject, element_id in by_iri.items()
    }


@lru_cache(maxsize=None)
def _children() -> dict[str, tuple[str, ...]]:
    kids = defaultdict(list)
    for element_id, element in elements().items():
        if element.parent is not None:
            kids[element.parent].append(element_id)
    return {k: tuple(sorted(v)) for k, v in kids.items()}


def branch(element_id: str) -> str | None:
    """The top-level branch an element sits under: the ancestor whose own parent
    is a root. Followed through memberOf; None for a root itself."""
    known = elements()
    element = known[element_id]
    if element.parent is None:
        return None
    while known[element.parent].parent is not None:
        element = known[element.parent]
    return element.element_id


def leaves(root_id: str) -> tuple[str, ...]:
    """Element ids under ``root_id`` that have no children of their own."""
    kids = _children()
    stack, found = [root_id], []
    while stack:
        element_id = stack.pop()
        if element_id in kids:
            stack.extend(kids[element_id])
        else:
            found.append(element_id)
    return tuple(sorted(found))


@lru_cache(maxsize=None)
def by_iri() -> dict[str, str]:
    """Element IRI -> element id."""
    return {element.iri: element_id for element_id, element in elements().items()}


@lru_cache(maxsize=None)
def workplace_examples() -> dict[str, tuple[str, ...]]:
    """Element id -> the workplace examples published for it, via exampleFor."""
    g = _graph("SoftwareWorkplaceExample.ttl")
    by_iri = {element.iri: element_id for element_id, element in elements().items()}
    found = defaultdict(set)
    for subject, target in g.subject_objects(ONET.exampleFor):
        found[by_iri[str(target)]].add(str(g.value(subject, ONET.workplaceExample)))
    return {k: tuple(sorted(v)) for k, v in found.items()}


@lru_cache(maxsize=None)
def activity_forms() -> dict[str, tuple[str, ...]]:
    """Generalized Work Activity id -> the task statements and Detailed Work
    Activity names O*NET maps onto it (read from the tab-separated files)."""
    root = RDF_DIR.parent / "onet"

    def rows(name):
        with (root / name).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    to_gwa, name_of = {}, {}
    for row in rows("GWAs to IWAs to DWAs.txt"):
        to_gwa[row["DWA Element ID"]] = row["GWA Element ID"]
        name_of[row["DWA Element ID"]] = row["DWA Element Name"]
    task = {row["Task ID"]: row["Task"] for row in rows("Task Statements.txt")}
    found = defaultdict(set)
    for dwa, gwa in to_gwa.items():
        if name_of.get(dwa):
            found[gwa].add(name_of[dwa])
    for row in rows("Tasks to DWAs.txt"):
        gwa = to_gwa.get(row["DWA Element ID"])
        if gwa and task.get(row["Task ID"]):
            found[gwa].add(task[row["Task ID"]])
    return {k: tuple(sorted(v)) for k, v in found.items()}


@lru_cache(maxsize=None)
def category_resources() -> dict[str, tuple[str, int, str]]:
    """Category IRI -> (element id, category number, published description).

    Only the ``refersTo`` target that is an Element is kept (not the Scale).
    """
    by_iri = {element.iri: element_id for element_id, element in elements().items()}
    found = {}
    for path in sorted(RDF_DIR.glob("*Category.ttl")):
        g = _graph(path.name)
        for subject, category in g.subject_objects(ONET.category):
            description = str(g.value(subject, ONET.categoryDescription))
            for target in g.objects(subject, ONET.refersTo):
                if str(target) in by_iri:
                    found[str(subject)] = (by_iri[str(target)], int(category), description)
    return found


@lru_cache(maxsize=None)
def categories() -> dict[str, dict[int, str]]:
    """Element id -> {category: its published description}, in category order."""
    found = defaultdict(dict)
    for element_id, number, description in category_resources().values():
        found[element_id][number] = description
    return {k: dict(sorted(v.items())) for k, v in found.items()}


def category_of(iri: str) -> tuple[str, int]:
    """The element and category number a category resource stands for."""
    element_id, number, _ = category_resources()[iri]
    return element_id, number


@lru_cache(maxsize=None)
def category_iri(element_id: str, number: int) -> str:
    """The published resource for one category of one element."""
    return next(
        iri for iri, (e, n, _) in category_resources().items()
        if e == element_id and n == number
    )
