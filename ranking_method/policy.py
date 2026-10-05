"""Satisfaction policy: which branches and scales are used, the ladder of
method variants, requirement weights, and coverage of a JD by a CV.
"""

from dataclasses import dataclass
from functools import lru_cache

from . import represent, resource

#: Content Model branches this method grounds.
CATEGORICAL_BRANCHES = ("1.A", "1.D", "2.A", "2.B", "2.C", "2.E", "4.A")

EXACT, RELATED, UNMATCHED = "exact", "related", "unmatched"
MET, SHORT, UNSTATED = "met", "falls short", "not stated"


@lru_cache(maxsize=None)
def ordinal_elements() -> tuple[str, ...]:
    """Elements whose published categories are read as ordered requirements.

    3.A.2 and 3.A.3 are excluded: their category wordings duplicate 3.A.1's.
    """
    return tuple(e for e in sorted(resource.categories()) if e in ("2.D.1", "3.A.1"))


@dataclass(frozen=True)
class Ladder:
    """Which rung: whether the ordered scales are stated, whether a sibling is
    reported as related evidence, and whether a requirement contributes by the
    CNL-O relation the posting stated it with."""

    ordinal: bool = True
    hierarchy: bool = False
    graded: bool = False
    #: Whether exceeding the required category satisfies it (False: identity only).
    exceeding: bool = True


#: Requirement weight by graded relation: the UWV relevance scores CNL-O cites.
WEIGHTS = {
    represent.REQUIRES_ESSENTIAL: 3.0,
    represent.REQUIRES_IMPORTANT: 2.0,
    represent.REQUIRES_SOMEWHAT: 1.0,
}

#: Weight of any requirement without a graded relation: the midpoint of 1-3.
UNGRADED_WEIGHT = 2.0

CONCEPTS = Ladder(ordinal=False)
CONCEPTS_ORDINAL_FLAT = Ladder()
CONCEPTS_GRADED = Ladder(graded=True)
#: The method with the ordered scales read by identity alone.
ORDINAL_IDENTITY = Ladder(graded=True, exceeding=False)
#: Flat structured matching: no grading, scales read by identity.
FLAT_IDENTITY = Ladder(exceeding=False)
#: The method without the ordered scales.
CONCEPTS_ONLY_GRADED = Ladder(ordinal=False, graded=True)
#: The method: concepts, the ordered scales, and graded requirement weights.
CONCEPTS_ORDINAL = CONCEPTS_GRADED
HIERARCHY = Ladder(hierarchy=True)


@dataclass(frozen=True)
class Verdict:
    """What became of one stated requirement. ``via`` (the sibling) and
    ``shared`` (their parent) are set only for ``related``."""

    status: str
    via: str | None = None
    shared: str | None = None


@dataclass(frozen=True)
class Coverage:
    """A CV against a JD: exact and related coverage, the number of stated
    requirements, and the total weight the coverages were taken over."""

    exact: float
    related: float
    stated: int
    total_weight: float = 0.0

    def lexicographic(self) -> float:
        """Exact coverage, with related coverage only breaking ties: the related
        term stays below one step of exact (1 / total_weight)."""
        if not self.stated:
            return 0.0
        return self.exact + self.related / (self.total_weight + 1)

    def additive(self, weight: float = 0.5) -> float:
        """Related counts as a fixed fraction of a requirement."""
        return self.exact + weight * self.related


def eligible_sibling(required: str, offered: str) -> bool:
    """Whether ``offered`` is a sibling of ``required`` under a shared parent
    that is not the top-level branch."""
    if required == offered:
        return False
    known = resource.elements()
    parent = known[required].parent
    return (parent is not None
            and parent == known[offered].parent
            and parent != resource.branch(required))


def assess(requirement, held, ladder=CONCEPTS_ORDINAL) -> Verdict:
    """The status of one stated requirement against the evidence.

    Categorical: identity of the O*NET resource. Ordinal: comparison of
    published category numbers (equality under ``exceeding=False``).
    """
    if requirement.predicate == represent.ONET_CATEGORY:
        element, required = resource.category_of(requirement.object)
        evidence = held.get((requirement.predicate, element))
        if evidence is None:
            return Verdict(UNSTATED)
        _, offered = resource.category_of(evidence.object)
        met = offered >= required if ladder.exceeding else offered == required
        return Verdict(MET if met else SHORT)

    if held.get(requirement.object) is not None:
        return Verdict(EXACT)
    if not ladder.hierarchy:
        return Verdict(UNMATCHED)

    known = resource.by_iri()
    required_id = known.get(requirement.object)
    if required_id is None:
        return Verdict(UNMATCHED)
    # Ordinal entries are keyed by a (predicate, element) tuple; categorical
    # ones by the resource. Only the latter can be a sibling.
    found = sorted(
        offered_id for target in held if not isinstance(target, tuple)
        for offered_id in (known.get(target),)
        if offered_id is not None and eligible_sibling(required_id, offered_id)
    )
    if not found:
        return Verdict(UNMATCHED)
    # Lowest sibling id, so the choice is reproducible.
    return Verdict(RELATED, found[0], resource.elements()[required_id].parent)


def explain(requirements, held, ladder=CONCEPTS_ORDINAL) -> tuple:
    """``((requirement, Verdict), ...)`` in stated order."""
    return tuple((r, assess(r, held, ladder)) for r in requirements)


def weight(requirement, ladder=CONCEPTS_ORDINAL) -> float:
    """What one requirement contributes to the denominator: 1 off the graded
    rung, else its :data:`WEIGHTS` entry or :data:`UNGRADED_WEIGHT`."""
    if not ladder.graded:
        return 1.0
    return WEIGHTS.get(requirement.predicate, UNGRADED_WEIGHT)


def coverage(requirements, held, ladder=CONCEPTS_ORDINAL) -> Coverage:
    """Weighted exact and related coverage of a JD's stated requirements."""
    stated = len(requirements)
    if not stated:
        return Coverage(0.0, 0.0, 0, 0.0)
    mass = {EXACT: 0.0, RELATED: 0.0, MET: 0.0}
    total = 0.0
    for requirement, verdict in explain(requirements, held, ladder):
        w = weight(requirement, ladder)
        total += w
        if verdict.status in mass:
            mass[verdict.status] += w
    return Coverage((mass[EXACT] + mass[MET]) / total,
                    mass[RELATED] / total, stated, total)
