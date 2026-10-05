"""Gold: how many of a JD's requirements the CV satisfies, unweighted.

    rel = |C_jd & C_cv| + education_met + experience_met
"""

from .corpus import Record

CATEGORICAL = frozenset(
    {"skill", "knowledge", "abilities", "task", "tech_skill", "work_style"}
)
EDUCATION = "required education"
EXPERIENCE = "related experience"
ABSENT = "<NA>"  # a null serialised as a level name, not a level


def _categorical_ids(record: Record) -> set[str]:
    return {k for k, v in record.profile_dict.items() if v.get("kind") in CATEGORICAL}


def _level(record: Record, kind: str) -> int | None:
    """Ordinal level from the descriptor id's trailing integer, or None where
    the record states no level for this kind."""
    for descriptor_id, value in record.profile_dict.items():
        if value.get("kind") == kind:
            if value.get("name") == ABSENT:
                return None
            return int(descriptor_id.rsplit(".", 1)[1])
    return None


def _threshold_met(jd: Record, cv: Record, kind: str) -> int:
    """1 when both sides state a level and the CV's is at least the JD's."""
    required = _level(jd, kind)
    offered = _level(cv, kind)
    return int(required is not None and offered is not None and offered >= required)


def categorical_overlap(jd: Record, cv: Record) -> int:
    return len(_categorical_ids(jd) & _categorical_ids(cv))


def education_met(jd: Record, cv: Record) -> int:
    return _threshold_met(jd, cv, EDUCATION)


def experience_met(jd: Record, cv: Record) -> int:
    return _threshold_met(jd, cv, EXPERIENCE)


def relevance(jd: Record, cv: Record) -> int:
    return categorical_overlap(jd, cv) + education_met(jd, cv) + experience_met(jd, cv)


def scorer(jd: Record):
    return lambda cv: relevance(jd, cv)
