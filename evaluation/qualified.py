"""Gold: stated minimums first, then importance-weighted fit (lexicographic).

    rel   = (education_met + experience_met) * sep_q
            + sum of w over satisfied categorical requirements
    sep_q = sum of that JD's weights + 1, above any achievable weighted sum

A CV that does not state its education or experience counts as not meeting it.
"""

from .corpus import Record
from .count import _categorical_ids, education_met, experience_met
from .importance import SCALE, weights


def _separator(w: dict[str, float]) -> float:
    """A value strictly above any weighted sum this JD can produce."""
    return sum(w.values()) + 1.0


def qualification(jd: Record, cv: Record) -> int:
    """How many of the JD's two stated minimums the CV meets: 0, 1 or 2."""
    return education_met(jd, cv) + experience_met(jd, cv)


def _graded(jd: Record, cv: Record, w: dict[str, float]) -> float:
    return sum(w[i] for i in _categorical_ids(jd) & _categorical_ids(cv))


def scorer(jd: Record):
    w = weights(jd)
    sep = _separator(w)
    return lambda cv: (qualification(jd, cv) * sep + _graded(jd, cv, w)) * SCALE
