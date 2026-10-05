"""Gold: requirement satisfaction weighted by O*NET importance.

    w   = I / mu_q  where O*NET publishes an importance, else 1
    rel = sum of w over satisfied categorical requirements
          + education_met + experience_met

mu_q is the mean importance over that JD's rated requirements.
"""

from .corpus import Record
from .count import CATEGORICAL, _categorical_ids, education_met, experience_met

#: pytrec_eval requires integer grades.
SCALE = 10_000


def weights(jd: Record) -> dict[str, float]:
    """Descriptor id -> weight, over the JD's categorical requirements."""
    scored = [
        v["importance"]
        for v in jd.profile_dict.values()
        if v.get("kind") in CATEGORICAL and "importance" in v
    ]
    # A JD whose importances are all zero gets rated weights 0, unrated 1.
    mu = sum(scored) / len(scored) if scored else 0.0
    return {
        descriptor: (value["importance"] / mu if mu else 0.0)
        if "importance" in value
        else 1.0
        for descriptor, value in jd.profile_dict.items()
        if value.get("kind") in CATEGORICAL
    }


def _satisfied(jd: Record, cv: Record, w: dict[str, float]) -> float:
    shared = _categorical_ids(jd) & _categorical_ids(cv)
    return sum(w[i] for i in shared) + education_met(jd, cv) + experience_met(jd, cv)


def scorer(jd: Record):
    w = weights(jd)
    return lambda cv: _satisfied(jd, cv, w) * SCALE
