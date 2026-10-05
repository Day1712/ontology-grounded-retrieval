"""Judgment matrices from a gold's ``scorer(jd)``, and their mapping onto blinded ids."""

from collections.abc import Callable, Iterable

from .blind import Blinded
from .corpus import Record

Qrels = dict[str, dict[str, int]]
Scorer = Callable[[Record], Callable[[Record], float]]


def build_qrels(
    records: Iterable[Record],
    scorer: Scorer,
    query_type: str = "jd",
    doc_type: str = "cv",
) -> Qrels:
    """Judge every query against every document in ``records``."""
    records = list(records)
    documents = [r for r in records if r.profile_type == doc_type]
    out: Qrels = {}
    for query in (r for r in records if r.profile_type == query_type):
        score = scorer(query)
        # pytrec_eval requires integer grades; each gold scales before this.
        out[query.custom_id] = {d.custom_id: round(score(d)) for d in documents}
    return out


def blind_qrels(qrels: Qrels, blinded: Blinded) -> Qrels:
    """Restate judgments against the ranker's anonymous ids; values unchanged."""
    anonymous = {custom_id: item_id for item_id, custom_id in blinded.source.items()}
    return {
        anonymous[query]: {anonymous[doc]: v for doc, v in row.items()}
        for query, row in qrels.items()
    }
