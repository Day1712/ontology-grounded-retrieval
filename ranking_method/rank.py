"""Ranking by requirement satisfaction: grounds JDs and CVs, states them as
assertions and scores each pair with :func:`ranking_method.policy.coverage`.
Nothing is sorted here; :mod:`evaluation.evaluate` does the ranking.
"""

import hashlib
from collections.abc import Sequence

from . import grade, link, policy, represent


def text_key(text: str) -> str:
    """Cache key: SHA-256 of the text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ground(texts: Sequence[str], vocab=None, enc=None) -> dict[str, tuple]:
    """The grounded items of every text, keyed by :func:`text_key`."""
    items = link.annotate(list(texts), vocab=vocab, enc=enc)
    return {text_key(t): i for t, i in zip(texts, items)}


def stated(item_id, text, subject, grounded, ladder, requiring=False):
    """The assertions one document states under ``ladder``: requirements when
    ``requiring``, evidence otherwise. Shared with :mod:`ranking_method.record`."""
    items = grounded[text_key(text)]
    grades = grade.read(text, items) if (requiring and ladder.graded) else None
    found = represent.assertions(
        subject(item_id), represent.document(item_id), items,
        requiring=requiring, grades=grades,
    )
    if ladder.ordinal:
        return found
    return tuple(a for a in found if a.predicate != represent.ONET_CATEGORY)


def rank(queries, documents, grounded=None, vocab=None, enc=None,
         ladder=policy.CONCEPTS_ORDINAL, key=None):
    """``{query id: {doc id: Coverage}}``, or one number per pair when ``key``
    names a scoring (see :func:`scores`)."""
    if grounded is None:
        grounded = ground([x.text for x in list(queries) + list(documents)], vocab, enc)

    def side(item_id, text, subject, requiring=False):
        """This run's grounding and rung, bound to :func:`stated`."""
        return stated(item_id, text, subject, grounded, ladder, requiring)

    held = [
        (d.id, represent.index(side(d.id, d.text, represent.person)))
        for d in documents
    ]
    out = {}
    for query in queries:
        required = side(query.id, query.text, represent.role, requiring=True)
        out[query.id] = {
            doc_id: policy.coverage(required, evidence, ladder)
            for doc_id, evidence in held
        }
    return out if key is None else scores(out, key)


LEXICOGRAPHIC = "lexicographic"


def scores(coverages, key=LEXICOGRAPHIC):
    """One number per pair: ``lexicographic`` (related only breaks ties) or,
    for any other key, ``additive`` (related counts half)."""
    pick = (lambda c: c.lexicographic()) if key == LEXICOGRAPHIC else \
        (lambda c: c.additive())
    return {q: {d: pick(c) for d, c in row.items()} for q, row in coverages.items()}
