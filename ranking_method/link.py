"""Text to O*NET entities: lexical matching of published surface forms plus
semantic matching of sentences against element descriptions (e5), with the
ordered scales read by :mod:`ranking_method.ordinal`.
"""

import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from . import ground, ordinal, policy, resource

MODEL = "intfloat/e5-base-v2"
SENTENCE = re.compile(r"[^.!?\n]+[.!?]*")
SEMANTIC_THRESHOLD = 0.830  # a sentence's best element is accepted
LEXICAL_SUPPORT_THRESHOLD = float("-inf")  # off: form matches are not gated
LEXICAL, SEMANTIC = "lexical", "semantic"

#: Branches whose single-word element names are ordinary trait words
#: ("initiative", "stamina"), so they are matched case-insensitively.
ATTRIBUTE_BRANCHES = ("1.A", "1.D")


@dataclass(frozen=True)
class Support:
    """One reason a text was taken to express an element.

    ``score`` is the sentence-element similarity (None for an ordinal reading);
    ``form`` is the matched surface form (lexical matches only).
    """

    element_id: str
    start: int
    end: int
    mechanism: str
    score: float | None
    form: str | None


@dataclass(frozen=True)
class GroundedItem:
    """One thing a document says, located in O*NET. ``category`` is set only
    for a reading on an ordered scale."""

    element_id: str
    category: int | None = None
    support: tuple = ()


def groundable(branch):
    """The deepest elements under ``branch`` that publish a name and a description."""
    known = resource.elements()
    out = set()
    for leaf in resource.leaves(branch):
        node = leaf
        while node is not None and node != branch and not (
                known[node].name and known[node].description):
            node = known[node].parent
        if node is not None and node != branch:
            out.add(node)
    return tuple(sorted(out))


def vocabulary(branches=policy.CATEGORICAL_BRANCHES, only=None, frontier=None):
    """``{element id: ((form, case significant), ...)}`` for groundable elements.

    Forms are the element name, its workplace examples and its mapped task and
    activity statements. Single-word forms are case-sensitive except in
    :data:`ATTRIBUTE_BRANCHES`. ``frontier`` defaults to :func:`groundable`.
    """
    examples = resource.workplace_examples()
    activities = resource.activity_forms()
    known = resource.elements()
    frontier = frontier or groundable
    out = {}
    for branch in branches:
        for element_id in frontier(branch):
            if only is not None and element_id not in only:
                continue
            name = known[element_id].name
            forms = [(name, len(name.split()) == 1
                      and branch not in ATTRIBUTE_BRANCHES)]
            forms += [(x, len(x.split()) == 1) for x in examples.get(element_id, ())]
            forms += [(x, False) for x in activities.get(element_id, ())]
            out[element_id] = tuple(forms)
    return out


def sentences(text):
    return [
        (m.group(), m.start(), m.end())
        for m in SENTENCE.finditer(text)
        if m.group().strip()
    ]


def encoder(name=MODEL, device=None):
    return SentenceTransformer(name, device=device)


def encode_elements(vocabulary, enc):
    """e5 wants its two sides marked; elements are the passages."""
    known = resource.elements()
    ids = sorted(vocabulary)
    texts = [f"passage: {known[i].name}. {known[i].description}" for i in ids]
    return ids, enc.encode(texts, normalize_embeddings=True, batch_size=64)


def scores(texts, matrix, enc, batch_size=128):
    """Per text, its sentence spans and their similarity to every element."""
    spans = [sentences(t) for t in texts]
    flat = [f"query: {s}" for doc in spans for s, _, _ in doc]
    sim = enc.encode(flat, normalize_embeddings=True, batch_size=batch_size,
                     show_progress_bar=False) @ matrix.T
    out, at = [], 0
    for doc in spans:
        out.append((doc, sim[at : at + len(doc)]))
        at += len(doc)
    return out


def linked(doc, sim, ids, threshold=SEMANTIC_THRESHOLD, k=1):
    """The best ``k`` elements per sentence, where the score clears the threshold."""
    best = {}
    for row, (_, start, end) in zip(sim, doc):
        top = [int(np.argmax(row))] if k == 1 else np.argpartition(row, -k)[-k:]
        for j in top:
            j = int(j)
            if row[j] >= threshold and row[j] > best.get(ids[j], (0.0,))[0]:
                best[ids[j]] = (float(row[j]), start, end)
    return best


def validated(occurrences, doc, sim, ids, threshold=LEXICAL_SUPPORT_THRESHOLD):
    """Form matches, carrying their sentence's score against the element named."""
    column = {target: j for j, target in enumerate(ids)}
    keep = []
    for o in occurrences:
        for row, (_, start, end) in zip(sim, doc):
            if start <= o.start < end and row[column[o.concept]] >= threshold:
                keep.append((o, float(row[column[o.concept]])))
                break
    return keep


def annotate(texts, vocab=None, enc=None,
             semantic_threshold=SEMANTIC_THRESHOLD,
             lexical_threshold=LEXICAL_SUPPORT_THRESHOLD,
             ordinal_elements=None, scored=None, k=1):
    """The grounded items each text expresses.

    ``scored`` supplies precomputed sentence similarities (element order
    ``sorted(vocab)``), so thresholds can be varied without re-encoding.
    """
    ordinal_elements = policy.ordinal_elements() if ordinal_elements is None \
        else ordinal_elements
    vocab = vocab if vocab is not None else vocabulary()
    if scored is None:
        enc = enc or encoder()
        ids, matrix = encode_elements(vocab, enc)
        scored = scores(texts, matrix, enc)
    else:
        ids = sorted(vocab)
    idx, width = ground.index(vocab)
    out = []
    for text, (doc, sim) in zip(texts, scored):
        support = [
            Support(o.concept, o.start, o.end, LEXICAL, score, o.form)
            for o, score in validated(ground.ground(text, idx, width), doc, sim,
                                      ids, lexical_threshold)
        ]
        support += [
            Support(element_id, start, end, SEMANTIC, score, None)
            for element_id, (score, start, end)
            in linked(doc, sim, ids, semantic_threshold, k).items()
        ]
        grouped = defaultdict(list)
        for s in support:
            grouped[s.element_id].append(s)
        items = [GroundedItem(e, None, tuple(g)) for e, g in grouped.items()]
        for element_id in ordinal_elements:
            found = ordinal.reading(element_id, text)
            if found is not None:
                items.append(GroundedItem(element_id, found.category, tuple(
                    Support(element_id, start, end, found.mechanism, None, None)
                    for start, end in found.spans
                )))
        out.append(tuple(items))
    return out
