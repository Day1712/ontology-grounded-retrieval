"""Lexical grounding: published surface forms located in free text."""

import re
from collections import defaultdict
from dataclasses import dataclass

TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True)
class Occurrence:
    concept: str
    start: int
    end: int
    form: str


def tokens(text):
    return [(m.group(), m.start(), m.end()) for m in TOKEN.finditer(text)]


def index(vocabulary):
    """{token tuple: [(target, case significant, form)]}, {head token: width}.

    ``vocabulary`` maps a target identifier to its (form, case significant) pairs.
    """
    idx = defaultdict(list)
    width = defaultdict(int)
    for target, forms in vocabulary.items():
        for form, strict in forms:
            key = tuple(t.casefold() for t, _, _ in tokens(form))
            if key:
                idx[key].append((target, strict, form))
                width[key[0]] = max(width[key[0]], len(key))
    return idx, width


def ground(text, idx, width):
    """Every occurrence of every form, nested matches included."""
    toks = tokens(text)
    fold = [t.casefold() for t, _, _ in toks]
    out = []
    for i, head in enumerate(fold):
        for n in range(1, min(width.get(head, 0), len(toks) - i) + 1):
            for concept, strict, form in idx.get(tuple(fold[i : i + n]), ()):
                start, end = toks[i][1], toks[i + n - 1][2]
                if strict and text[start:end] != form:
                    continue
                out.append(Occurrence(concept, start, end, form))
    return out
