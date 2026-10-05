"""Reads O*NET's ordered requirement categories (education, experience) out of
free text: by published category wording, or else by a stated duration that
falls in a category's published bounds.
"""

import re
from dataclasses import dataclass
from functools import lru_cache

from . import resource

MONTHS = {"month": 1, "year": 12}
DURATION = re.compile(r"(\d+) (month|year)s?")
TOKEN = re.compile(r"[A-Za-z0-9']+")
SENTENCE = re.compile(r"[^.!?\n]+")
WORDS = {
    w: n
    for n, w in enumerate(
        "one two three four five six seven eight nine ten eleven twelve".split(), 1
    )
}
DIGITS = {w: str(n) for w, n in WORDS.items()}
#: Ignored when matching published wording ("over four years and up to ...").
IGNORED = frozenset({"and"})

STATED, MEASURED = "stated", "measured"


@dataclass(frozen=True)
class Reading:
    """A category read from a text, and where it was read from."""

    category: int
    mechanism: str
    spans: tuple  # (start, end) pairs, end excluded


def located(text):
    """(token, start, end) for every comparable token, normalised as in
    :func:`tokens` but keeping offsets into the text."""
    out = []
    for match in TOKEN.finditer(text):
        token = match.group().lower()
        token = DIGITS.get(token, token)
        if token not in IGNORED:
            out.append((token, match.start(), match.end()))
    return tuple(out)


def tokens(text):
    """Comparable tokens: lowercased, number words as digits, IGNORED dropped."""
    return tuple(token for token, _, _ in located(text))


def _find(haystack, needle):
    """Where ``needle`` occurs in ``haystack``, or None."""
    n = len(needle)
    if n == 0:
        return None
    for i in range(len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return None


def _contains(haystack, needle):
    return _find(haystack, needle) is not None


def _head(description):
    """The category's name, before any elaboration O*NET appends."""
    return re.split(r" - | \(", description)[0]


@lru_cache(maxsize=None)
def wordings(element_id):
    """{category: tokens of its published name}, for categories that name
    something locatable in prose. A category described as "None" states the
    absence of the thing and has no wording to find."""
    return {
        category: tokens(_head(description))
        for category, description in resource.categories()[element_id].items()
        if description.strip().lower() != "none"
    }


@lru_cache(maxsize=None)
def bands(element_id):
    """{category: (low, high) in months}, where the categories partition a
    duration. Empty for a scale that does not, which is how education gets no
    duration reading rather than by being named here."""
    found = {}
    for category, description in resource.categories()[element_id].items():
        head = _head(description)
        spans = [int(n) * MONTHS[u] for n, u in DURATION.findall(" ".join(tokens(head)))]
        if not spans:
            if head.strip().lower() == "none":
                found[category] = (0, 0)
            continue
        # Lower bounds are exclusive ("Over 2 years"); uppers are inclusive.
        if len(spans) == 1:
            found[category] = (0, spans[0]) if head.lower().startswith("up to") \
                else (spans[0], None)
        else:
            found[category] = (spans[0], spans[1])
    return found if len(found) == len(resource.categories()[element_id]) else {}


def _durations(marks):
    """(months, span) for every digit token followed by a month/year token."""
    out = []
    for i in range(len(marks) - 1):
        value, unit = marks[i][0], marks[i + 1][0]
        if not value.isdigit():
            continue
        for name in MONTHS:
            if unit.startswith(name):
                out.append((int(value) * MONTHS[name], (marks[i][1], marks[i + 1][2])))
                break
    return out


def _stated(element_id, text):
    """(category, spans) for the highest category whose wording the text states.

    Wording contained in another matched wording is dropped first, so "less than
    a high school diploma" is not also read as "high school diploma".
    """
    marks = located(text)
    found = tuple(token for token, _, _ in marks)
    hits = {c: w for c, w in wordings(element_id).items() if _contains(found, w)}
    hits = {
        c: w
        for c, w in hits.items()
        if not any(_contains(other, w) and other != w for other in hits.values())
    }
    if not hits:
        return None
    category = max(hits)
    wording = hits[category]
    at = _find(found, wording)
    return category, ((marks[at][1], marks[at + len(wording) - 1][2]),)


def _measured(element_id, text):
    """(category, spans) for the longest duration stated about this element.

    A sentence is about the element when a word in it starts with the last word
    of the element's name ("experience"); the trigger and the quantity are
    returned as separate spans.
    """
    scale = bands(element_id)
    if not scale:
        return None
    trigger = tokens(resource.elements()[element_id].name)[-1]
    marks = located(text)
    best = None
    for sentence in SENTENCE.finditer(text):
        run = tuple(m for m in marks if sentence.start() <= m[1] < sentence.end())
        found = next((m for m in run if m[0].startswith(trigger)), None)
        if found is None:
            continue
        for months, span in _durations(run):
            if best is None or months > best[0]:
                best = (months, (found[1], found[2]), span)
    if best is None:
        return None
    value, trigger_span, quantity_span = best
    for category, (_, high) in sorted(scale.items()):
        if high is None or value <= high:
            return category, (trigger_span, quantity_span)
    return None


def reading(element_id, text):
    """The category this text states for this element, with its spans."""
    found = _stated(element_id, text)
    if found is not None:
        return Reading(found[0], STATED, found[1])
    found = _measured(element_id, text)
    if found is not None:
        return Reading(found[0], MEASURED, found[1])
    return None


def level(element_id, text):
    """The category this text states for this element, if any."""
    found = reading(element_id, text)
    return found.category if found is not None else None
