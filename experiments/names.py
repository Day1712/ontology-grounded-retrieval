"""Shared helpers for the name-replacement runs: stimulus sets, the seeded pool
names, focal CVs, closed-form ranks and the BM25 control. Every pool CV carries a
name, so BM25 sees the same vocabulary under every stimulus."""

import csv
import hashlib
import os
import random
from pathlib import Path

import numpy as np

from evaluation import blind
from evaluation.baselines import bm25

ROOT = Path(__file__).resolve().parent.parent
NAMES_CSV = ROOT / "data" / "names" / "wilson_caliskan_names.csv"
PROBE_CSV = ROOT / "data" / "names" / "lexical_probe_names.csv"
CACHE = ROOT / "cache" / "names"

# The pool's names always come from this set, whichever set is swept.
POOL_DESIGN = "frequency_matched"

BATCHES = ("oneitem", "twoitem", "threeitem")
K = 10

# The name is prepended to the CV, followed by a blank line.
SEPARATOR = "\n\n"

# Separate seeds for the pool name assignment and the focal CV sample.
POOL_SEED = 20240717
FOCAL_SEED = 20240718

DROPPED = "Elmer"  # printed under WF in Wilson & Caliskan's Table 6; not used


def named(text: str, full_name: str) -> str:
    """The document as the encoder sees it."""
    return f"{full_name}{SEPARATOR}{text}"


def load_names(design: str = "frequency_matched", keep_elmer: bool = False):
    """The stimulus set, in file order. ``frequency_matched``: Black names with
    frequency-matched White names; ``primary``: Wilson & Caliskan's headline
    set; ``lexical_probe``: own stimuli."""
    if design == "lexical_probe":
        return list(csv.DictReader(PROBE_CSV.open(encoding="utf-8")))
    rows = list(csv.DictReader(NAMES_CSV.open(encoding="utf-8")))
    if design == "primary":
        chosen = [r for r in rows if r["set"] == "primary"]
    elif design == "frequency_matched":
        chosen = [r for r in rows
                  if (r["group"] in ("BF", "BM") and r["set"] == "primary")
                  or r["set"] == "supplemental_freq_matched"]
    else:
        raise ValueError(f"unknown design set {design!r}")
    if not keep_elmer:
        chosen = [r for r in chosen if r["first_name"] != DROPPED]
    seen = {r["FullName"] for r in chosen}
    if len(seen) != len(chosen):
        raise ValueError("stimulus set contains a repeated full name")
    return chosen


def _digest(strings) -> str:
    digest = hashlib.sha256()
    for s in sorted(strings):
        digest.update(s.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _write_atomic(path: Path, write) -> None:
    """Write through a temporary file in the same directory, then rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _save_npz(path: Path, arrays) -> None:
    # Through a file object: savez_compressed would otherwise append ".npz".
    with path.open("wb") as fh:
        np.savez_compressed(fh, **arrays)


def reference_assignment(records, names, seed: int = POOL_SEED) -> dict:
    """One name per CV, drawn independently, ordered by ``custom_id``."""
    rng = random.Random(seed)
    cvs = sorted((r for r in records if r.profile_type == "cv"),
                 key=lambda r: r.custom_id)
    return {r.custom_id: names[rng.randrange(len(names))]["FullName"] for r in cvs}


def focal_records(records, count: int, seed: int = FOCAL_SEED):
    """``count`` CVs spread evenly over batch and style."""
    cells: dict[tuple, list] = {}
    for r in sorted((r for r in records if r.profile_type == "cv"),
                    key=lambda r: r.custom_id):
        cells.setdefault((r.batch_name, r.style), []).append(r)
    rng = random.Random(seed)
    order = sorted(cells)
    rng.shuffle(order)
    for key in order:
        rng.shuffle(cells[key])
    chosen, position = [], 0
    while len(chosen) < count:
        drained = True
        for key in order:
            if position < len(cells[key]):
                chosen.append(cells[key][position])
                drained = False
                if len(chosen) == count:
                    break
        if drained:
            raise ValueError(f"corpus holds fewer than {count} CVs")
        position += 1
    return sorted(chosen, key=lambda r: r.custom_id)


def ranks_for(scores, base, focal_index, doc_ids):
    """Rank of the focal document under each name; ties go to the larger document
    id, as in pytrec_eval. ``scores`` is (names, queries) for the focal
    document, ``base`` (queries, documents) for the pool."""
    higher_id = np.array([d > doc_ids[focal_index] for d in doc_ids])
    out = np.empty(scores.shape, dtype=np.int32)
    for i, row in enumerate(scores):                       # row: (queries,)
        s = row[:, None]                                   # (queries, 1)
        # the focal document's own pool column is excluded
        greater = (base > s).sum(1) - (base[:, focal_index] > row)
        equal = ((base == s) & higher_id[None, :]).sum(1)   # self is never > self
        out[i] = 1 + greater + equal
    return out


def _bm25_names(records, blinded, pool_names, focal, name_list, batches):
    """BM25 effect of each name, over the same named pool."""
    if not focal:
        return {"checked": 0, "by_name": {}}
    text_of = {r.custom_id: r.profile_text for r in records}
    out = {n: {"max_score_change": 0.0, "entered_top10": 0, "left_top10": 0}
           for n in name_list}
    checked = 0
    for batch in batches:
        here = [r for r in focal if r.batch_name == batch]
        if not here:
            continue
        recs = [r for r in records if r.batch_name == batch]
        view = blind.select(blinded, recs)
        custom_of = {d.id: blinded.source[d.id] for d in view.documents}
        docs = [blind.Document(d.id, named(text_of[custom_of[d.id]],
                                           pool_names[custom_of[d.id]]))
                for d in view.documents]
        dids = [d.id for d in docs]
        qids = [q.id for q in view.queries]

        def matrix(run_):
            # The whole matrix: BM25's length normalisation can move other documents.
            return np.array([[run_[q][d] for d in dids] for q in qids])

        base = matrix(bm25.rank(view.queries, docs))
        for record in here:
            index = dids.index(blinded_id(blinded, record.custom_id))
            base_top = ranks_for(base[:, index][None], base, index, dids)[0] <= K
            for name in name_list:
                swapped = list(docs)
                swapped[index] = blind.Document(
                    dids[index], named(text_of[record.custom_id], name))
                moved = matrix(bm25.rank(view.queries, swapped))
                top = ranks_for(moved[:, index][None], moved, index, dids)[0] <= K
                row = out[name]
                row["max_score_change"] = max(
                    row["max_score_change"], float(np.abs(moved - base).max()))
                row["entered_top10"] += int((top & ~base_top).sum())
                row["left_top10"] += int((~top & base_top).sum())
                checked += 1
        print(f"  bm25: {batch}, {len(here)} focal CVs x {len(name_list)} names",
              flush=True)
    worst = max(r["max_score_change"] for r in out.values())
    return {"checked": checked, "identical": worst == 0.0,
            "max_score_change": worst, "by_name": out}


def blinded_id(blinded, custom_id: str) -> str:
    for item_id, source in blinded.source.items():
        if source == custom_id:
            return item_id
    raise KeyError(custom_id)
