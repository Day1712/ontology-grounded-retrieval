"""Grid search over the grounding settings (case rule, k, semantic threshold),
scored by macro-F1 over branches against the concepts each profile states.
Writes cache/similarity.npz and cache/thresholds.json (table "grounding grid search")."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluation import corpus
from ranking_method import link, rank, resource

from experiments import benchmark

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache" / "similarity.npz"
RESULTS = ROOT / "cache" / "thresholds.json"
BRANCHES = ("1.A", "1.D", "2.A", "2.B", "2.C", "2.E", "4.A")

SEMANTIC = (0.780, 0.805, 0.830, 0.850, 0.870, 0.890)
KS = (1, 2, 3)
#: Branch sets in which a single-word element name is matched case sensitively.
CASE = {
    "none": (),
    "skills+knowledge": ("2.A", "2.B", "2.C"),
    "all": BRANCHES,
}


def _meta(texts, device, vocab):
    digest = __import__("hashlib").sha256()
    for text in sorted(texts):
        digest.update(rank.text_key(text).encode())
    return {"format": 1, "release": resource.release(), "model": link.MODEL,
            "device": device, "vocabulary": len(vocab),
            "corpus": digest.hexdigest()}


def similarity(texts, device=benchmark.DEVICE, cache=CACHE, vocab=None):
    """``(ids, {text key: (spans, similarities)})``, cached across runs."""
    vocab = vocab if vocab is not None else link.vocabulary()
    meta = _meta(texts, device, vocab)
    keys = [rank.text_key(t) for t in texts]

    def unpack(ids, counts, spans, sim):
        out, at = {}, 0
        for key, n in zip(keys, counts):
            out[key] = (tuple((None, int(a), int(b)) for a, b in spans[at:at + n]),
                        sim[at:at + n])
            at += n
        return ids, out

    if cache is not None and cache.exists():
        stored = np.load(cache, allow_pickle=False)
        if json.loads(str(stored["meta"])) == meta:
            return unpack([str(i) for i in stored["ids"]], stored["counts"],
                          stored["spans"], stored["sim"])

    enc = link.encoder(device=device)
    ids, matrix = link.encode_elements(vocab, enc)
    scored = list(link.scores(texts, matrix, enc))
    counts = np.array([len(doc) for doc, _ in scored], dtype=np.int64)
    spans = np.array([[s, e] for doc, _ in scored for _, s, e in doc], dtype=np.int64)
    sim = np.concatenate([s for _, s in scored]).astype(np.float32)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, meta=json.dumps(meta), ids=np.array(ids), counts=counts,
                 spans=spans, sim=sim, keys=np.array(keys))
    return unpack(ids, counts, spans, sim)


def intended(records):
    """What each profile states, as element ids."""
    root = ROOT / "data" / "onet"

    def rows(name):
        with (root / name).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle, delimiter="\t"))

    to_gwa = {r["DWA Element ID"]: r["GWA Element ID"]
              for r in rows("GWAs to IWAs to DWAs.txt")}
    task = defaultdict(set)
    for r in rows("Tasks to DWAs.txt"):
        if r["DWA Element ID"] in to_gwa:
            task[r["Task ID"]].add(to_gwa[r["DWA Element ID"]])

    known = resource.elements()
    by_name = {e.name.lower(): i for i, e in known.items()}
    out = []
    for record in records:
        want = set()
        for key, value in record.profile_dict.items():
            kind = value.get("kind")
            if kind == "task":
                want |= task.get(key, set())
            elif kind in ("skill", "knowledge", "abilities", "work_style"):
                if key in known:
                    want.add(key)
            elif kind == "tech_skill":
                found = by_name.get((value.get("name") or "").lower())
                if found:
                    want.add(found)
        out.append({i for i in want if resource.branch(i) in BRANCHES})
    return out


def vocabulary(case):
    """The vocabulary, with single-word names strict in the ``case`` branches."""
    examples = resource.workplace_examples()
    activities = resource.activity_forms()
    known = resource.elements()
    out = {}
    for branch in BRANCHES:
        for element_id in link.groundable(branch):
            name = known[element_id].name
            forms = [(name, len(name.split()) == 1 and branch in case)]
            forms += [(x, len(x.split()) == 1) for x in examples.get(element_id, ())]
            forms += [(x, False) for x in activities.get(element_id, ())]
            out[element_id] = tuple(forms)
    return out


def quality(items, gold):
    """Per-branch precision, recall and F1, and the macro-average."""
    tp, fp, fn = defaultdict(int), defaultdict(int), defaultdict(int)
    for want, row in zip(gold, items):
        got = {x.element_id for x in row if x.category is None}
        got = {i for i in got if resource.branch(i) in BRANCHES}
        for i in got & want:
            tp[resource.branch(i)] += 1
        for i in got - want:
            fp[resource.branch(i)] += 1
        for i in want - got:
            fn[resource.branch(i)] += 1
    per = {}
    for branch in BRANCHES:
        p = tp[branch] / (tp[branch] + fp[branch]) if tp[branch] + fp[branch] else 0.0
        r = tp[branch] / (tp[branch] + fn[branch]) if tp[branch] + fn[branch] else 0.0
        per[branch] = (p, r, 2 * p * r / (p + r) if p + r else 0.0)
    return sum(v[2] for v in per.values()) / len(BRANCHES), per


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=benchmark.DEVICE)
    args = parser.parse_args()

    records = corpus.load_corpus()
    texts = [r.profile_text for r in records]
    gold = intended(records)
    rows = []
    for label, case in CASE.items():
        vocab = vocabulary(case)
        _, scored = similarity(texts, args.device, CACHE, link.vocabulary())
        order = [scored[rank.text_key(t)] for t in texts]
        for k in KS:
            for sem in SEMANTIC:
                items = link.annotate(texts, vocab=vocab, scored=order,
                                      semantic_threshold=sem,
                                      lexical_threshold=link.LEXICAL_SUPPORT_THRESHOLD,
                                      k=k)
                macro, per = quality(items, gold)
                rows.append({"case": label, "k": k, "semantic": sem,
                             "macro_f1": macro,
                             "per_branch": {b: per[b] for b in BRANCHES}})
                print(f"  case={label:17} k={k} sem={sem:.3f}  macro-F1 {macro:.4f}",
                      flush=True)

    best = max(rows, key=lambda r: r["macro_f1"])
    print(f"\nbest: case={best['case']} k={best['k']} semantic={best['semantic']}"
          f"   macro-F1 {best['macro_f1']:.4f}")
    print(f"  {'branch':8} {'P':>7} {'R':>7} {'F1':>7}")
    for branch in BRANCHES:
        p, r, f = best["per_branch"][branch]
        print(f"  {branch:8} {p:7.3f} {r:7.3f} {f:7.3f}")
    print(f"\nshipped: semantic {link.SEMANTIC_THRESHOLD}, k=1, "
          f"strict outside {link.ATTRIBUTE_BRANCHES}")
    RESULTS.write_text(json.dumps(rows, indent=1) + "\n")


if __name__ == "__main__":
    main()
