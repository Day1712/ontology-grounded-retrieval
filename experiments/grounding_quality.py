"""Grounding F1 per branch with each channel (lexical, semantic) switched off in
turn, and the overlap of evidence sentences with O*NET descriptions.
Writes cache/grounding_quality.json (table "F1 per branch and channel")."""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from evaluation import corpus
from ranking_method import link, rank, resource

from . import benchmark, thresholds

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "cache" / "grounding_quality.json"
OFF = 2.0   # a cosine threshold above 1 switches a channel off
WORD = re.compile(r"[A-Za-z]+")

CHANNELS = {
    "both": (link.SEMANTIC_THRESHOLD, link.LEXICAL_SUPPORT_THRESHOLD),
    "lexical only": (OFF, link.LEXICAL_SUPPORT_THRESHOLD),
    "semantic only": (link.SEMANTIC_THRESHOLD, OFF),
}


def channels(texts, gold, order, vocab):
    """Per-branch quality with each mechanism switched off in turn."""
    out = {}
    for label, (semantic, lexical) in CHANNELS.items():
        items = link.annotate(texts, vocab=vocab, scored=order,
                              semantic_threshold=semantic, lexical_threshold=lexical)
        macro, per = thresholds.quality(items, gold)
        out[label] = {"macro_f1": macro,
                      "per_branch": {b: per[b] for b in thresholds.BRANCHES}}
        print(f"  {label:14} macro-F1 {macro:.4f}", flush=True)
    return out


def overlap(records, grounded):
    """Share of an element's description words reused by its evidence sentence in
    CVs (the sentence holding the item's first supporting span)."""
    elements = resource.elements()
    per = defaultdict(list)
    for record in records:
        if record.profile_type != "cv":
            continue
        spans = link.sentences(record.profile_text)
        for item in grounded[rank.text_key(record.profile_text)]:
            element = elements.get(item.element_id)
            if element is None or not item.support:
                continue
            want = {w.lower() for w in WORD.findall(element.description or "")}
            if not want:
                continue
            start = item.support[0].start
            sentence = next((s for s, a, b in spans if a <= start < b), "")
            have = {w.lower() for w in WORD.findall(sentence)}
            per[resource.branch(item.element_id)].append(len(want & have) / len(want))
    out = {b: {"mean": sum(v) / len(v), "n": len(v)} for b, v in sorted(per.items())}
    every = [x for v in per.values() for x in v]
    out["all"] = {"mean": sum(every) / len(every), "n": len(every)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default=benchmark.DEVICE)
    args = parser.parse_args()

    records = corpus.load_corpus()
    texts = [r.profile_text for r in records]
    vocab = link.vocabulary()
    _, scored = thresholds.similarity(texts, args.device, vocab=vocab)
    order = [scored[rank.text_key(t)] for t in texts]

    print("channels")
    channel_rows = channels(texts, thresholds.intended(records), order, vocab)

    print("\ndescription overlap, CV evidence sentences")
    overlap_rows = overlap(records, benchmark.grounding(texts, device=args.device,
                                                        vocab=vocab))
    for branch, row in overlap_rows.items():
        print(f"  {branch:4} {row['mean']:.3f}  (n={row['n']:,})")

    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps({"channels": channel_rows,
                                   "description_overlap": overlap_rows}, indent=1) + "\n")
    print(f"\nwrote {RESULTS}")


if __name__ == "__main__":
    main()
