"""The method under name replacement, with the same pool and focal CVs as the
dense runs. Writes cache/names/probe_method.json (Ada, Pascal, Ruby)."""

import argparse
import json

import numpy as np

from evaluation import blind, corpus
from experiments.names import (CACHE, K, POOL_DESIGN, _write_atomic, focal_records,
                               load_names, named, ranks_for, reference_assignment)
from ranking_method import grade, link, policy, rank as rank_mod, represent

GROUNDING = CACHE / "grounding.json"
DEVICE = "mps"
LADDER = policy.CONCEPTS_ORDINAL
BATCHES = ("oneitem", "twoitem", "threeitem")
CHUNK = 256


def _meta(vocab):
    return {
        "format": 1,
        "model": link.MODEL,
        "semantic_threshold": link.SEMANTIC_THRESHOLD,
        "lexical_threshold": link.LEXICAL_SUPPORT_THRESHOLD,
        "vocabulary": len(vocab),
        "ordinal": len(policy.ordinal_elements()),
    }


def _dump(items):
    from dataclasses import astuple
    return [[i.element_id, i.category, [astuple(s) for s in i.support]] for i in items]


def _load(rows):
    return tuple(link.GroundedItem(e, c, tuple(link.Support(*s) for s in support))
                 for e, c, support in rows)


def grounding(texts, vocab, enc=None, path=GROUNDING):
    """{text_key: items}, computing only what is not cached."""
    meta = _meta(vocab)
    stored = {}
    if path.exists():
        held = json.loads(path.read_text())
        if held.get("meta") != meta:
            raise SystemExit(f"{path} was written under different grounding settings; "
                             f"move or delete it to re-ground.")
        stored = held["items"]

    keys = {t: rank_mod.text_key(t) for t in dict.fromkeys(texts)}
    missing = [t for t, k in keys.items() if k not in stored]
    print(f"  grounding: {len(keys) - len(missing)} cached, {len(missing)} to compute",
          flush=True)
    if missing:
        enc = enc if enc is not None else link.encoder(device=DEVICE)
        for i in range(0, len(missing), CHUNK):
            block = missing[i:i + CHUNK]
            for text, items in zip(block, link.annotate(block, vocab=vocab, enc=enc)):
                stored[keys[text]] = _dump(items)
            _write_atomic(path, lambda p: p.write_text(
                json.dumps({"meta": meta, "items": stored})))
            print(f"    {min(i + CHUNK, len(missing))}/{len(missing)}", flush=True)
    return {k: _load(stored[k]) for k in keys.values()}


def _stated(item_id, text, subject, grounded, requiring=False):
    """One side's assertions, exactly as `rank.rank` derives them."""
    items = grounded[rank_mod.text_key(text)]
    grades = grade.read(text, items) if (requiring and LADDER.graded) else None
    found = represent.assertions(subject(item_id), represent.document(item_id),
                                 items, requiring=requiring, grades=grades)
    if LADDER.ordinal:
        return found
    return tuple(a for a in found if a.predicate != represent.ONET_CATEGORY)


def run(focal_count=100, design="lexical_probe", batches=BATCHES, tag="probe_method"):
    records = corpus.load_corpus()
    names = load_names(design)
    name_list = [n["FullName"] for n in names]
    pool_names = reference_assignment(records, load_names(POOL_DESIGN))
    focal = [r for r in focal_records(records, focal_count) if r.batch_name in batches]
    blinded = blind.blind(records)
    by_custom = {c: i for i, c in blinded.source.items()}
    text_of = {r.custom_id: r.profile_text for r in records}
    vocab = link.vocabulary()

    wanted = []
    for batch in batches:
        for r in records:
            if r.batch_name != batch:
                continue
            wanted.append(r.profile_text if r.profile_type == "jd"
                          else named(r.profile_text, pool_names[r.custom_id]))
    for r in focal:
        wanted += [named(r.profile_text, n) for n in name_list]
    print(f"{len(name_list)} names, {len(focal)} focal CVs")
    grounded = grounding(wanted, vocab)

    results, arrays = {}, {}
    for batch in batches:
        recs = [r for r in records if r.batch_name == batch]
        view = blind.select(blinded, recs)
        custom_of = {d.id: blinded.source[d.id] for d in view.documents}
        docs = [blind.Document(d.id, named(text_of[custom_of[d.id]],
                                           pool_names[custom_of[d.id]]))
                for d in view.documents]
        dids = [d.id for d in docs]
        print(f"\n{batch}: {len(view.queries)} JDs x {len(docs)} CVs", flush=True)

        run_ = rank_mod.rank(view.queries, docs, grounded=grounded,
                             ladder=LADDER, key=rank_mod.LEXICOGRAPHIC)
        base = np.array([[run_[q.id][d] for d in dids] for q in view.queries])
        required = {q.id: _stated(q.id, q.text, represent.role, grounded, True)
                    for q in view.queries}

        for r in [f for f in focal if f.batch_name == batch]:
            index = dids.index(by_custom[r.custom_id])
            reference = base[:, index]
            rows = []
            for name in name_list:
                evidence = represent.index(
                    _stated(dids[index], named(r.profile_text, name),
                            represent.person, grounded))
                rows.append([policy.coverage(required[q.id], evidence,
                                             LADDER).lexicographic()
                             for q in view.queries])
            scores = np.array(rows)
            place = ranks_for(scores, base, index, dids)
            base_place = ranks_for(reference[None], base, index, dids)[0]
            arrays[f"scores/{r.custom_id}"] = scores.astype(np.float32)
            arrays[f"ranks/{r.custom_id}"] = place
            arrays[f"reference_ranks/{r.custom_id}"] = base_place
            top, base_top = place <= K, base_place <= K
            results[r.custom_id] = {
                "batch": batch,
                "queries": int(scores.shape[1]),
                "by_name": {
                    n: {
                        "mean_delta_score": float((scores[i] - reference).mean()),
                        "queries_score_changed": int((scores[i] != reference).sum()),
                        "entered_top10": int((top[i] & ~base_top).sum()),
                        "left_top10": int((~top[i] & base_top).sum()),
                    }
                    for i, n in enumerate(name_list)
                },
            }
        print(f"  {len([f for f in focal if f.batch_name == batch])} focal CVs done",
              flush=True)

    summary = {"meta": {**_meta(vocab), "design_set": design, "names": name_list,
                        "focal": len(focal), "ladder": "concepts + ordinal", "k": K},
               "focal": results}
    _write_atomic(CACHE / f"{tag}.json",
                  lambda p: p.write_text(json.dumps(summary, indent=1) + "\n"))
    _write_atomic(CACHE / f"{tag}_ranks.npz",
                  lambda p: np.savez_compressed(p.open("wb"), **arrays))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--focal", type=int, default=100)
    parser.add_argument("--design", default="lexical_probe")
    parser.add_argument("--tag", default="probe_method")
    args = parser.parse_args()
    summary = run(focal_count=args.focal, design=args.design, tag=args.tag)

    rows = summary["focal"]
    print(f"\n{'name':20} {'changed':>9} {'mean d':>10} {'entered':>9} {'left':>7} {'net':>7}")
    for name in summary["meta"]["names"]:
        by = [r["by_name"][name] for r in rows.values()]
        ent = sum(b["entered_top10"] for b in by)
        lef = sum(b["left_top10"] for b in by)
        print(f"{name:20} {sum(b['queries_score_changed'] for b in by):9,}"
              f" {np.mean([b['mean_delta_score'] for b in by]):+10.5f}"
              f" {ent:9d} {lef:7d} {ent - lef:+7d}")
    print(f"\n{len(rows)} focal CVs x {sum(r['queries'] for r in rows.values()):,} "
          f"vacancy-slots per name")


if __name__ == "__main__":
    main()
