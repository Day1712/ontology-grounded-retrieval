"""Reports an invariance run: score sensitivity (A), rank consequence and shortlist
instability (B), demographic structure (C) and the unnamed reference (D).
Writes cache/invariance/report.json (table "median rank without a name")."""

import argparse
import json
from pathlib import Path

import numpy as np

from experiments.invariance import CACHE, _paths
from experiments.names import _write_atomic

BANDS = ((1, 10), (11, 25), (26, 50), (51, 100), (101, 250), (251, 10 ** 9))
BOUNDARY = (1, 20)
K = 10
DEMOGRAPHIC = ("BF", "BM", "WF", "WM")
CONTRASTS = (("F-M", ("BF", "WF"), ("BM", "WM")),
             ("B-W", ("BF", "BM"), ("WF", "WM")),
             ("BF-WM", ("BF",), ("WM",)))


def load(slug: str):
    _, results_path, ranks_path = _paths(slug)
    if not Path(results_path).exists() or not Path(ranks_path).exists():
        raise SystemExit(
            f"no stored run for {slug!r}. Run `python3 -m experiments.invariance "
            f"--encoder {slug} --device DEVICE` first.")
    summary = json.loads(Path(results_path).read_text())
    arrays = np.load(ranks_path)
    cvs = sorted({k.split("/", 1)[1] for k in arrays.files if k.startswith("ranks/")})
    groups = np.array(summary["meta"]["groups"])
    kinds = np.array(summary["meta"]["kinds"])
    scores = np.concatenate([arrays[f"scores/{c}"] for c in cvs], axis=1)
    ranks = np.concatenate([arrays[f"ranks/{c}"] for c in cvs], axis=1)
    pool_sd = np.concatenate([arrays[f"pool_sd/{c}"] for c in cvs])
    plain = None
    if f"plain_ranks/{cvs[0]}" in arrays.files:
        plain = np.concatenate([arrays[f"plain_ranks/{c}"] for c in cvs])
    return summary, cvs, groups, kinds, scores, ranks, pool_sd, plain


def mad(x, axis=0):
    return np.median(np.abs(x - np.median(x, axis=axis, keepdims=True)), axis=axis)


def sensitivity(scores, kinds, pool_sd):
    """A. Score dispersion per pair, in units of the vacancy's own field spread."""
    out = {}
    for kind in sorted(set(kinds.tolist())):
        rows = kinds == kind
        if rows.sum() < 2:
            continue          # a dispersion over one stimulus is not a number
        normalised = mad(scores[rows]) / np.maximum(pool_sd, 1e-12)
        out[kind] = {
            "stimuli": int(rows.sum()),
            "median_normalised_mad": float(np.median(normalised)),
            "p90_normalised_mad": float(np.percentile(normalised, 90)),
            "median_raw_mad": float(np.median(mad(scores[rows]))),
        }
    if {"name", "identifier"} <= set(out):
        out["name_over_identifier"] = (out["name"]["median_normalised_mad"] /
                                       max(out["identifier"]["median_normalised_mad"], 1e-12))
    return out


def consequence(ranks, kinds):
    """B1. 10-90 rank span across the names, within bands of central rank."""
    r = ranks[kinds == "name"]
    central = np.median(r, axis=0)
    span = np.percentile(r, 90, axis=0) - np.percentile(r, 10, axis=0)
    rows = []
    for lo, hi in BANDS:
        m = (central >= lo) & (central <= hi)
        if not m.any():
            continue
        rows.append({"band": f"{lo}-{'+' if hi > 10 ** 8 else hi}",
                     "pairs": int(m.sum()),
                     "median_span": float(np.median(span[m])),
                     "p90_span": float(np.percentile(span[m], 90))})
    return {"bands": rows, "note": "rank span is comparable within a band only"}


def instability(ranks, kinds, k=K, boundary=BOUNDARY):
    """B2. Probability that two randomly drawn names disagree on shortlisting:
    ``2p(1-p)`` scaled to sampling without replacement."""
    r = ranks[kinds == "name"]
    n = r.shape[0]
    central = np.median(r, axis=0)
    live = (central >= boundary[0]) & (central <= boundary[1])
    p = (r <= k).mean(0)
    d = 2 * p * (1 - p) * n / (n - 1)
    return {"k": k, "boundary_band": f"{boundary[0]}-{boundary[1]}",
            "pairs_in_band": int(live.sum()),
            "pairs_total": int(central.size),
            "mean_disagreement": float(d[live].mean()) if live.any() else None,
            "median_disagreement": float(np.median(d[live])) if live.any() else None,
            "share_fully_stable": float((d[live] == 0).mean()) if live.any() else None}


def structure(scores, ranks, groups, kinds, pool_sd, k=K, boundary=BOUNDARY):
    """C. How much of the name variation is demographic, and in what direction."""
    keep = kinds == "name"
    s, r, g = scores[keep], ranks[keep], groups[keep]
    centred = s - s.mean(0, keepdims=True)          # within pair, then aggregate
    per_name = centred.mean(1)                      # one value per name

    grand = per_name.mean()
    fitted = np.array([per_name[g == x].mean() for x in g])
    ss_total = ((per_name - grand) ** 2).sum()
    eta2 = float(((fitted - grand) ** 2).sum() / ss_total) if ss_total else 0.0

    central = np.median(r, axis=0)
    live = (central >= boundary[0]) & (central <= boundary[1])
    top = r <= k

    by_group = {}
    for x in DEMOGRAPHIC:
        m = g == x
        if not m.any():
            continue
        by_group[x] = {
            "names": int(m.sum()),
            "mean_centred_score": float(centred[m].mean()),
            "mean_centred_score_pool_sd": float(
                (centred[m] / np.maximum(pool_sd, 1e-12)[None, :]).mean()),
            "mean_rank": float(r[m].mean()),
            "top_k_probability_boundary": (float(top[m][:, live].mean())
                                           if live.any() else None),
        }
    contrasts = {}
    for label, a, b in CONTRASTS:
        ma, mb = np.isin(g, a), np.isin(g, b)
        if not (ma.any() and mb.any()):
            continue
        entry = {"names": [int(ma.sum()), int(mb.sum())],
                 "centred_score": float(centred[ma].mean() - centred[mb].mean()),
                 "mean_rank": float(r[ma].mean() - r[mb].mean())}
        if live.any():
            entry["top_k_probability_boundary"] = float(
                top[ma][:, live].mean() - top[mb][:, live].mean())
        contrasts[label] = entry
    return {"variance_share_demographic": eta2,
            "interpretation": ("demographic grouping accounts for "
                               f"{eta2 * 100:.1f}% of the variation between names; "
                               "the remainder is individual-name variation"),
            "by_group": by_group, "contrasts": contrasts}


def plain_reference(ranks, kinds, plain):
    """D. Where the candidate sits with no name, beside where names put them;
    reported apart from A, B and C."""
    if plain is None or not (kinds == "plain").any():
        return None
    central = np.median(ranks[kinds == "name"], axis=0)
    unnamed_named_field = ranks[kinds == "plain"][0]
    return {
        "median_central_rank_under_names": float(np.median(central)),
        "median_rank_unnamed_in_named_field": float(np.median(unnamed_named_field)),
        "median_rank_unnamed_in_unnamed_field": float(np.median(plain)),
        "note": ("reference only; excluded from A, B and C because an unnamed CV "
                 "among named competitors varies in more than which name it carries"),
    }


def report(slug: str):
    summary, cvs, groups, kinds, scores, ranks, pool_sd, plain = load(slug)
    return {
        "meta": {"encoder": summary["meta"]["encoder"],
                 "model": summary["meta"]["model"],
                 "revision": summary["meta"]["revision"],
                 "device": summary["meta"]["device"],
                 "dtype": summary["meta"]["dtype"],
                 "focal_cvs": len(cvs),
                 "pairs": int(ranks.shape[1]),
                 "stimuli": int(ranks.shape[0]),
                 "bm25_identical": summary["bm25"].get("identical")},
        "A_sensitivity": sensitivity(scores, kinds, pool_sd),
        "B1_rank_consequence": consequence(ranks, kinds),
        "B2_decision_instability": instability(ranks, kinds),
        "C_demographic_structure": structure(scores, ranks, groups, kinds, pool_sd),
        "D_plain_reference": plain_reference(ranks, kinds, plain),
    }


def _print(r):
    m = r["meta"]
    print(f"\n{m['encoder']}  ({m['model']} @ {m['revision'][:8]}, {m['device']})")
    print(f"{m['focal_cvs']} focal CVs x {m['pairs'] // m['focal_cvs']} vacancies, "
          f"{m['stimuli']} stimuli; BM25 identical: {m['bm25_identical']}")

    a = r["A_sensitivity"]
    print("\nA. score sensitivity (MAD, in pool-SD units)")
    for kind in ("name", "identifier"):
        if kind in a:
            print(f"  {kind:11} median {a[kind]['median_normalised_mad']:.4f}   "
                  f"p90 {a[kind]['p90_normalised_mad']:.4f}   "
                  f"({a[kind]['stimuli']} stimuli)")
    if "name_over_identifier" in a:
        print(f"  names are {a['name_over_identifier']:.2f}x the arbitrary-identifier floor")

    print("\nB1. rank span across names, 10th-90th percentile, by central rank")
    print(f"  {'band':>10} {'pairs':>7} {'median':>8} {'p90':>8}")
    for row in r["B1_rank_consequence"]["bands"]:
        print(f"  {row['band']:>10} {row['pairs']:>7} {row['median_span']:>8.0f} "
              f"{row['p90_span']:>8.0f}")

    b2 = r["B2_decision_instability"]
    print(f"\nB2. shortlist instability, central rank {b2['boundary_band']} "
          f"({b2['pairs_in_band']} of {b2['pairs_total']} pairs)")
    if b2["mean_disagreement"] is not None:
        print(f"  two randomly drawn names disagree on top-{b2['k']} membership "
              f"{b2['mean_disagreement'] * 100:.1f}% of the time")
        print(f"  fully stable pairs: {b2['share_fully_stable'] * 100:.1f}%")

    c = r["C_demographic_structure"]
    print("\nC. demographic structure")
    print(f"  {c['interpretation']}")
    print(f"  {'group':>6} {'names':>6} {'centred score':>14} {'top-k prob':>11}")
    for g, row in c["by_group"].items():
        prob = row["top_k_probability_boundary"]
        print(f"  {g:>6} {row['names']:>6} {row['mean_centred_score']:>14.5f} "
              f"{'--' if prob is None else f'{prob:>11.3f}'}")
    d = r.get("D_plain_reference")
    if d:
        print("\nD. unnamed reference (not part of A-C)")
        print(f"  median rank under names          {d['median_central_rank_under_names']:.0f}")
        print(f"  unnamed, named competitors       {d['median_rank_unnamed_in_named_field']:.0f}")
        print(f"  unnamed, unnamed competitors     {d['median_rank_unnamed_in_unnamed_field']:.0f}")

    for label, row in c["contrasts"].items():
        extra = row.get("top_k_probability_boundary")
        print(f"  {label:>6} centred score {row['centred_score']:+.5f}"
              + (f"   top-k prob {extra:+.3f}" if extra is not None else ""))


def comparable(slugs):
    """Refuse runs that differ in corpus, pool assignment, stimuli, batches or k."""
    fields = ("corpus", "pool_assignment", "stimuli", "batches", "k")
    seen = {}
    for slug in slugs:
        _, results_path, _ = _paths(slug)
        meta = json.loads(Path(results_path).read_text())["meta"]
        seen[slug] = {f: meta.get(f) for f in fields}
    first = next(iter(seen))
    differing = sorted({f for slug in seen for f in fields
                        if seen[slug][f] != seen[first][f]})
    if differing:
        raise SystemExit(
            f"runs are not comparable: {', '.join(differing)} differ between "
            f"{', '.join(seen)}. Re-run the odd one out rather than reporting "
            f"them side by side.")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True, action="append")
    args = parser.parse_args()
    if len(args.encoder) > 1:
        comparable(args.encoder)
        print(f"{len(args.encoder)} runs share corpus, pool assignment and stimuli")
    out = {}
    for slug in args.encoder:
        out[slug] = report(slug)
        _print(out[slug])
    path = CACHE / "report.json"
    stored = json.loads(path.read_text()) if path.exists() else {}
    stored.update(out)
    _write_atomic(path, lambda p: p.write_text(json.dumps(stored, indent=1) + "\n"))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
