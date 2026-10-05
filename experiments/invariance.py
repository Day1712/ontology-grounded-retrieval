"""Dense ranking under 99 name/identifier stimuli substituted into a focal CV.
Writes cache/invariance/<encoder>.json and _ranks.npz, and the cached dense
vectors also used for the dense rows of Table 2."""

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import torch

from evaluation import blind, corpus
from evaluation.baselines import encoders
from experiments.names import (BATCHES, FOCAL_SEED, K, POOL_DESIGN, POOL_SEED,
                               SEPARATOR, _bm25_names, _digest, _save_npz,
                               _write_atomic, focal_records,
                               load_names, named, ranks_for,
                               reference_assignment)
from ranking_method import rank as rank_mod

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "cache" / "invariance"
IDS_CSV = ROOT / "data" / "names" / "arbitrary_ids.csv"
VERIFICATION = ROOT / "cache" / "encoder_verification.json"
FOCAL = 100
CHUNK = 32
PLAIN_LABEL = "<none>"    # how the unnamed condition is written down
WORD = re.compile(r"[a-z0-9]+")


def stimuli(design: str = POOL_DESIGN, with_controls: bool = True,
            with_plain: bool = True):
    """The names, then the arbitrary identifiers, then the unnamed condition
    (``text=None``), which is stored as a reference point only."""
    out = [{"text": n["FullName"], "group": n["group"], "kind": "name"}
           for n in load_names(design)]
    if with_controls:
        out += [{"text": r["FullName"], "group": "ID", "kind": "identifier"}
                for r in csv.DictReader(IDS_CSV.open(encoding="utf-8"))]
    if with_plain:
        out.append({"text": None, "group": "PLAIN", "kind": "plain"})
    seen = {s["text"] for s in out}
    if len(seen) != len(out):
        raise ValueError("stimulus set contains a repeated entry")
    return out


def render(spec, cv_text: str, stimulus: str | None) -> str:
    """The document as the encoder sees it under one stimulus."""
    if stimulus is None:
        return spec.document_text(cv_text)
    return spec.document_text(named(cv_text, stimulus))


def assert_lexically_inert(records, texts):
    """Stimuli with a token that occurs in the corpus, mapped to those tokens."""
    corpus_words = set()
    for r in records:
        corpus_words.update(WORD.findall(r.profile_text.lower()))
    hits = {t: sorted(set(WORD.findall(t.lower())) & corpus_words)
            for t in texts if t is not None}
    hits = {t: w for t, w in hits.items() if w}
    return hits


def _paths(slug: str):
    return (CACHE / "embeddings" / slug, CACHE / f"{slug}.json",
            CACHE / f"{slug}_ranks.npz")


def _vector_path(root: Path, key: str) -> Path:
    # sharded by the first two hex characters of the key
    return root / key[:2] / f"{key}.npy"


def _check_cache_meta(root: Path, meta: dict) -> None:
    """Refuse a cache written under different encoder settings."""
    path = root / "meta.json"
    if path.exists():
        stored = json.loads(path.read_text())
        if stored != meta:
            differing = sorted(k for k in set(stored) | set(meta)
                               if stored.get(k) != meta.get(k))
            raise SystemExit(
                f"{path} was written under different encoder settings "
                f"({', '.join(differing)} differ). Move or delete {root} to "
                f"re-encode; it will not be reused.")
    else:
        _write_atomic(path, lambda p: p.write_text(json.dumps(meta, indent=1)))


def _require_verified(spec) -> dict:
    """A spec runs only after checks.encoders has confirmed it."""
    if not VERIFICATION.exists():
        raise SystemExit(
            f"{VERIFICATION} missing. Run `python3 -m checks.encoders "
            f"--device DEVICE` before encoding: a wrong prefix or a silent "
            f"truncation is invisible in the output.")
    stored = json.loads(VERIFICATION.read_text())
    record = stored.get(spec.slug)
    if not record or not record.get("verified"):
        raise SystemExit(
            f"{spec.slug} is not verified in {VERIFICATION}. Run "
            f"`python3 -m checks.encoders --device DEVICE --encoder {spec.slug}`.")
    return record


def environment() -> dict:
    """Library versions of this run; recorded, not part of the cache key."""
    import platform
    import transformers
    return {"python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "transformers": transformers.__version__,
            "numpy": np.__version__}


def encoder_meta(spec, device, dtype) -> dict:
    """What decides an embedding. Anything here differing invalidates a vector."""
    return {"format": 1, "encoder": spec.slug, "model": spec.model,
            "revision": spec.revision, "pooling": spec.pooling,
            "query_template": spec.query_template,
            "document_template": spec.document_template,
            "max_length": spec.max_length, "dtype": str(dtype),
            "device": str(device), "separator": SEPARATOR}


def _save_npy(path: Path, array) -> None:
    with path.open("wb") as fh:
        np.save(fh, array)


def embed(texts, spec, loaded, root: Path, chunk: int = CHUNK, label: str = ""):
    """{text_key: vector} for every text, computing only what is not cached."""
    keys = {t: rank_mod.text_key(t) for t in dict.fromkeys(texts)}
    out, missing = {}, []
    for text, key in keys.items():
        path = _vector_path(root, key)
        if path.exists():
            out[key] = np.load(path)
        else:
            missing.append(text)

    if missing:
        started = time.time()
        for i in range(0, len(missing), chunk):
            block = missing[i:i + chunk]
            vectors = encoders.encode(block, spec, *loaded).numpy().astype(np.float32)
            for text, vector in zip(block, vectors, strict=True):
                key = keys[text]
                out[key] = vector
                _write_atomic(_vector_path(root, key),
                              lambda p, v=vector: _save_npy(p, v))
            done = min(i + chunk, len(missing))
            rate = (time.time() - started) / done
            print(f"  {label}: {done}/{len(missing)} encoded ({rate:.3f}s each, "
                  f"{rate * (len(missing) - done) / 60:.1f} min left)", flush=True)
    print(f"  {label}: {len(keys) - len(missing)} cached, {len(missing)} computed",
          flush=True)
    return out


def _valid_device(device: str):
    try:
        torch.zeros(1).to(device)
    except Exception as exc:
        raise SystemExit(f"device {device!r} is not usable here: {exc}")
    return device


def _bm25_assertion(records, blinded, pool_names, focal, texts, batches, digest):
    """The BM25 control, cached by digest so a resumed run does not repeat it."""
    path = CACHE / "bm25_check.json"
    stored = json.loads(path.read_text()) if path.exists() else {}
    if stored.get("digest") == digest:
        print(f"bm25: reusing the check for this corpus and stimulus set "
              f"({stored['result']['checked']} comparisons, identical="
              f"{stored['result'].get('identical')})")
        return stored["result"]
    result = _bm25_names(records, blinded, pool_names, focal, texts, batches)
    _write_atomic(path, lambda p: p.write_text(
        json.dumps({"digest": digest, "result": result}, indent=1) + "\n"))
    return result


def run(spec, device, dtype=torch.float32, focal_count=FOCAL, batches=BATCHES,
        with_controls=True, bm25_focal=3, dry_run=False):
    embeddings, results_path, ranks_path = _paths(spec.slug)
    records = corpus.load_corpus()
    stim = stimuli(with_controls=with_controls)
    texts = [s["text"] for s in stim]
    labels = [PLAIN_LABEL if t is None else t for t in texts]
    groups = [s["group"] for s in stim]
    named_texts = [t for t in texts if t is not None]

    collisions = assert_lexically_inert(records, texts)
    if collisions:
        shown = ", ".join(f"{t} ({'/'.join(w)})" for t, w in list(collisions.items())[:5])
        raise SystemExit(
            f"{len(collisions)} stimuli occur in the corpus: {shown}. BM25 "
            f"invariance no longer follows, and the control floor is not inert.")

    pool_names = reference_assignment(records, load_names(POOL_DESIGN))
    corpus_digest = _digest(r.custom_id for r in records)
    focal = [r for r in focal_records(records, focal_count)
             if r.batch_name in batches]
    print(f"{spec.slug}: {len(stim)} stimuli ("
          + ", ".join(f"{g}={groups.count(g)}" for g in sorted(set(groups)))
          + f"), {len(focal)} focal CVs, lexically inert")

    pool_digest = _digest(f"{k}={v}" for k, v in pool_names.items())
    blinded = blind.blind(records)
    by_custom = {c: i for i, c in blinded.source.items()}
    text_of = {r.custom_id: r.profile_text for r in records}

    wanted: list[str] = []
    for batch in batches:
        for r in (x for x in records if x.batch_name == batch):
            if r.profile_type == "jd":
                wanted.append(spec.query_text(r.profile_text))
            else:
                wanted.append(spec.document_text(
                    named(r.profile_text, pool_names[r.custom_id])))
            if r.profile_type == "cv" and any(s is None for s in texts):
                wanted.append(spec.document_text(r.profile_text))
    for r in focal:
        wanted += [render(spec, r.profile_text, t) for t in texts]

    keys = {rank_mod.text_key(t) for t in dict.fromkeys(wanted)}
    todo = [k for k in keys if not _vector_path(embeddings, k).exists()]
    print(f"{len(keys):,} distinct texts, {len(todo):,} to encode")
    if dry_run:
        return None

    _valid_device(device)
    verification = _require_verified(spec)
    meta = encoder_meta(spec, device, dtype)
    _check_cache_meta(embeddings, meta)

    # BM25 must not move; checked before any encoding.
    bm25_result = _bm25_assertion(
        records, blinded, pool_names, focal[:bm25_focal], named_texts, batches,
        _digest([corpus_digest, pool_digest, *labels, *batches,
                 *(r.custom_id for r in focal[:bm25_focal])]))
    if not bm25_result.get("identical", True):
        raise SystemExit(
            f"BM25 moved (max {bm25_result['max_score_change']}). The harness "
            f"is wrong; do not read dense numbers from this configuration.")

    loaded = encoders.load(spec, dtype=dtype, device=device) if todo else None

    vectors: dict[str, np.ndarray] = {}
    arrays = {}
    for batch in batches:
        recs = [r for r in records if r.batch_name == batch]
        view = blind.select(blinded, recs)
        doc_ids = [d.id for d in view.documents]
        custom_of = {d.id: blinded.source[d.id] for d in view.documents}

        print(f"\n{batch}: {len(view.queries)} JDs, {len(view.documents)} CVs")
        q_texts = [spec.query_text(q.text) for q in view.queries]
        d_texts = [render(spec, text_of[custom_of[i]], pool_names[custom_of[i]])
                   for i in doc_ids]
        vectors |= embed(q_texts, spec, loaded, embeddings, label=f"{batch} JDs")
        vectors |= embed(d_texts, spec, loaded, embeddings, label=f"{batch} pool CVs")

        Q = np.stack([vectors[rank_mod.text_key(t)] for t in q_texts])
        D = np.stack([vectors[rank_mod.text_key(t)] for t in d_texts])
        base = Q @ D.T                                    # (queries, documents)
        arrays[f"queries/{batch}"] = np.array([q.id for q in view.queries])

        base_plain = None
        if any(s is None for s in texts):
            plain_texts = [render(spec, text_of[custom_of[i]], None) for i in doc_ids]
            vectors |= embed(plain_texts, spec, loaded, embeddings,
                             label=f"{batch} unnamed pool CVs")
            D_plain = np.stack([vectors[rank_mod.text_key(t)] for t in plain_texts])
            base_plain = Q @ D_plain.T

        for r in (x for x in focal if x.batch_name == batch):
            rendered = [render(spec, r.profile_text, t) for t in texts]
            vectors |= embed(rendered, spec, loaded, embeddings,
                             label=f"{batch} {r.custom_id}")
            E = np.stack([vectors[rank_mod.text_key(t)] for t in rendered])
            scores = (E @ Q.T).astype(np.float32)          # (stimuli, queries)
            index = doc_ids.index(by_custom[r.custom_id])

            # Score spread of the competitors (focal column excluded), per vacancy.
            field = np.delete(base, index, axis=1)
            arrays[f"pool_sd/{r.custom_id}"] = field.std(1).astype(np.float32)
            arrays[f"scores/{r.custom_id}"] = scores
            arrays[f"ranks/{r.custom_id}"] = ranks_for(scores, base, index, doc_ids)

            # The unnamed candidate against an unnamed pool, stored as a reference.
            if base_plain is not None:
                row = texts.index(None)
                plain = (np.stack([vectors[rank_mod.text_key(rendered[row])]])
                         @ Q.T).astype(np.float32)
                arrays[f"plain_scores/{r.custom_id}"] = plain[0]
                arrays[f"plain_ranks/{r.custom_id}"] = ranks_for(
                    plain, base_plain, index, doc_ids)[0]
                arrays[f"plain_pool_sd/{r.custom_id}"] = np.delete(
                    base_plain, index, axis=1).std(1).astype(np.float32)

    summary = {
        "meta": {**meta, "stimuli": labels, "groups": groups,
                 "kinds": [s["kind"] for s in stim],
                 "pool_seed": POOL_SEED, "focal_seed": FOCAL_SEED,
                 "pool_assignment": pool_digest, "corpus": corpus_digest,
                 "batches": list(batches), "k": K, "focal": len(focal),
                 "lexically_inert": True, "environment": environment(),
                 "verification": {k: verification.get(k) for k in
                                  ("longest_tokens", "headroom", "sides")}},
        "bm25": bm25_result,
    }
    _write_atomic(results_path, lambda p: p.write_text(json.dumps(summary, indent=1) + "\n"))
    _write_atomic(ranks_path, lambda p: _save_npz(p, arrays))
    print(f"\nwrote {results_path} and {ranks_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", required=True,
                        help="slug from evaluation.baselines.encoders")
    parser.add_argument("--device", required=True,
                        help="cuda, mps or cpu; recorded, never inferred")
    parser.add_argument("--focal", type=int, default=FOCAL)
    parser.add_argument("--no-controls", action="store_true",
                        help="names only; drops the scale the identifiers supply")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(encoders.spec(args.encoder), args.device, focal_count=args.focal,
        with_controls=not args.no_controls, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
