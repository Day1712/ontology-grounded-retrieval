"""Verifies each encoder spec before the name-replacement run may use it.

Compares this harness's embeddings with the model's own SentenceTransformer
configuration and checks that no corpus text, with the longest name prepended,
exceeds ``max_length``. Writes ``cache/encoder_verification.json``, which
``experiments.invariance`` requires.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluation import corpus
from evaluation.baselines import encoders
from experiments.names import load_names, named

ROOT = Path(__file__).resolve().parent.parent
RECORD = ROOT / "cache" / "encoder_verification.json"
TOLERANCE = 1e-4
PROBE_TEXTS = (
    "Five years coordinating maintenance schedules for a fleet of service vehicles.",
    "Wrote and maintained Python tooling for a data ingestion pipeline.",
    "Seeking a warehouse supervisor able to plan shifts and track inventory.",
)


def _harness(spec, texts, side, loaded):
    tokenizer, model = loaded
    render = spec.query_text if side == "query" else spec.document_text
    return encoders.encode([render(t) for t in texts], spec, tokenizer, model).numpy()


def _sentence_transformers(spec, device):
    """The publisher's own configuration, in float32 to match verify()
    (Qwen3's config would otherwise load bfloat16)."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(spec.model, revision=spec.revision, device=device,
                               model_kwargs={"dtype": torch.float32},
                               trust_remote_code=spec.trust_remote_code)


def _st_encode(st, spec, texts, side):
    """Encodes the spec's rendered texts; also returns the shipped prompt and
    whether the spec's prefix matches it."""
    prompts = getattr(st, "prompts", {}) or {}
    wanted = "query" if side == "query" else ("document" if "document" in prompts
                                              else "passage")
    shipped = prompts.get(wanted)
    render = spec.query_text if side == "query" else spec.document_text
    rendered = [render(t) for t in texts]
    vectors = st.encode(rendered, prompt_name=None, normalize_embeddings=True,
                        show_progress_bar=False)
    matches = all(r == (shipped or "") + t for r, t in zip(rendered, texts))
    return np.asarray(vectors), shipped, matches


def verify(spec, device, dtype=torch.float32):
    """Encodes with this harness, frees it, then encodes with SentenceTransformer
    and compares; the two models are never loaded at once."""
    loaded = encoders.load(spec, dtype=dtype, device=device)
    tokenizer, model = loaded
    resolved = getattr(model.config, "_commit_hash", None) or spec.revision
    out = {"model": spec.model, "revision": spec.revision,
           "resolved_revision": resolved, "pinned": spec.revision == resolved,
           "pooling": spec.pooling, "max_length": spec.max_length,
           "device": str(device), "dtype": str(dtype), "sides": {}}

    mine = {side: _harness(spec, PROBE_TEXTS, side, loaded)
            for side in ("query", "document")}

    # The longest text the run will hand this encoder, stimulus included.
    records = corpus.load_corpus()
    stimulus = max((n["FullName"] for n in load_names()), key=len)
    docs = [spec.document_text(named(r.profile_text, stimulus))
            for r in records if r.profile_type == "cv"]
    queries = [spec.query_text(r.profile_text) for r in records
               if r.profile_type == "jd"]
    out["longest_tokens"] = {
        "document": max(len(tokenizer.encode(t)) for t in docs),
        "query": max(len(tokenizer.encode(t)) for t in queries)}
    out["headroom"] = spec.max_length - max(out["longest_tokens"].values())
    out["no_truncation"] = out["headroom"] > 0

    del model, loaded, tokenizer
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    elif str(device).startswith("mps"):
        torch.mps.empty_cache()

    st = _sentence_transformers(spec, device)
    for side in ("query", "document"):
        theirs, shipped, matches = _st_encode(st, spec, PROBE_TEXTS, side)
        agreement = float((mine[side] * theirs).sum(1).min())
        out["sides"][side] = {
            "min_cosine_agreement": agreement,
            "st_shipped_prompt": shipped,
            "prefix_matches_published": matches,
            "agrees": agreement > 1 - TOLERANCE,
        }
    del st

    out["verified"] = out["no_truncation"] and all(
        s["agrees"] for s in out["sides"].values())
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True,
                        help="cuda, mps or cpu; recorded, never inferred")
    parser.add_argument("--encoder", action="append", default=None,
                        help="slug; repeatable. Default: the whole panel.")
    args = parser.parse_args()

    specs = ([encoders.spec(s) for s in args.encoder] if args.encoder
             else list(encoders.PANEL))
    stored = json.loads(RECORD.read_text()) if RECORD.exists() else {}
    for spec in specs:
        print(f"\n{spec.slug}")
        result = verify(spec, args.device)
        stored[spec.slug] = result
        for side, s in result["sides"].items():
            note = ("" if s["prefix_matches_published"]
                    else "  (prefix is the spec's own, not the published prompt)")
            print(f"  {side:9} cosine vs SentenceTransformer "
                  f"{s['min_cosine_agreement']:.6f}{note}")
        print(f"  longest {result['longest_tokens']}, limit {spec.max_length}, "
              f"headroom {result['headroom']}")
        if not result["pinned"]:
            print(f"  revision {spec.revision!r} resolved to "
                  f"{result['resolved_revision']} -- pin that in the spec before "
                  f"the run, or a later reader cannot reproduce it")
        print(f"  {'VERIFIED' if result['verified'] else 'FAILED'}")

    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(stored, indent=1) + "\n")
    print(f"\nwrote {RECORD}")
    if not all(stored[s.slug]["verified"] for s in specs):
        raise SystemExit("at least one spec failed; the run will refuse it")


if __name__ == "__main__":
    main()
