"""Generates the 20 arbitrary identifiers (matched in token length to the names,
absent from the corpus) once into data/names/arbitrary_ids.csv."""

import argparse
import csv
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "names" / "arbitrary_ids.csv"
SEED = 20240719
COUNT = 20
ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no I/O/0/1: unambiguous in print
WORD = re.compile(r"[a-z0-9]+")


def corpus_tokens():
    from evaluation import corpus
    seen = set()
    for r in corpus.load_corpus():
        seen.update(WORD.findall(r.profile_text.lower()))
    return seen


def candidates(rng, n=4000):
    for _ in range(n):
        a = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(2, 4)))
        b = "".join(rng.choice(ALPHABET) for _ in range(rng.randint(2, 4)))
        yield f"{a} {b}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", default="qwen3-embedding-0.6b",
                        help="reference encoder whose tokenizer sets the target lengths")
    parser.add_argument("--count", type=int, default=COUNT)
    args = parser.parse_args()

    from transformers import AutoTokenizer
    from evaluation.baselines import encoders
    from experiments.names import SEPARATOR, load_names

    spec = encoders.spec(args.encoder)
    tok = AutoTokenizer.from_pretrained(spec.model, revision=spec.revision)
    length = lambda s: len(tok.encode(f"{s}{SEPARATOR}"))

    names = [n["FullName"] for n in load_names()]
    target: dict[int, int] = {}
    for n in names:
        target[length(n)] = target.get(length(n), 0) + 1
    scale = args.count / len(names)
    want = {k: max(1, round(v * scale)) for k, v in target.items()}
    print(f"name lengths {dict(sorted(target.items()))} over {len(names)} names")
    print(f"target for {args.count} identifiers: {dict(sorted(want.items()))}")

    inert = corpus_tokens()
    rng = random.Random(SEED)
    chosen: list[str] = []
    have: dict[int, int] = {}
    for cand in candidates(rng):
        parts = cand.lower().split()
        if any(p in inert for p in parts) or cand in chosen:
            continue
        n = length(cand)
        if have.get(n, 0) >= want.get(n, 0):
            continue
        chosen.append(cand)
        have[n] = have.get(n, 0) + 1
        if sum(have.values()) >= sum(want.values()):
            break
    if sum(have.values()) < sum(want.values()):
        raise SystemExit(f"only found {sum(have.values())} of {sum(want.values())}")

    print(f"selected {len(chosen)}: {dict(sorted(have.items()))}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["FullName", "group", "kind", "tokens", "seed", "reference_encoder"])
        for c in sorted(chosen):
            w.writerow([c, "ID", "arbitrary_identifier", length(c), SEED, spec.slug])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
