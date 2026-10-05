# Ontology-grounded CV–vacancy retrieval: thesis code

Code for the ranking method and the baselines of my master thesis. The repository holds the method, the evaluation, the experiment scripts and the data they read. 

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Run all commands from the repository root with `.venv/bin/python`, shown below as `python`. `requirements.txt` pins the versions the results were produced with (Python 3.13, macOS, Apple M3, device `mps`); use `cuda` or `cpu` on other machines.

Grounding and the dense retrievers download models from Hugging Face on first use: `intfloat/e5-base-v2` for grounding, and `Qwen/Qwen3-Embedding-0.6B`, `BAAI/bge-m3` and `google/embeddinggemma-300m` as dense baselines.

## Layout

| folder | contents |
|---|---|
| `ranking_method/` | the method: grounding, RDF representation, grading, scoring, PROV-O records |
| `evaluation/` | corpus, gold scores, nDCG@10, concordance C_q, BM25 and dense baselines |
| `experiments/` | runs that produce results, written to `cache/` |
| `checks/` | conformance of the RDF graph, PROV-O record check, encoder verification |
| `data/` | corpus, O*NET, ontologies, name lists |
| `cache/` | cached grounding and the result files the thesis reports |

## Commands

Grounding is cached in `cache/items.json`; with it present every command except the dense runs finishes in seconds to minutes. Delete it to redo grounding from scratch (needs `e5-base-v2`).

### Method and baselines

| item | command | output |
|---|---|---|
| Grounding and the method vs BM25 | `python -m experiments.benchmark` | `cache/items.json`, `cache/results.json` |
| Grid search over grounding thresholds (54 settings) | `python -m experiments.thresholds` | `cache/similarity.npz`, `cache/thresholds.json` |
| Grounding F1 per branch and channel | `python -m experiments.grounding_quality` | `cache/grounding_quality.json` |
| Occupation-held-out selection (200 runs) | `python -m experiments.heldout_occupations grid --case none`, the same with `--case skills+knowledge` and `--case all`, then `python -m experiments.heldout_occupations splits` | `cache/heldout/` |
| Occupation-held-out ranking | `python -m experiments.heldout_occupations rank` | `cache/heldout/ranked.json` |
| Ablation variants and their intervals | `python -m experiments.ablation`; `python -m experiments.ablation_flat` | `cache/ablation.json`, `cache/ablation_flat.json` |
| Grading coverage, with and without grading | `python -m experiments.grading_quality` | `cache/grading_quality.json` |
| Concept-BM25 | `python -m experiments.concept_bm25 oneitem twoitem threeitem` | `cache/concept_bm25.json` |
| Dense retrievers (vectors) | see "Dense retrievers" below | `cache/invariance/embeddings/` |
| Qwen3 dense run (cross-check) | `python -m experiments.dense_baseline` | `cache/dense_results.json` |
| RDF graph and domain/range check | `python -m ranking_method.emit`; `python -m checks.conformance cache/graph.ttl` | `cache/graph.ttl`, `cache/conformance.json` |
| PROV-O records and their check | `python -m ranking_method.record`; `python -m checks.record cache/decisions.ttl` | `cache/decisions.ttl`, `cache/record_checks.json` |

### Name replacement (robustness analysis)

| item | command | output |
|---|---|---|
| Arbitrary identifiers | `python -m experiments.make_controls` (run once; the output is included) | `data/names/arbitrary_ids.csv` |
| Dense runs under 99 stimuli | see below | `cache/invariance/<encoder>.json`, `_ranks.npz` |
| Score sensitivity report | `python -m experiments.invariance_report --encoder qwen3-embedding-0.6b --encoder bge-m3-dense --encoder embeddinggemma-300m` | `cache/invariance/report.json` |
| Name probe on the method (Ada, Pascal, Ruby) | `python -m experiments.names_method` | `cache/names/probe_method.json` |

### Dense retrievers

The only long step: about 1.2 hours per encoder on a laptop GPU. Vectors are cached per text, so an interrupted run can be started again.

```bash
python -m checks.encoders --device mps          # verifies the three encoders
for e in qwen3-embedding-0.6b bge-m3-dense embeddinggemma-300m; do
  python -m experiments.invariance --encoder $e --device mps
done
```

## Third-party data

- `data/onet/`, `data/onet_rdf/` — O*NET 30.3 Database by the U.S. Department of Labor, Employment and Training Administration, used under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) license. O*NET® is a trademark of USDOL/ETA.
- `data/ontology/coreo.ttl` — Core-O competence reference ontology, CC BY 4.0.
- `data/ontology/gufo.ttl` — gUFO, CC BY 4.0.
- `data/ontology/cnlo.ttl` — CompetentNL ontology (linkeddata.competentnl.nl).
- `data/ontology/prov.ttl`, `data/ontology/oa.ttl` — W3C PROV-O and Web Annotation vocabularies, W3C Software and Document License.
- `data/ontology/alignment.ttl`, `ext.ttl`, `grading.ttl` — written for this thesis.
- `data/names/wilson_caliskan_names.csv` — name stimuli from Wilson & Caliskan (AIES 2024); see `data/names/README.md`.
- `data/synthetic_resumes/` — synthetic CV–vacancy corpus used for evaluation, generated by Tünde de Vries.
