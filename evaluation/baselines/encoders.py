"""The three dense encoders of the name-replacement experiment (Qwen3, BGE-M3, EmbeddingGemma).

Each model uses its own pooling, prefixes and length limit; scoring is always cosine.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

TOKEN_BUDGET = 2048


@dataclass(frozen=True)
class Spec:
    """Everything that decides an embedding, other than device and dtype.
    ``slug`` names the vector cache directory."""

    slug: str
    model: str
    revision: str
    pooling: str            # "last" | "cls" | "mean"
    query_template: str     # must contain {text}
    document_template: str  # must contain {text}
    max_length: int
    trust_remote_code: bool = False
    # Apply the model's own sentence-transformers Dense layers between pooling
    # and normalisation (EmbeddingGemma needs this).
    published_projection: bool = False

    def query_text(self, jd_text: str) -> str:
        return self.query_template.format(text=jd_text)

    def document_text(self, cv_text: str) -> str:
        return self.document_template.format(text=cv_text)

    @property
    def padding_side(self) -> str:
        # Last-token pooling needs the final position to be a real token.
        return "left" if self.pooling == "last" else "right"


# Same configuration as evaluation.baselines.dense.
QWEN3 = Spec(
    slug="qwen3-embedding-0.6b",
    model="Qwen/Qwen3-Embedding-0.6B",
    revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    pooling="last",
    query_template=("Instruct: Given a job description, retrieve resumes that "
                    "satisfy the requirements\nQuery:{text}"),
    document_template="{text}",
    max_length=1280,
)

# Dense head only; the sparse and multi-vector heads are not used.
BGE_M3 = Spec(
    slug="bge-m3-dense",
    model="BAAI/bge-m3",
    revision="5617a9f61b028005a4858fdac845db406aefb181",
    pooling="cls",
    query_template="{text}",
    document_template="{text}",
    max_length=8192,
)

# Mean pooling is followed by two trained Dense maps (768 -> 3072 -> 768).
EMBEDDING_GEMMA = Spec(
    slug="embeddinggemma-300m",
    model="google/embeddinggemma-300m",
    revision="57c266a740f537b4dc058e1b0cda161fd15afa75",
    pooling="mean",
    query_template="task: search result | query: {text}",
    document_template="title: none | text: {text}",
    max_length=2048,
    published_projection=True,
)

PANEL = (QWEN3, BGE_M3, EMBEDDING_GEMMA)
BY_SLUG = {s.slug: s for s in PANEL}


def spec(slug: str) -> Spec:
    if slug not in BY_SLUG:
        raise SystemExit(f"unknown encoder {slug!r}; have {', '.join(BY_SLUG)}")
    return BY_SLUG[slug]


def load(spec: Spec, dtype: torch.dtype | None = None,
         device: str | torch.device | None = None):
    """The device is never inferred: a run always records what it ran on."""
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model, revision=spec.revision, padding_side=spec.padding_side,
        trust_remote_code=spec.trust_remote_code)
    model = AutoModel.from_pretrained(
        spec.model, revision=spec.revision, dtype=dtype,
        trust_remote_code=spec.trust_remote_code)
    if device is not None:
        model = model.to(device)
    model.eval()
    return tokenizer, model


def groups(lengths: Sequence[int], budget: int) -> list[list[int]]:
    """Indices sorted by length and grouped so padded width times batch size stays
    under ``budget``; a text longer than the budget forms its own group."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    out: list[list[int]] = []
    current: list[int] = []
    widest = 0
    for index in order:
        if current and (len(current) + 1) * max(lengths[index], widest) > budget:
            out.append(current)
            current, widest = [index], lengths[index]
        else:
            current.append(index)
            widest = max(widest, lengths[index])
    if current:
        out.append(current)
    return out


_PROJECTIONS: dict[str, torch.nn.Sequential] = {}


def _projection(spec: Spec) -> torch.nn.Sequential:
    """The model's sentence-transformers Dense stack at the pinned revision (cached).
    Only Identity-activation layers are supported."""
    if spec.slug in _PROJECTIONS:
        return _PROJECTIONS[spec.slug]
    import json

    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    def fetch(filename):
        return hf_hub_download(spec.model, filename, revision=spec.revision)

    with open(fetch("modules.json"), encoding="utf-8") as fh:
        modules = json.load(fh)
    layers = []
    for module in modules:
        if not module["type"].endswith(".Dense"):
            continue
        folder = module["path"]
        with open(fetch(f"{folder}/config.json"), encoding="utf-8") as fh:
            config = json.load(fh)
        if not config["activation_function"].endswith("Identity"):
            raise ValueError(
                f"{spec.slug}: {folder} uses {config['activation_function']}; "
                f"only Identity-activation Dense layers are implemented")
        weights = load_file(fetch(f"{folder}/model.safetensors"))
        linear = torch.nn.Linear(config["in_features"], config["out_features"],
                                 bias=config.get("bias", False))
        linear.weight.data = weights["linear.weight"]
        if config.get("bias", False):
            linear.bias.data = weights["linear.bias"]
        layers.append(linear)
    if not layers:
        raise ValueError(f"{spec.slug}: published_projection is set but the "
                         f"model ships no Dense modules")
    stack = torch.nn.Sequential(*layers).eval().requires_grad_(False)
    _PROJECTIONS[spec.slug] = stack
    return stack


def _pool(spec: Spec, hidden: Tensor, mask: Tensor) -> Tensor:
    if spec.pooling == "last":
        return hidden[:, -1]                      # padding is left
    if spec.pooling == "cls":
        return hidden[:, 0]
    if spec.pooling == "mean":
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
    raise ValueError(f"unknown pooling {spec.pooling!r}")


@torch.inference_mode()
def encode(texts: Sequence[str], spec: Spec, tokenizer, model,
           budget: int = TOKEN_BUDGET) -> Tensor:
    """L2-normalised embeddings, in the order the texts were given. Raises
    rather than truncating a text over ``spec.max_length``."""
    lengths = [len(tokenizer.encode(t)) for t in texts]
    over = [(n, t[:60]) for n, t in zip(lengths, texts) if n > spec.max_length]
    if over:
        raise ValueError(
            f"{spec.slug}: {len(over)} text(s) exceed max_length "
            f"{spec.max_length}; longest {max(n for n, _ in over)}. "
            f"Truncation would make name length a confound; raise the limit "
            f"or drop the spec.")
    out: list[Tensor | None] = [None] * len(texts)
    for group in groups(lengths, budget):
        batch = tokenizer([texts[i] for i in group], padding=True,
                          return_tensors="pt")
        batch = {k: v.to(model.device) for k, v in batch.items()}
        hidden = model(**batch).last_hidden_state
        pooled = _pool(spec, hidden, batch["attention_mask"])
        if spec.published_projection:
            pooled = _projection(spec).to(pooled.device, pooled.dtype)(pooled)
        pooled = F.normalize(pooled.float(), p=2, dim=1).cpu()
        if not torch.isfinite(pooled).all():
            raise ValueError("non-finite embedding; check dtype for overflow")
        for position, index in enumerate(group):
            out[index] = pooled[position]
        del batch, hidden
        if model.device.type == "mps":
            torch.mps.empty_cache()
        elif model.device.type == "cuda":
            torch.cuda.empty_cache()
    return torch.stack(out)  # type: ignore[arg-type]
