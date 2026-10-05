"""Dense baseline: Qwen3-Embedding-0.6B (pretrained), last-token pooled, cosine scores.

Pass the dtype explicitly: unset, transformers loads the checkpoint's bfloat16.
"""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer

from ..blind import Document, Query

MODEL = "Qwen/Qwen3-Embedding-0.6B"
REVISION = "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
INSTRUCTION = "Given a job description, retrieve resumes that satisfy the requirements"
MAX_LENGTH = 1280  # above the longest query/document; nothing truncates
TOKEN_BUDGET = 2048  # padded tokens per batch; affects memory only


def query_text(jd_text: str) -> str:
    """The model card's query template (no space after "Query:")."""
    return f"Instruct: {INSTRUCTION}\nQuery:{jd_text}"


def load(dtype: torch.dtype | None = None, device: str | torch.device | None = None):
    """Tokenizer and model; the device is never inferred."""
    # Left padding makes position -1 a real token (the appended <|endoftext|>).
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, padding_side="left"
    )
    model = AutoModel.from_pretrained(MODEL, revision=REVISION, dtype=dtype)
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


@torch.inference_mode()
def encode(texts: Sequence[str], tokenizer, model, budget: int = TOKEN_BUDGET) -> Tensor:
    """L2-normalised embeddings, in the order the texts were given."""
    lengths = [
        len(tokenizer.encode(t, truncation=True, max_length=MAX_LENGTH)) for t in texts
    ]
    out: list[Tensor | None] = [None] * len(texts)
    for group in groups(lengths, budget):
        batch = tokenizer(
            [texts[i] for i in group],
            max_length=MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {k: v.to(model.device) for k, v in batch.items()}
        hidden = model(**batch).last_hidden_state
        pooled = F.normalize(hidden[:, -1].float(), p=2, dim=1).cpu()
        if not torch.isfinite(pooled).all():
            raise ValueError("non-finite embedding; check dtype for overflow")
        for position, index in enumerate(group):
            out[index] = pooled[position]
        del batch, hidden
        if model.device.type == "mps":
            torch.mps.empty_cache()
    return torch.stack(out)  # type: ignore[arg-type]


def rank(
    queries: Sequence[Query],
    documents: Sequence[Document],
    loaded=None,
    token_budget: int = TOKEN_BUDGET,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> dict[str, dict[str, float]]:
    """Cosine score for every query against every document. Pass ``loaded`` from
    :func:`load` to reuse the model across batches."""
    tokenizer, model = loaded if loaded is not None else load(dtype, device)
    q = encode([query_text(x.text) for x in queries], tokenizer, model, token_budget)
    d = encode([x.text for x in documents], tokenizer, model, token_budget)
    scores = q @ d.T
    return {
        query.id: {
            document.id: float(score)
            for document, score in zip(documents, row, strict=True)
        }
        for query, row in zip(queries, scores, strict=True)
    }
