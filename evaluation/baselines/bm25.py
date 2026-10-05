"""Lexical baseline: BM25S (Lucene scoring) over the blinded per-batch view; raw scores.

All settings are passed explicitly, since BM25S's own defaults differ (k1=1.5).
"""

from collections.abc import Sequence

import bm25s

from ..blind import Document, Query

K1 = 1.2
B = 0.75
METHOD = "lucene"
LOWER = True
TOKEN_PATTERN = r"(?u)\b\w\w+\b"
STOPWORDS = "en"
STEMMER = None


def _tokens(texts: Sequence[str]) -> list[list[str]]:
    return bm25s.tokenize(
        texts,
        lower=LOWER,
        token_pattern=TOKEN_PATTERN,
        stopwords=STOPWORDS,
        stemmer=STEMMER,
        return_ids=False,
        show_progress=False,
    )


def rank(
    queries: Sequence[Query], documents: Sequence[Document]
) -> dict[str, dict[str, float]]:
    """Raw BM25 score for every query against every document."""
    retriever = bm25s.BM25(k1=K1, b=B, method=METHOD)
    retriever.index(_tokens([d.text for d in documents]), show_progress=False)
    return {
        query.id: {
            document.id: float(score)
            for document, score in zip(
                documents, retriever.get_scores(tokens), strict=True
            )
        }
        for query, tokens in zip(
            queries, _tokens([q.text for q in queries]), strict=True
        )
    }
