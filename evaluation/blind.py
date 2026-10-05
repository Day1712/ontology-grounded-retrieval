"""The ranker-facing view: opaque ids (numbered in sha256(custom_id) order) and the text.

Blinding runs once over the whole corpus; ``select`` takes a per-batch view.
"""

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass

from .corpus import Record


@dataclass(frozen=True)
class Query:
    id: str
    text: str


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True)
class Blinded:
    queries: list[Query]
    documents: list[Document]
    source: dict[str, str]  # blind id -> custom_id; evaluator side only


def _order_key(record: Record) -> str:
    # Not Python's hash(): that is randomised per process.
    return hashlib.sha256(record.custom_id.encode("utf-8")).hexdigest()


def blind(
    records: Iterable[Record], query_type: str = "jd", doc_type: str = "cv"
) -> Blinded:
    queries: list[Query] = []
    documents: list[Document] = []
    source: dict[str, str] = {}
    for record in sorted(records, key=_order_key):
        if record.profile_type == query_type:
            item_id = f"q{len(queries) + 1:06d}"
            queries.append(Query(item_id, record.profile_text))
        elif record.profile_type == doc_type:
            item_id = f"d{len(documents) + 1:06d}"
            documents.append(Document(item_id, record.profile_text))
        else:
            raise ValueError(f"unexpected profile_type {record.profile_type!r}")
        source[item_id] = record.custom_id
    return Blinded(queries, documents, source)


def select(blinded: Blinded, records: Iterable[Record]) -> Blinded:
    """The blinded view restricted to ``records``, keeping the global ids."""
    wanted = {record.custom_id for record in records}
    keep = {i for i, custom_id in blinded.source.items() if custom_id in wanted}
    return Blinded(
        queries=[q for q in blinded.queries if q.id in keep],
        documents=[d for d in blinded.documents if d.id in keep],
        source={i: c for i, c in blinded.source.items() if i in keep},
    )
