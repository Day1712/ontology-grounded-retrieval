"""Loads the three benchmark files as-is; structural violations raise with file and line."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic_resumes"
FILES = ("final_oneitem.jsonl", "final_twoitem.jsonl", "final_threeitem.jsonl")
FIELDS: dict[str, type] = {
    "custom_id": str,
    "occ_code": str,
    "batch_name": str,
    "style": str,
    "profile_type": str,
    "profile_text": str,
    "profile_dict": dict,
}


@dataclass(frozen=True)
class Record:
    custom_id: str
    occ_code: str
    batch_name: str
    style: str
    profile_type: str
    profile_text: str
    profile_dict: dict[str, Any]


def load_corpus(data_dir: Path = DATA_DIR) -> list[Record]:
    """All three files in batch order. Raises ValueError on a bad line, field or duplicate id."""
    records: list[Record] = []
    seen: set[str] = set()
    for name in FILES:
        with (data_dir / name).open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                where = f"{name}:{lineno}"
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{where}: invalid JSON: {exc}") from exc
                for field, type_ in FIELDS.items():
                    if field not in obj:
                        raise ValueError(f"{where}: missing field {field!r}")
                    if not isinstance(obj[field], type_):
                        raise ValueError(f"{where}: {field!r} is not {type_.__name__}")
                if obj["custom_id"] in seen:
                    raise ValueError(f"{where}: duplicate custom_id {obj['custom_id']!r}")
                seen.add(obj["custom_id"])
                records.append(Record(**{f: obj[f] for f in FIELDS}))
    return records
