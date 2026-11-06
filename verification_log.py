"""Persist Codex visual reviews by their inputs, independently of sample numbering."""

import hashlib
import json
from pathlib import Path
from typing import TypedDict, cast

from annotations import atomic_write_text


class ReviewIssue(TypedDict):
    type: str
    filepath: str
    reason: str
    confidence: float


class SampleReview(TypedDict):
    sample_id: str
    status: str
    summary: str
    confidence: float
    issues: list[ReviewIssue]


class ReviewEntry(TypedDict):
    inputs: dict[str, object]
    reviewed_at: str
    result: dict[str, object]


class ReviewLog(TypedDict):
    version: int
    samples: dict[str, ReviewEntry]
    taxonomy: dict[str, ReviewEntry]


def fingerprint(inputs: dict[str, object]) -> str:
    encoded = json.dumps(inputs, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_review_log(path: Path) -> ReviewLog:
    if not path.exists():
        return {"version": 1, "samples": {}, "taxonomy": {}}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("version") != 1:
        raise ValueError("Unsupported visual review log format.")
    for section in ("samples", "taxonomy"):
        entries = value.get(section)
        if not isinstance(entries, dict):
            raise ValueError(f"Invalid visual review log section: {section}.")
        for key, entry in entries.items():
            if (
                not isinstance(key, str)
                or not isinstance(entry, dict)
                or not isinstance(entry.get("inputs"), dict)
                or not isinstance(entry.get("reviewed_at"), str)
                or not isinstance(entry.get("result"), dict)
                or fingerprint(entry["inputs"]) != key
            ):
                raise ValueError(f"Invalid visual review log entry in {section}.")
    return cast(ReviewLog, value)


def save_review_log(path: Path, log: ReviewLog) -> None:
    atomic_write_text(path, json.dumps(log, indent=2, ensure_ascii=False) + "\n")
