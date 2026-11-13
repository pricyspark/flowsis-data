"""Shared frame-level export schema for base and temporal training."""

from typing import TypedDict

from datasets import ClassLabel, Features, Sequence, Value


class ObjectRecord(TypedDict):
    instance_id: str
    area: float
    bbox: list[int]
    category: int
    tool_name: str
    tool_class: str
    tool_id: str
    mask_path: str
    mask_index: int
    annotation_confidence: float
    track_id: str
    visibility: str
    mask_valid: bool
    class_valid: bool
    base_valid: bool
    visible: bool
    preferred_track: bool


class ImageRecord(TypedDict):
    image_path: str
    video_path: str
    frame_index: int
    image_id: str
    height: int
    width: int
    tool_ids: list[str]
    sample_id: str
    capture_session_id: str
    objects: list[ObjectRecord]
    source_type: str
    video_id: str
    sequence_id: str
    timestamp_seconds: float
    mask_bundle_path: str
    base_eligible: bool


LABEL_FIELDS = ("tool_name", "tool_class")


def object_label(obj: ObjectRecord, fields: tuple[str, ...]) -> str:
    """Join selected metadata fields while avoiding repeated name suffixes."""
    values = {"tool_name": obj["tool_name"], "tool_class": obj["tool_class"]}
    label = ""
    for field in fields:
        value = " ".join(values[field].split())
        if value and not (
            label.casefold() == value.casefold()
            or label.casefold().endswith(f" {value.casefold()}")
        ):
            label = f"{label} {value}".strip()
    if not label:
        raise ValueError(f"Selected label fields are empty for {obj['instance_id']}.")
    return label


def dataset_features(names: list[str]) -> Features:
    return Features(
        {
            "source_type": Value("string"),
            "video_id": Value("string"),
            "sequence_id": Value("string"),
            "timestamp_seconds": Value("float64"),
            "mask_bundle_path": Value("string"),
            "base_eligible": Value("bool"),
            "image_path": Value("string"),
            "video_path": Value("string"),
            "frame_index": Value("int64"),
            "image_id": Value("string"),
            "height": Value("int64"),
            "width": Value("int64"),
            "tool_ids": Sequence(Value("string")),
            "sample_id": Value("string"),
            "capture_session_id": Value("string"),
            "objects": [
                {
                    "track_id": Value("string"),
                    "visibility": Value("string"),
                    "mask_valid": Value("bool"),
                    "class_valid": Value("bool"),
                    "base_valid": Value("bool"),
                    "visible": Value("bool"),
                    "preferred_track": Value("bool"),
                    "instance_id": Value("string"),
                    "area": Value("float32"),
                    "bbox": Sequence(Value("float32"), length=4),
                    "category": ClassLabel(names=names),
                    "tool_name": Value("string"),
                    "tool_class": Value("string"),
                    "tool_id": Value("string"),
                    "mask_path": Value("string"),
                    "mask_index": Value("int64"),
                    "annotation_confidence": Value("float32"),
                }
            ],
        }
    )
