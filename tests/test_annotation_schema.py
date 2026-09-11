import csv
from pathlib import Path

import numpy as np
import pytest

from annotate import (
    Candidate,
    ReviewLog,
    _image_object_rows,
    image_prompt,
)
from annotations import canonical_tool_label, extend_image_manifest
from masks import load_mask_bundle, save_mask_bundle


def test_manifest_extension_preserves_existing_cells_and_ids(tmp_path: Path) -> None:
    path = tmp_path / "images.csv"
    fields = ["Filepath", "Tool Name", "Image ID"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(
            {
                "Filepath": "images/a.jpg",
                "Tool Name": "  exact value  ",
                "Image ID": "reviewed-id",
            }
        )

    extend_image_manifest(path)

    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
        assert file.seek(0) == 0
        resulting_fields = next(csv.reader(file))
    assert rows[0]["Tool Name"] == "  exact value  "
    assert rows[0]["Image ID"] == "reviewed-id"
    assert resulting_fields == fields + ["Capture Session ID"]


def test_review_log_latest_decision_controls_resume(tmp_path: Path) -> None:
    review = ReviewLog(tmp_path / "review.jsonl")
    review.append("image-1", "skipped", prompt="tool")
    assert not review.completed("image-1")
    review.append("image-1", "accepted", prompt="tool")

    resumed = ReviewLog(review.path)
    assert resumed.completed("image-1")


def test_image_prompt_defaults_to_coarse_class() -> None:
    row = {"Tool Name": "iris", "Tool Class": "scissors"}

    assert image_prompt(row, "tool-class", {}) == "scissors"
    assert image_prompt(row, "name-and-class", {}) == "iris scissors"
    assert (
        image_prompt(row, "name-and-class", {"iris": "iris scissors"})
        == "iris scissors"
    )


def test_canonical_tool_label_combines_descriptor_and_class_once() -> None:
    assert canonical_tool_label("iris", "scissors", {}) == "iris scissors"
    assert canonical_tool_label("mayo scissors", "scissors", {}) == "mayo scissors"
    assert canonical_tool_label("mallet", "mallet", {}) == "mallet"


def test_single_reprompt_candidate_keeps_manifest_identity(tmp_path: Path) -> None:
    mask = np.ones((4, 5), dtype=np.bool_)
    rows = _image_object_rows(
        {
            "Image ID": "image-1",
            "Tool ID": "tool-1",
            "Tool Name": "iris",
            "Tool Class": "scissors",
        },
        [Candidate(mask, 0.9, "shears", "instance-0")],
        tmp_path / "mask.npz",
        "shears",
    )

    assert rows[0]["Tool ID"] == "tool-1"
    assert rows[0]["Tool Name"] == "iris"
    assert rows[0]["Tool Class"] == "scissors"


def test_multi_instance_bundle_round_trip_and_ignore(tmp_path: Path) -> None:
    masks = np.zeros((2, 9, 11), dtype=np.bool_)
    masks[0, 1:4, 2:8] = True
    masks[1, 5:8, 1:5] = True
    ignore = np.zeros((9, 11), dtype=np.bool_)
    ignore[0, :3] = True
    path = tmp_path / "frame.npz"

    save_mask_bundle(path, masks, ["left", "right"], ignore=ignore)
    loaded = load_mask_bundle(path)

    np.testing.assert_array_equal(loaded.masks, masks)
    np.testing.assert_array_equal(loaded.ignore, ignore)
    assert loaded.identities == ("left", "right")


def test_unversioned_mask_archive_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unversioned.npz"
    np.savez_compressed(path, packed=np.array([1], dtype=np.uint8), shape=(1, 1))

    with pytest.raises(ValueError, match="Unversioned"):
        load_mask_bundle(path)
