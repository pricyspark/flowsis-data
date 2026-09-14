import csv
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import annotate
from annotations import IMAGE_OBJECT_FIELDS, read_csv, stable_media_id, write_csv
from masks import load_mask_bundle


@pytest.mark.parametrize("empty", [False, True])
def test_image_cli_writes_local_outputs_and_resumes(tmp_path, monkeypatch, empty):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["annotate.py", "images", "--auto-accept"])
    Path("images/7").mkdir(parents=True)
    Image.new("RGB", (12, 10)).save("images/7/tool.jpg")
    with Path("image_manifest.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["Filepath", "Image ID"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "Filepath": "images/7/tool.jpg",
                "Image ID": stable_media_id("image", "images/7/tool.jpg"),
            }
        )
    write_csv("image_object_manifest.csv", [{
        "Image ID": stable_media_id("image", "images/7/tool.jpg"),
        "Instance ID": "declared-0", "Tool ID": "7", "Tool Name": "iris",
        "Tool Class": "scissors", "Confidence": "0.8",
    }], IMAGE_OBJECT_FIELDS)
    calls = []

    class Backend:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, images, prompts):
            calls.append(prompts)
            mask = np.zeros((10, 12), dtype=np.bool_)
            mask[2:8, 3:9] = True
            candidates = [] if empty else [
                annotate.Candidate(mask, 0.9, prompts[0], "instance-0")
            ]
            return [candidates]

    monkeypatch.setattr(annotate, "Sam3ImageBackend", Backend)
    annotate.main()
    row = read_csv("image_object_manifest.csv")[0]
    bundle = load_mask_bundle("masks/images/7/tool.npz")
    assert row["Confidence"] == "0.8"
    assert row["Instance ID"] == "declared-0"
    if empty:
        assert bundle.identities == ()
        assert bundle.masks.shape == (0, 10, 12)
        assert row["Mask Path"] == ""
        assert row["Reviewed"] == ""
    else:
        assert bundle.identities == ("declared-0",)
        assert bundle.masks.shape == (1, 10, 12)
        assert Path(row["Mask Path"]) == Path("masks/images/7/tool.npz")
    assert row["Image ID"] == read_csv("image_manifest.csv")[0]["Image ID"]
    assert annotate.ReviewLog(Path("image_review.jsonl")).completed(row["Image ID"])
    annotate.main()
    assert calls == [["scissors"]]


@pytest.mark.parametrize("skipped_first", [False, True])
def test_skipped_priority_is_stable_across_batches(tmp_path, monkeypatch, skipped_first):
    monkeypatch.chdir(tmp_path)
    argv = ["annotate.py", "images", "--batch-size", "2"]
    if skipped_first:
        argv.append("--skipped-first")
    monkeypatch.setattr(sys, "argv", argv)
    Path("images").mkdir()
    keys = ["skip-a", "new-a", "accepted", "skip-b", "new-b", "rejected", "missing"]
    rows = [{"Image ID": key, "Filepath": f"images/{key}.jpg"} for key in keys]
    for row in rows[:-1]:
        Image.new("RGB", (8, 8)).save(row["Filepath"])
    write_csv("image_manifest.csv", rows, ["Image ID", "Filepath"])
    objects = [{
        "Image ID": key, "Instance ID": "instance-0", "Tool ID": "7",
        "Tool Class": "scissors", "Tool Name": "iris", "Confidence": "1",
    } for key in keys]
    write_csv("image_object_manifest.csv", objects, IMAGE_OBJECT_FIELDS)
    log = annotate.ReviewLog(Path("image_review.jsonl"))
    for key in ("skip-a", "skip-b", "accepted"):
        log.append(key, "skipped")
    log.append("accepted", "accepted")
    log.append("rejected", "rejected")
    batches = []

    class Backend:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, images, prompts):
            batches.append(len(images))
            return [[] for _ in images]

    seen = []

    def review(image, candidates, *, key, **kwargs):
        seen.append(key)
        return "skipped", [], [], None, "clear"

    monkeypatch.setattr(annotate, "Sam3ImageBackend", Backend)
    monkeypatch.setattr(annotate, "review_candidates", review)
    annotate.main()
    expected = (
        ["skip-a", "skip-b", "new-a", "new-b"] if skipped_first
        else ["new-a", "new-b", "skip-a", "skip-b"]
    )
    assert seen == expected
    assert batches == [2, 2]
    assert [row["Image ID"] for row in read_csv("image_manifest.csv")] == keys
