import csv
import sys
from pathlib import Path

import numpy as np
from PIL import Image

import annotate
from annotations import read_csv
from masks import load_mask_bundle


def test_image_cli_writes_local_outputs_and_resumes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["annotate.py", "images", "--auto-accept"])
    Path("images").mkdir()
    Image.new("RGB", (12, 10)).save("images/tool.jpg")
    with Path("image_manifest.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file, fieldnames=["Filepath", "Tool ID", "Tool Name", "Tool Class"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "Filepath": "images/tool.jpg",
                "Tool ID": "1",
                "Tool Name": "iris",
                "Tool Class": "scissors",
            }
        )
    calls = []

    class Backend:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, images, prompts):
            calls.append(prompts)
            mask = np.zeros((10, 12), dtype=np.bool_)
            mask[2:8, 3:9] = True
            return [[annotate.Candidate(mask, 0.9, prompts[0], "instance-0")]]

    monkeypatch.setattr(annotate, "Sam3ImageBackend", Backend)
    annotate.main()
    row = read_csv("image_object_manifest.csv")[0]
    bundle = load_mask_bundle(row["Mask Path"])
    assert bundle.masks.shape == (1, 10, 12)
    assert Path(row["Mask Path"]).parent == Path("masks/images")
    assert row["Image ID"] == read_csv("image_manifest.csv")[0]["Image ID"]
    assert annotate.ReviewLog(Path("image_review.jsonl")).completed(row["Image ID"])
    annotate.main()
    assert calls == [["scissors"]]
