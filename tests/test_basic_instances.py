from pathlib import Path

from PIL import Image

import basic
from annotations import IMAGE_OBJECT_FIELDS, read_csv, write_csv


def test_basic_creates_instances_reuses_metadata_and_preserves_annotations(
    tmp_path, monkeypatch
):
    images = tmp_path / "images"
    raw = tmp_path / "raw-images"
    (images / "7").mkdir(parents=True)
    (raw / "7").mkdir(parents=True)
    manifest = tmp_path / "image_manifest.csv"
    object_manifest = tmp_path / "image_object_manifest.csv"
    fields = [
        "Image ID", "Capture Session ID", "Filepath", "Sample ID", "Angle", "Lights",
        *basic.METADATA_COLUMNS,
    ]
    write_csv(manifest, [], fields)
    write_csv(object_manifest, [], IMAGE_OBJECT_FIELDS)
    metadata = dict.fromkeys(basic.METADATA_COLUMNS, "known")
    metadata["Time Taken"] = "2026-09-01T12:00:00-04:00"
    monkeypatch.setattr(basic, "extract_metadata", lambda path: metadata)
    answers = iter(["Scissors", "Iris", "0.8"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(answers))
    for index in range(8):
        Image.new("RGB", (8, 8)).save(images / "7" / f"{index:02}.jpg")
        Image.new("RGB", (8, 8)).save(raw / "7" / f"{index:02}.jpg")
    basic.main(images, raw, manifest)
    rows = read_csv(manifest)
    objects = read_csv(object_manifest)
    assert len(rows) == len(objects) == 8
    assert not {"Tool ID", "Tool Class", "Tool Name", "Confidence"} & rows[0].keys()
    assert {row["Angle"] for row in rows} == {str(angle) for angle in basic.ANGLES}
    assert all(row["Capture Session ID"] for row in rows)
    assert all(obj["Tool ID"] == "7" and obj["Confidence"] == "0.8" for obj in objects)
    assert all(not obj["Reviewed"] and not obj["Mask Path"] for obj in objects)
    objects[0].update({
        "Mask Path": "masks/images/7/00.npz",
        "Reviewed": "TRUE",
        "Annotation Confidence": "0.9",
    })
    write_csv(object_manifest, objects, IMAGE_OBJECT_FIELDS)
    preserved = dict(objects[0])
    for index in range(8, 16):
        Image.new("RGB", (8, 8)).save(images / "7" / f"{index:02}.jpg")
        Image.new("RGB", (8, 8)).save(raw / "7" / f"{index:02}.jpg")
    answers = iter(["y"])
    basic.main(images, raw, manifest)
    updated = read_csv(object_manifest)
    assert len(read_csv(manifest)) == len(updated) == 16
    assert updated[0] == preserved
    assert all(
        obj["Tool Name"] == "iris" and obj["Confidence"] == "0.8"
        for obj in updated
    )
    edited = read_csv(manifest)
    for row in edited[:8]:
        row["Sample ID"] = "50"
    edited[0]["Angle"], edited[1]["Angle"] = edited[1]["Angle"], edited[0]["Angle"]
    write_csv(manifest, edited, edited[0].keys())
    basic.main(images, raw, manifest)
    reordered = read_csv(manifest)
    assert reordered[0]["Sample ID"] == "50"
    assert reordered[0]["Filepath"].endswith("00.jpg")
    assert reordered[0]["Angle"] == "45"
    assert read_csv(object_manifest)[0]["Sample ID"] == "50"
    assert all(row["Tool IDs"] == "7" for row in reordered)
    before = (manifest.read_bytes(), object_manifest.read_bytes())
    basic.main(images, raw, manifest)
    assert before == (manifest.read_bytes(), object_manifest.read_bytes())
