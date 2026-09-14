from annotations import read_csv, write_image_manifests


def test_navigation_uses_numeric_tool_ids_and_keeps_manual_sample_ids(tmp_path):
    manifest = tmp_path / "images.csv"
    object_manifest = tmp_path / "objects.csv"
    images = [
        {"Image ID": "a", "Sample ID": "10", "Angle": "90", "Filepath": "images/first.jpg"},
        {"Image ID": "b", "Sample ID": "2", "Angle": "45", "Filepath": "images/second.jpg"},
        {"Image ID": "c", "Sample ID": "10", "Angle": "0", "Filepath": "images/third.jpg"},
    ]
    objects = [
        {"Image ID": "a", "Instance ID": "0", "Tool ID": "12", "Mask Path": "masks/first.npz"},
        {"Image ID": "a", "Instance ID": "1", "Tool ID": "2"},
        {"Image ID": "b", "Instance ID": "0", "Tool ID": "4"},
        {"Image ID": "c", "Instance ID": "0", "Tool ID": "12"},
    ]
    write_image_manifests(manifest, images, object_manifest, objects)
    rows = read_csv(manifest)
    instances = read_csv(object_manifest)
    assert list(rows[0])[:5] == ["Tool IDs", "Sample ID", "Angle", "Filepath", "Lights"]
    assert [row["Image ID"] for row in rows] == ["a", "b", "c"]
    assert rows[0]["Tool IDs"] == "2;12"
    assert [row["Image ID"] for row in instances] == ["a", "a", "b", "c"]
    assert instances[1]["Mask Path"] == "masks/first.npz"
    assert instances[1]["Sample ID"] == "10"
    assert instances[1]["Filepath"] == "images/first.jpg"
    rows[1]["Sample ID"] = "50"
    instances[2]["Tool ID"] = "99"
    write_image_manifests(manifest, rows, object_manifest, instances)
    assert read_csv(manifest)[-1]["Tool IDs"] == "99"
    assert read_csv(object_manifest)[-1]["Sample ID"] == "50"


def test_verifier_reports_stale_navigation_columns(tmp_path, monkeypatch):
    import verify_data

    monkeypatch.setattr(verify_data, "ROOT", tmp_path)
    monkeypatch.setattr(verify_data, "IMAGES_DIR", tmp_path / "images")
    image = {
        "Image ID": "a", "Tool IDs": "12", "Sample ID": "", "Angle": "",
        "Filepath": "images/2/a.jpg", "Time Taken": "2026-09-01",
    }
    obj = {
        "Image ID": "a", "Instance ID": "0", "Tool ID": "2",
        "Tool Class": "scissors", "Tool Name": "iris", "Confidence": "1",
        "Sample ID": "old", "Filepath": "images/old.jpg",
    }
    issues, _ = verify_data.validate_manifest([image], [obj])
    codes = [issue["code"] for issue in issues]
    assert "stale_tool_summary" in codes
    assert codes.count("stale_instance_reference") == 2


def test_unfinished_images_go_last_with_stable_tool_order(tmp_path):
    images = []
    objects = []
    entries = [
        ("a", "2", "", "45"),
        ("b", "12", "1", "90"),
        ("c", "2", "1", "90"),
        ("d", "2", "1", "0"),
        ("e", "12", "", ""),
        ("f", "2", "1", ""),
        ("g", "2", " ", " "),
    ]
    for key, tool, sample, angle in entries:
        images.append({
            "Image ID": key, "Sample ID": sample, "Angle": angle,
            "Filepath": f"images/{key}.jpg",
        })
        objects.append({"Image ID": key, "Instance ID": "0", "Tool ID": tool})
    manifest = tmp_path / "images.csv"
    object_manifest = tmp_path / "objects.csv"
    write_image_manifests(manifest, images, object_manifest, objects)
    assert [row["Image ID"] for row in read_csv(manifest)] == [
        "c", "d", "b", "a", "f", "g", "e",
    ]
