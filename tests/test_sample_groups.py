import sys

import pytest
from PIL import Image

import prepare_manifests
import verify_data
from annotations import read_csv, write_csv, write_image_manifests


def sample_manifests(tmp_path, monkeypatch, tool_sets, angles=None):
    monkeypatch.setattr(verify_data, "ROOT", tmp_path)
    monkeypatch.setattr(verify_data, "IMAGES_DIR", tmp_path / "images")
    images, objects = [], []
    for index, tool_ids in enumerate(tool_sets):
        key = f"image-{index}"
        filepath = f"images/multi/setup/{index}.jpg"
        path = tmp_path / filepath
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(path)
        images.append(
            {
                "Image ID": key,
                "Sample ID": "7",
                "Angle": angles[index] if angles else "",
                "Filepath": filepath,
                "Capture Session ID": "session-1",
                **dict.fromkeys(prepare_manifests.METADATA_COLUMNS, ""),
                "Time Taken": "2026-10-01T10:00:00-04:00",
            }
        )
        for tool_id in tool_ids:
            objects.append(
                {
                    "Image ID": key,
                    "Instance ID": f"{key}-{tool_id}",
                    "Tool ID": tool_id,
                    "Tool Class": "scissors",
                    "Tool Name": f"tool-{tool_id}",
                    "Confidence": "0.9",
                }
            )
    manifest = tmp_path / "image_manifest.csv"
    object_manifest = tmp_path / "image_object_manifest.csv"
    write_image_manifests(manifest, images, object_manifest, objects)
    return manifest, object_manifest


def error_codes(manifest, object_manifest):
    issues, _ = verify_data.validate_manifest(
        read_csv(manifest), read_csv(object_manifest)
    )
    return {issue["code"] for issue in issues if issue["severity"] == "error"}


@pytest.mark.parametrize("count", [1, 3, 9])
def test_randomized_multi_tool_samples_accept_any_count_with_blank_angles(
    tmp_path, monkeypatch, count
):
    manifest, objects = sample_manifests(tmp_path, monkeypatch, [["2", "12"]] * count)
    assert not error_codes(manifest, objects)


def test_multi_tool_sample_compares_tool_sets_independently_of_instance_order(
    tmp_path, monkeypatch
):
    manifest, objects = sample_manifests(
        tmp_path, monkeypatch, [["2", "12"], ["12", "2"]]
    )
    assert not error_codes(manifest, objects)


def test_multi_tool_sample_rejects_changed_tool_set(tmp_path, monkeypatch):
    manifest, objects = sample_manifests(
        tmp_path, monkeypatch, [["2", "12"], ["2", "13"]]
    )
    assert "mixed_sample_values" in error_codes(manifest, objects)


@pytest.mark.parametrize(
    "angles, expected",
    [
        (["", ""], set()),
        (["0", "45"], set()),
        (["0", "90", "180", "270"], set()),
        (["270", "0", "180", "90"], set()),
        (["0", "45", "90", "135"], set()),
        (["0", "90", "180", "180"], set()),
        (["0", "90", "180", "360"], {"invalid_angles"}),
        (["0", "90", "180", "nan"], {"invalid_angles"}),
        ([str(angle) for angle in range(0, 360, 45)], set()),
        (["0"] * 8, set()),
    ],
)
def test_single_tool_collection_gaps_are_allowed_but_malformed_angles_fail(
    tmp_path, monkeypatch, angles, expected
):
    manifest, objects = sample_manifests(
        tmp_path, monkeypatch, [["2"]] * len(angles), angles
    )
    assert error_codes(manifest, objects) == expected


@pytest.mark.parametrize("angles", [["", ""], ["0", "90", "180"], ["0"] * 8])
def test_collection_gaps_are_reported_as_warnings(tmp_path, monkeypatch, angles):
    manifest, objects = sample_manifests(
        tmp_path, monkeypatch, [["2"]] * len(angles), angles
    )
    issues, _ = verify_data.validate_manifest(read_csv(manifest), read_csv(objects))
    coverage = [item for item in issues if item["code"] == "incomplete_angle_coverage"]
    assert coverage and all(item["severity"] == "warning" for item in coverage)
    assert not any(item["severity"] == "error" for item in issues)


def test_partial_validation_distinguishes_undownloaded_from_missing(
    tmp_path, monkeypatch
):
    manifest, objects = sample_manifests(
        tmp_path, monkeypatch, [["2"]] * 4, ["0", "90", "180", "270"]
    )
    rows = read_csv(manifest)
    path = rows[0]["Filepath"]
    (tmp_path / path).unlink()
    tracked = {row["Filepath"] for row in rows}
    issues, _ = verify_data.validate_manifest(
        rows, read_csv(objects), tracked_files=tracked
    )
    assert not any(item["severity"] == "error" for item in issues)
    assert any(
        item["code"] == "not_downloaded" and item["severity"] == "info"
        for item in issues
    )
    issues, _ = verify_data.validate_manifest(
        read_csv(manifest), read_csv(objects), tracked_files=tracked - {path}
    )
    assert any(
        item["code"] == "missing_image" and item["severity"] == "error"
        for item in issues
    )


def test_partial_validation_still_checks_remote_mask_metadata(tmp_path, monkeypatch):
    manifest, objects = sample_manifests(
        tmp_path, monkeypatch, [["2"]] * 4, ["0", "90", "180", "270"]
    )
    instances = read_csv(objects)
    mask = "masks/images/2/0.npz"
    instances[0].update(
        {
            "Mask Path": mask,
            "Mask Index": "0",
            "Visibility": "clear",
            "Reviewed": "TRUE",
            "Annotation Complete": "TRUE",
            "Visible": "TRUE",
            "Mask Valid": "TRUE",
            "Class Valid": "TRUE",
            "Base Valid": "TRUE",
        }
    )
    rows = read_csv(manifest)
    tracked = {row["Filepath"] for row in rows} | {mask}
    issues, _ = verify_data.validate_manifest(rows, instances, tracked_files=tracked)
    assert not any(item["severity"] == "error" for item in issues)
    instances[0]["Class Valid"] = "FALSE"
    issues, _ = verify_data.validate_manifest(
        read_csv(manifest), instances, tracked_files=tracked
    )
    assert any(item["code"] == "invalid_annotation" for item in issues)


def test_duplicate_physical_tool_is_rejected_before_annotation(tmp_path, monkeypatch):
    manifest, objects = sample_manifests(tmp_path, monkeypatch, [["2", "12"]])
    instances = read_csv(objects)
    instances[1]["Tool ID"] = "2"
    write_csv(objects, instances, instances[0].keys())
    assert "duplicate_tool_instance" in error_codes(manifest, objects)


@pytest.mark.parametrize("angle, expected", [("270", 0), ("180", 0), ("360", 1)])
def test_static_command_exit_status_reflects_errors(
    tmp_path, monkeypatch, angle, expected
):
    manifest, objects = sample_manifests(
        tmp_path, monkeypatch, [["2"]] * 4, ["0", "90", "180", angle]
    )
    monkeypatch.setattr(verify_data, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(verify_data, "write_reports", lambda report: None)
    monkeypatch.setattr(sys, "argv", ["verify_data.py", "--static-only"])
    assert verify_data.main() == expected


@pytest.mark.parametrize(
    "angle, valid", [("17.5", True), ("NA", False), ("360", False), ("nan", False)]
)
def test_optional_multi_tool_angle_is_validated(tmp_path, monkeypatch, angle, valid):
    manifest, objects = sample_manifests(tmp_path, monkeypatch, [["2", "12"]], [angle])
    assert ("invalid_angles" not in error_codes(manifest, objects)) == valid


def test_preparation_preserves_multi_tool_sample_and_blank_angles(
    tmp_path, monkeypatch
):
    manifest, objects = sample_manifests(tmp_path, monkeypatch, [["2", "12"]] * 3)
    monkeypatch.setattr(
        prepare_manifests,
        "extract_metadata",
        lambda _: dict.fromkeys(prepare_manifests.METADATA_COLUMNS, ""),
    )
    monkeypatch.setattr(
        "builtins.input", lambda _: pytest.fail("No metadata prompts expected")
    )
    prepare_manifests.main(tmp_path / "images", tmp_path / "raw-images", manifest)
    assert {row["Sample ID"] for row in read_csv(manifest)} == {"7"}
    assert {row["Sample ID"] for row in read_csv(objects)} == {"7"}
    assert all(not row["Angle"] for row in read_csv(manifest))
    assert not error_codes(manifest, objects)


def test_visual_review_identifies_randomized_samples(tmp_path, monkeypatch):
    manifest, objects = sample_manifests(tmp_path, monkeypatch, [["2", "12"]] * 3)
    _, samples = verify_data.validate_manifest(read_csv(manifest), read_csv(objects))
    prompt = verify_data.agent_prompt(
        [("7", samples["7"], tmp_path / "sheet.jpg")], None
    )
    assert '"capture_type": "randomized_multi_tool"' in prompt
    assert "Blank angles and any image count are expected" in prompt
    assert "Do not flag missing angles" in prompt
