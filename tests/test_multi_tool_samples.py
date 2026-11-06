from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image

import prepare_manifests
from annotations import read_csv, write_image_manifests
from prepare_manifests import assign_multi_tool_samples


def declared_setup(setup, count, sample_id="", tools=("2", "12")):
    images, objects = [], {}
    for index in range(count):
        key = f"{setup}-{index}"
        images.append(
            {
                "Image ID": key,
                "Filepath": f"images/multi/{setup}/{index}.jpg",
                "Sample ID": sample_id,
                "Angle": "",
            }
        )
        objects[key] = [
            {
                "Image ID": key,
                "Instance ID": f"{key}-{tool}",
                "Tool ID": tool,
                "Tool Class": "scissors",
                "Tool Name": f"instrument_{tool}",
                "Confidence": "0.9",
                "Reviewed": "TRUE",
                "Mask Path": f"{key}.npz",
            }
            for tool in tools
        ]
    return images, objects


@pytest.mark.parametrize("count", [1, 3, 9])
def test_setup_assigns_one_sample_for_any_image_count_and_is_idempotent(count):
    images, objects = declared_setup("setup-a", count)
    original_objects = deepcopy(objects)
    assert (
        assign_multi_tool_samples(images, objects, Path("images/multi"), set()) == count
    )
    assert {row["Sample ID"] for row in images} == {"0"}
    assert all(not row["Angle"] for row in images)
    assert objects == original_objects
    before = deepcopy(images)
    assert assign_multi_tool_samples(images, objects, Path("images/multi"), set()) == 0
    assert images == before


def test_separate_folders_with_same_tools_receive_distinct_globally_unique_ids():
    first, first_objects = declared_setup("first", 2)
    second, second_objects = declared_setup("second", 2)
    single = {
        "Image ID": "single",
        "Filepath": "images/2/single.jpg",
        "Sample ID": "40",
        "Angle": "90",
    }
    images = [single, *second, *first]
    assign_multi_tool_samples(
        images, first_objects | second_objects, Path("images/multi"), set()
    )
    assert {row["Sample ID"] for row in first} == {"41"}
    assert {row["Sample ID"] for row in second} == {"42"}
    assert single["Sample ID"] == "40" and single["Angle"] == "90"


def test_existing_sample_extends_to_new_views_and_preserves_explicit_angles():
    images, objects = declared_setup("setup", 3)
    images[0].update({"Sample ID": "75", "Angle": "17.5"})
    images[2]["Filepath"] = "images/multi/setup/nested/2.jpg"
    assert assign_multi_tool_samples(images, objects, Path("images/multi"), set()) == 2
    assert {row["Sample ID"] for row in images} == {"75"}
    assert images[0]["Angle"] == "17.5"


@pytest.mark.parametrize(
    "conflict",
    [
        "tools",
        "samples",
        "external_sample",
        "missing_metadata",
        "duplicate_tool",
        "undeclared_file",
    ],
)
def test_conflicting_setup_is_reported_without_changes_but_other_setups_continue(
    conflict, capsys
):
    images, objects = declared_setup("conflict", 3)
    undeclared = set()
    if conflict == "tools":
        objects[images[1]["Image ID"]][1]["Tool ID"] = "99"
    elif conflict == "samples":
        images[0]["Sample ID"], images[1]["Sample ID"] = "5", "6"
    elif conflict == "external_sample":
        images[0]["Sample ID"] = "5"
        images.append(
            {
                "Image ID": "single",
                "Filepath": "images/2/other.jpg",
                "Sample ID": "5",
                "Angle": "",
            }
        )
    elif conflict == "missing_metadata":
        objects[images[1]["Image ID"]] = []
    elif conflict == "duplicate_tool":
        objects[images[1]["Image ID"]][1]["Tool ID"] = "2"
    else:
        undeclared.add("images/multi/conflict/unregistered.jpg")
    before = deepcopy(images)
    valid, valid_objects = declared_setup("valid", 1)
    assert (
        assign_multi_tool_samples(
            images + valid, objects | valid_objects, Path("images/multi"), undeclared
        )
        == 1
    )
    assert images == before
    assert valid[0]["Sample ID"]
    assert (
        "skipping sample assignment for images/multi/conflict"
        in capsys.readouterr().out
    )


def test_images_directly_under_multi_are_not_guessed_into_a_setup(capsys):
    images, objects = declared_setup("setup", 1)
    images[0]["Filepath"] = "images/multi/ungrouped.jpg"
    assert assign_multi_tool_samples(images, objects, Path("images/multi"), set()) == 0
    assert not images[0]["Sample ID"]
    assert "no setup subfolder" in capsys.readouterr().out


def test_main_assigns_samples_refreshes_objects_and_reuses_id_for_added_images(
    tmp_path, monkeypatch
):
    images, objects = declared_setup("setup", 3)
    images_dir, raw_dir = tmp_path / "images", tmp_path / "raw-images"
    raw_dir.mkdir()
    for row in images:
        row.update(
            {
                "Capture Session ID": "",
                **dict.fromkeys(prepare_manifests.METADATA_COLUMNS, ""),
            }
        )
        path = tmp_path / row["Filepath"]
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 8)).save(path)
    manifest, object_manifest = (
        tmp_path / "image_manifest.csv",
        tmp_path / "image_object_manifest.csv",
    )
    write_image_manifests(
        manifest,
        images,
        object_manifest,
        [obj for group in objects.values() for obj in group],
    )
    before_objects = {
        (obj["Image ID"], obj["Instance ID"]): obj for obj in read_csv(object_manifest)
    }
    monkeypatch.setattr(
        "builtins.input", lambda _: pytest.fail("No tool intake prompts expected")
    )
    prepare_manifests.main(images_dir, raw_dir, manifest)
    assert {row["Sample ID"] for row in read_csv(manifest)} == {"0"}
    for obj in read_csv(object_manifest):
        before = before_objects[(obj["Image ID"], obj["Instance ID"])]
        assert obj["Sample ID"] == "0"
        assert {k: v for k, v in obj.items() if k != "Sample ID"} == {
            k: v for k, v in before.items() if k != "Sample ID"
        }
    new_images, new_objects = declared_setup("setup", 1)
    new_images[0].update(
        {"Image ID": "added", "Filepath": "images/multi/setup/added.jpg"}
    )
    new_images[0].update(
        {
            "Capture Session ID": "",
            **dict.fromkeys(prepare_manifests.METADATA_COLUMNS, ""),
        }
    )
    additions = [
        {**obj, "Image ID": "added"} for group in new_objects.values() for obj in group
    ]
    Image.new("RGB", (8, 8)).save(tmp_path / new_images[0]["Filepath"])
    write_image_manifests(
        manifest,
        read_csv(manifest) + new_images,
        object_manifest,
        read_csv(object_manifest) + additions,
    )
    prepare_manifests.main(images_dir, raw_dir, manifest)
    assert len(read_csv(manifest)) == 4
    assert {row["Sample ID"] for row in read_csv(manifest)} == {"0"}
    assert {obj["Sample ID"] for obj in read_csv(object_manifest)} == {"0"}
    before = manifest.read_bytes(), object_manifest.read_bytes()
    prepare_manifests.main(images_dir, raw_dir, manifest)
    assert before == (manifest.read_bytes(), object_manifest.read_bytes())
