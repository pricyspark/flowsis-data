import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import annotate
import verify_data
from annotations import IMAGE_OBJECT_FIELDS, read_csv, write_csv
from image_review import Candidate, review_image_instances
from masks import load_mask_bundle


@pytest.fixture(autouse=True)
def use_mockable_image_preview(monkeypatch):
    monkeypatch.setattr(annotate, "show_overlap_review", lambda *_: None)
    monkeypatch.setattr(
        annotate,
        "show_tool_review",
        lambda image, candidates, title, tools: annotate.show_candidates(
            image, candidates, title
        ),
    )


def instance(identity, tool_id, tool_class="scissors"):
    return {
        "Image ID": "image-1",
        "Instance ID": identity,
        "Tool ID": tool_id,
        "Tool Class": tool_class,
        "Tool Name": f"tool {tool_id}",
        "Confidence": "0.8",
    }


def candidates():
    masks = np.zeros((3, 10, 12), dtype=np.bool_)
    masks[0, 1:4, 2:5] = True
    masks[1, 3:8, 3:8] = True  # Crosses the other tool's visible region.
    masks[2, 7:9, 2:5] = True
    return [
        Candidate(mask, 0.9, "scissors", f"candidate-{index}")
        for index, mask in enumerate(masks)
    ]


def interactive_review(
    monkeypatch,
    tmp_path,
    commands,
    rows=None,
    initial=None,
    predict=None,
    show_tools=None,
    inspect_order=None,
):
    commands = iter(commands)
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    log = annotate.ReviewLog(tmp_path / "review.jsonl")
    result = review_image_instances(
        Image.new("RGB", (12, 10)),
        candidates() if initial is None else initial,
        rows or [instance("a", "12"), instance("b", "13")],
        key="image-1",
        show=lambda *args: None,
        predict=predict or (lambda _: []),
        log=log.append,
        show_tools=show_tools,
        inspect_order=inspect_order,
    )
    return result, [json.loads(line) for line in log.path.read_text().splitlines()]


def test_grouping_preserves_gaps_and_source_history(monkeypatch, tmp_path, capsys):
    result, records = interactive_review(
        monkeypatch,
        tmp_path,
        [
            "a",
            "g a 0,2",
            "v a partial",
            "g b 1",
            "order 12 13",
            "a",
        ],
    )
    assert "Resolve tool 12" in capsys.readouterr().out
    merged = result.selected[0]
    np.testing.assert_array_equal(
        merged.mask, candidates()[0].mask | candidates()[2].mask
    )
    assert not merged.mask[5, 3]
    assert merged.visibility == "partial"
    grouped = next(
        record for record in records if record.get("action") == "assign_fragments"
    )
    assert [part["candidate_id"] for part in grouped["assignments"]["a"]] == [
        "candidate-0",
        "candidate-2",
    ]


def test_reprompt_preserves_assignments_and_resolves_extra_tool(monkeypatch, tmp_path):
    result, records = interactive_review(
        monkeypatch,
        tmp_path,
        [
            "g a 0,2",
            "v b occluded",
            "x 1",
            "p forceps",
            'n c 14 "forceps" "fine forceps" 0.9',
            "g c 3",
            "order 12 14",
            "a",
        ],
        predict=lambda _: [candidates()[1]],
    )
    assert [part.identity for part in result.selected] == ["a", "b", "c"]
    assert result.selected[1].visibility == "occluded"
    assert not result.selected[1].mask.any()
    assert records[-1]["assignments"]["a"][0]["index"] == 0
    assert result.instances[2]["Tool Class"] == "forceps"


def test_conflicting_assignments_and_unresolved_candidates_block_acceptance(
    monkeypatch, tmp_path, capsys
):
    result, _ = interactive_review(
        monkeypatch,
        tmp_path,
        [
            "g a 0",
            "g b 0",
            "v b absent",
            "a",
            "i 1",
            "x 2",
            "a",
        ],
    )
    output = capsys.readouterr().out
    assert "Clear the other tool" in output
    assert "Assign, exclude, or ignore" in output
    assert result.selected[1].visibility == "absent"
    np.testing.assert_array_equal(result.ignored[0].mask, candidates()[1].mask)


def setup_manifest(tmp_path, rows):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (12, 10)).save(tmp_path / "images/tools.jpg")
    write_csv(
        tmp_path / "image_manifest.csv",
        [{"Image ID": "image-1", "Filepath": "images/tools.jpg"}],
        ["Image ID", "Filepath"],
    )
    write_csv(tmp_path / "image_object_manifest.csv", rows, IMAGE_OBJECT_FIELDS)


def seed_known_tools(tool_ids):
    images = read_csv("image_manifest.csv")
    images.append({"Image ID": "catalog", "Filepath": "images/catalog.jpg"})
    write_csv("image_manifest.csv", images, images[0].keys())
    Image.new("RGB", (12, 10)).save("images/catalog.jpg")
    objects = read_csv("image_object_manifest.csv")
    objects.extend(
        {**instance(f"catalog-{tool_id}", tool_id, "forceps"), "Image ID": "catalog"}
        for tool_id in tool_ids
    )
    write_csv("image_object_manifest.csv", objects, IMAGE_OBJECT_FIELDS)


def test_cli_saves_ordered_tools_and_resumes(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    setup_manifest(
        tmp_path,
        [instance("a", "12"), instance("b", "13"), instance("c", "14", "forceps")],
    )
    monkeypatch.setattr(sys, "argv", ["annotate.py", "images", "--data-dir", "."])
    prompts_seen = []

    class Backend:
        def __init__(self, *args, **kwargs):
            self.refine_crop = kwargs["refine_crop"]

        def predict(self, images, prompts):
            assert not self.refine_crop
            prompts_seen.extend(prompts)
            return [
                (
                    [candidates()[0], candidates()[2]]
                    if prompt == "tool 12 scissors"
                    else [candidates()[1]] if prompt == "tool 13 scissors" else []
                )
                for prompt in prompts
            ]

    monkeypatch.setattr(annotate, "Sam3ImageBackend", Backend)
    monkeypatch.setattr(annotate, "show_candidates", lambda *args: None)
    commands = iter(["g a 0,1", "g b 2", "v c occluded", "a", "order 12 13", "a"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    annotate.main()
    assert prompts_seen == ["tool 12 scissors", "tool 13 scissors", "forceps"]
    bundle = load_mask_bundle("masks/images/tools.npz")
    assert bundle.identities == ("a", "b")
    assert not np.any(bundle.masks[0] & bundle.masks[1])
    source = candidates()[1].mask
    np.testing.assert_array_equal(
        bundle.masks[1], source & ~(candidates()[0].mask | candidates()[2].mask)
    )
    rows = read_csv("image_object_manifest.csv")
    assert [row["Tool ID"] for row in rows] == ["12", "13", "14"]
    assert rows[2]["Visibility"] == "occluded"
    assert rows[2]["Mask Index"] == "-1"
    records = [
        json.loads(line) for line in Path("image_review.jsonl").read_text().splitlines()
    ]
    assert records[-1]["depth_order_tool_ids"] == ["12", "13"]
    monkeypatch.setattr(verify_data, "ROOT", tmp_path)
    assert not verify_data.validate_image_annotations(
        read_csv("image_manifest.csv")[0], rows
    )
    rows[0]["Mask Index"] = "1"
    assert verify_data.validate_image_annotations(
        read_csv("image_manifest.csv")[0], rows
    )
    annotate.main()
    assert prompts_seen == ["tool 12 scissors", "tool 13 scissors", "forceps"]


@pytest.mark.parametrize("flag", ["--merge-masks", "--single-tool", "--auto-accept"])
def test_unsafe_automatic_multi_tool_modes_fail_before_model_load(
    monkeypatch, tmp_path, flag
):
    monkeypatch.chdir(tmp_path)
    setup_manifest(tmp_path, [instance("a", "12"), instance("b", "13")])
    monkeypatch.setattr(sys, "argv", ["annotate.py", "images", "--data-dir", ".", flag])
    monkeypatch.setattr(
        annotate,
        "Sam3ImageBackend",
        lambda *args, **kwargs: pytest.fail("Loaded model"),
    )
    with pytest.raises(ValueError):
        annotate.main()


def test_high_iou_candidates_survive_until_review(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    setup_manifest(tmp_path, [instance("a", "12"), instance("b", "13")])
    monkeypatch.setattr(sys, "argv", ["annotate.py", "images", "--data-dir", "."])
    mask = candidates()[0].mask

    class Backend:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, images, prompts):
            return [
                [Candidate(mask.copy(), 0.9, prompt, str(index)) for index in range(1)]
                for prompt in prompts
            ]

    displayed = []
    monkeypatch.setattr(annotate, "Sam3ImageBackend", Backend)
    monkeypatch.setattr(
        annotate,
        "show_candidates",
        lambda image, parts, key: displayed.append(len(parts)),
    )
    commands = iter(["g a 0", "g b 1", "order 12 13", "v 13 occluded", "x 1", "a"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    annotate.main()
    assert displayed[0] == 2
    bundle = load_mask_bundle("masks/images/tools.npz")
    assert len(bundle.masks) == 1
    np.testing.assert_array_equal(bundle.masks[0], mask)


def test_declare_sample_preserves_metadata_and_reopen_clears_annotations(
    monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    setup_manifest(tmp_path, [instance("a", "12")])
    seed_known_tools(["13", "14"])
    images = read_csv("image_manifest.csv")
    images[0]["Sample ID"] = "4"
    images[0]["Angle"] = "0"
    write_csv("image_manifest.csv", images, images[0].keys())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "annotate.py",
            "declare", "--data-dir", ".",
            "--sample-id",
            "4",
            "--tool-id",
            "13",
        ],
    )
    annotate.main()
    assert read_csv("image_manifest.csv")[0]["Tool IDs"] == "12;13"
    assert read_csv("image_object_manifest.csv")[0]["Instance ID"] == "a"
    log = annotate.ReviewLog(Path("image_review.jsonl"))
    log.append("image-1", "accepted")
    argv = [
        "annotate.py",
        "declare", "--data-dir", ".",
        "--sample-id",
        "4",
        "--tool-id",
        "14",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="--reopen"):
        annotate.main()
    monkeypatch.setattr(sys, "argv", [*argv, "--reopen"])
    annotate.main()
    assert not annotate.ReviewLog(Path("image_review.jsonl")).completed("image-1")
    assert len(read_csv("image_object_manifest.csv")) == 5


def test_resolved_rows_follow_mask_order_and_reject_missing_identity():
    rows = [instance("a", "12"), instance("b", "13")]
    selected = [
        Candidate(candidates()[1].mask, 0.9, "scissors", "b"),
        Candidate(candidates()[0].mask, 0.9, "scissors", "a"),
    ]
    result = annotate.resolved_image_rows(rows, selected, Path("mask.npz"))
    assert [row["Mask Index"] for row in result] == [1, 0]
    with pytest.raises(ValueError, match="Every declared instance"):
        annotate.resolved_image_rows(rows, selected[:1], Path("mask.npz"))


def test_explicit_empty_image_retains_resolved_tool_metadata(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    setup_manifest(tmp_path, [instance("a", "12"), instance("b", "13")])
    monkeypatch.setattr(sys, "argv", ["annotate.py", "images", "--data-dir", "."])

    class Backend:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, images, prompts):
            return [[] for _ in images]

    monkeypatch.setattr(annotate, "Sam3ImageBackend", Backend)
    monkeypatch.setattr(annotate, "show_candidates", lambda *args: None)
    monkeypatch.setattr("builtins.input", lambda _: "e")
    annotate.main()
    bundle = load_mask_bundle("masks/images/tools.npz")
    assert bundle.masks.shape == (0, 10, 12)
    rows = read_csv("image_object_manifest.csv")
    assert [row["Tool ID"] for row in rows] == ["12", "13"]
    assert all(row["Visibility"] == "absent" for row in rows)
    assert all(row["Annotation Complete"] == "TRUE" for row in rows)
    assert all(row["Reviewed"] == "FALSE" for row in rows)
    monkeypatch.setattr(verify_data, "ROOT", tmp_path)
    assert not verify_data.validate_image_annotations(
        read_csv("image_manifest.csv")[0], rows
    )


def test_declare_new_image_and_reject_duplicate_tool(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    setup_manifest(tmp_path, [instance("a", "12")])
    seed_known_tools(["13"])
    Image.new("RGB", (12, 10)).save("images/new.jpg")
    argv = [
        "annotate.py",
        "declare", "--data-dir", ".",
        "--filepath",
        "images/new.jpg",
        "--tool-id",
        "13",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    annotate.main()
    rows = read_csv("image_manifest.csv")
    assert len(rows) == 3
    added = next(row for row in rows if row["Filepath"] == "images/new.jpg")
    assert added["Tool IDs"] == "13"
    assert added["Capture Session ID"]
    before = Path("image_object_manifest.csv").read_text()
    with pytest.raises(ValueError, match="unique"):
        annotate.main()
    assert Path("image_object_manifest.csv").read_text() == before


@pytest.mark.parametrize("folder", ["setup-001", "images/multi/setup-001"])
def test_declare_folder_with_known_tool_ids(monkeypatch, tmp_path, folder):
    monkeypatch.chdir(tmp_path)
    setup_manifest(tmp_path, [instance("a", "12"), instance("b", "13", "forceps")])
    root = tmp_path / "images/multi/setup-001"
    (root / "nested").mkdir(parents=True)
    for path in [root / "one.jpg", root / "nested/two.PNG"]:
        Image.new("RGB", (12, 10)).save(path)
    (root / "notes.txt").write_text("ignore this")
    other = root.parent / "other"
    other.mkdir()
    Image.new("RGB", (12, 10)).save(other / "other.jpg")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "annotate.py",
            "declare",
            "--data-dir",
            ".",
            "--folder",
            folder,
            "--tool-ids",
            "12",
            "13",
        ],
    )
    annotate.main()
    images = read_csv("image_manifest.csv")
    assert len(images) == 3
    objects = read_csv("image_object_manifest.csv")
    for image in images[1:]:
        assert image["Tool IDs"] == "12;13"
        declared = [row for row in objects if row["Image ID"] == image["Image ID"]]
        assert [
            (row["Tool ID"], row["Tool Class"], row["Tool Name"], row["Confidence"])
            for row in declared
        ] == [("12", "scissors", "tool 12", "0.8"), ("13", "forceps", "tool 13", "0.8")]


@pytest.mark.parametrize("failure", ["unknown", "conflict", "confidence"])
def test_declare_invalid_lookup_leaves_manifests_unchanged(
    monkeypatch, tmp_path, failure
):
    monkeypatch.chdir(tmp_path)
    rows = [instance("a", "12")]
    if failure != "unknown":
        rows.append({**instance("b", "13"), "Tool Class": "forceps"})
        rows.append(
            {
                **instance("c", "13"),
                "Confidence": "invalid" if failure == "confidence" else "0.8",
                "Tool Class": "forceps" if failure == "confidence" else "scissors",
            }
        )
    setup_manifest(tmp_path, rows)
    before = {
        path: path.read_bytes()
        for path in [Path("image_manifest.csv"), Path("image_object_manifest.csv")]
    }
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "annotate.py",
            "declare", "--data-dir", ".",
            "--filepath",
            "images/tools.jpg",
            "--tool-ids",
            "12",
            "13",
        ],
    )
    with pytest.raises(
        ValueError, match="Unknown Tool ID|conflicting|invalid identification"
    ):
        annotate.main()
    assert all(path.read_bytes() == content for path, content in before.items())
    assert not Path("image_review.jsonl").exists()


@pytest.mark.parametrize("folder", ["missing", "../outside", "/tmp", "empty"])
def test_declare_rejects_invalid_or_empty_folder(monkeypatch, tmp_path, folder):
    monkeypatch.chdir(tmp_path)
    setup_manifest(tmp_path, [instance("a", "12")])
    (tmp_path / "images/multi/empty").mkdir(parents=True)
    monkeypatch.setattr(
        sys, "argv",
        [
            "annotate.py",
            "declare",
            "--data-dir",
            ".",
            "--folder",
            folder,
            "--tool-id",
            "12",
        ],
    )
    with pytest.raises(ValueError, match="folder|Folder"):
        annotate.main()


def test_explicit_reviewer_explains_assignment_before_accepting(
    monkeypatch, tmp_path, capsys
):
    result, _ = interactive_review(
        monkeypatch,
        tmp_path,
        ["a", "g instance-0 0", "a"],
        rows=[instance("instance-0", "11", "pliers")],
        initial=[candidates()[0]],
    )
    output = capsys.readouterr().out
    assert "Mask candidates: 1" in output
    assert "0: scissors [unassigned] | score 0.90" in output
    assert "'g 11 0', then 'a'" in output
    assert "Resolve tool 11 before acceptance. Assign its mask" in output
    assert result.decision == "accepted"
    assert result.selected[0].identity == "instance-0"
    np.testing.assert_array_equal(result.selected[0].mask, candidates()[0].mask)


def test_no_predictions_explains_reprompt_and_retains_unresolved_tool(
    monkeypatch, tmp_path, capsys
):
    prompts = []

    def predict(prompt):
        prompts.append(prompt)
        return [candidates()[0]]

    result, _ = interactive_review(
        monkeypatch,
        tmp_path,
        ["a", "p lineman pliers", "g instance-0 0", "a"],
        rows=[instance("instance-0", "11", "pliers")],
        initial=[],
        predict=predict,
    )
    output = capsys.readouterr().out
    assert "Mask candidates: 0" in output
    assert "No masks were predicted" in output
    assert "No mask candidates are available; try 'p pliers'" in output
    assert "Prompt 'lineman pliers' returned 1 mask candidate(s)." in output
    assert prompts == ["lineman pliers"]
    assert result.decision == "accepted"
    assert result.selected[0].mask.any()


def test_failed_reprompt_does_not_silently_accept_visible_tool(
    monkeypatch, tmp_path, capsys
):
    result, _ = interactive_review(
        monkeypatch,
        tmp_path,
        ["p pliers", "a", "s"],
        rows=[instance("instance-0", "11", "pliers")],
        initial=[],
        predict=lambda _: [],
    )
    output = capsys.readouterr().out
    assert "returned 0 mask candidate(s)" in output
    assert "Resolve tool 11 before acceptance" in output
    assert result.decision == "skipped"
    assert not result.selected


def test_tool_ids_control_grouping_visibility_and_clearing(
    monkeypatch, tmp_path, capsys
):
    result, records = interactive_review(
        monkeypatch,
        tmp_path,
        [
            "g 0", "g 99 0", "g 0 0,2", "u 0", "g 0 0,2", "v 0 partial",
            "g 3 1", "order 0 3", "a",
        ],
        rows=[instance("internal-a", "0"), instance("internal-b", "3", "pliers")],
    )
    output = capsys.readouterr().out
    assert "Declared tools (use Tool ID in commands)" in output
    assert "Provide a candidate number: g 0 0" in output
    assert "Unknown Tool ID" in output
    assert "[tool 0]" in output
    assert "[tool 3]" in output
    assert "internal-a" not in output and "internal-b" not in output
    assert result.selected[0].identity == "internal-a"
    assert result.selected[0].visibility == "partial"
    np.testing.assert_array_equal(
        result.selected[0].mask, candidates()[0].mask | candidates()[2].mask
    )
    assert records[-1]["assignments"]["internal-b"][0]["index"] == 1
    assert any(record.get("action") == "clear_assignment" for record in records)


def test_tool_id_takes_precedence_over_legacy_instance_id(monkeypatch, tmp_path):
    result, _ = interactive_review(
        monkeypatch,
        tmp_path,
        ["g 0 0", "g 3 1", "order 0 3", "a"],
        rows=[instance("3", "0"), instance("0", "3", "pliers")],
        initial=candidates()[:2],
    )
    assert [candidate.identity for candidate in result.selected] == ["3", "0"]
    np.testing.assert_array_equal(result.selected[0].mask, candidates()[0].mask)
    np.testing.assert_array_equal(
        result.selected[1].mask, candidates()[1].mask & ~candidates()[0].mask
    )


def test_new_tool_command_generates_internal_instance_id(monkeypatch, tmp_path):
    from annotations import stable_media_id

    result, _ = interactive_review(
        monkeypatch,
        tmp_path,
        ["g 0 0", 'n 3 "pliers" "rongeur" 0.9', "p rongeur", "g 3 1", "order 0 3", "a"],
        rows=[instance("existing", "0")],
        initial=candidates()[:1],
        predict=lambda _: [candidates()[1]],
    )
    added = result.instances[1]
    assert added["Tool ID"] == "3"
    assert added["Instance ID"] == stable_media_id("instance", "image-1/3")
    assert result.selected[1].identity == added["Instance ID"]


def test_shared_classes_use_name_and_class_prompts():
    rows = [
        {**instance("a", "0"), "Tool Name": "iris"},
        {**instance("b", "3"), "Tool Name": "mayo"},
        {**instance("c", "4", "pliers"), "Tool Name": "rongeur"},
    ]
    assert annotate.image_prompts(rows, "tool-class", {}) == [
        "iris scissors",
        "mayo scissors",
        "pliers",
    ]
    assert annotate.image_prompts(rows, "name-and-class", {}) == [
        "iris scissors",
        "mayo scissors",
        "rongeur pliers",
    ]
    # Class sharing is determined within each image, not across the batch.
    assert annotate.image_prompts(rows[:1], "tool-class", {}) == ["scissors"]


def test_shared_class_prompt_normalization_and_deduplication():
    rows = [
        {**instance("a", "0", " scissors "), "Tool Name": "iris scissors"},
        {**instance("b", "3", "scissors"), "Tool Name": "iris"},
    ]
    assert annotate.image_prompts(rows, "tool-class", {}) == ["iris scissors"]
    assert annotate.image_prompts(rows, "tool-class", {"iris": "fine"}) == [
        "iris scissors",
        "fine scissors",
    ]


def test_declared_tools_display_name_class_label(monkeypatch, tmp_path, capsys):
    result, _ = interactive_review(
        monkeypatch,
        tmp_path,
        ["q"],
        rows=[
            {**instance("a", "0"), "Tool Name": "iris"},
            {**instance("b", "3"), "Tool Name": "mayo"},
        ],
    )
    output = capsys.readouterr().out
    assert "tool 0 | iris scissors |" in output
    assert "tool 3 | mayo scissors |" in output
    assert result.decision == "quit"


def test_order_requires_confirmation_and_previews_disputed_pixels(
    monkeypatch, tmp_path
):
    panels = []
    result, records = interactive_review(
        monkeypatch,
        tmp_path,
        ["g 12 0,2", "g 13 1", "order 13 12", "q"],
        show_tools=lambda image, parts, title, tools: panels.append(tools),
    )
    assert result.decision == "quit"  # Ordering alone cannot accept/save an image.
    front, back = candidates()[1].mask, candidates()[0].mask | candidates()[2].mask
    np.testing.assert_array_equal(panels[-1][0].mask, back & ~front)
    np.testing.assert_array_equal(panels[-1][1].mask, front)
    np.testing.assert_array_equal(panels[-1][0].original, back)
    np.testing.assert_array_equal(panels[-1][0].disputed, back & front)
    assert panels[-1][0].edit_note == "Ordered mask — order reset to undo"
    ordered = next(
        record for record in records if record.get("action") == "overlap_order"
    )
    assert ordered["depth_order_tool_ids"] == ["13", "12"]
    assert ordered["depth_order_instance_ids"] == ["b", "a"]
    assert ordered["overlap_pixels"] == 4
    assert ordered["removed_pixels"] == {"13": 0, "12": 4}


def test_reordering_and_reset_use_source_masks(monkeypatch, tmp_path, capsys):
    result, _ = interactive_review(
        monkeypatch,
        tmp_path,
        [
            "g 12 0,2", "g 13 1", "order 12 13", "order reset", "a",
            "order 12 13", "order 13 12", "a",
        ],
    )
    assert "Instance masks overlap" in capsys.readouterr().out
    source_a = candidates()[0].mask | candidates()[2].mask
    source_b = candidates()[1].mask
    np.testing.assert_array_equal(result.selected[0].mask, source_a & ~source_b)
    np.testing.assert_array_equal(result.selected[1].mask, source_b)
    np.testing.assert_array_equal(
        result.selected[0].mask | result.selected[1].mask, source_a | source_b
    )
    assert result.depth_order == ["13", "12"]


@pytest.mark.parametrize(
    "edit", ["g 12 0,2", "v 12 partial", "clean 12 holes 1", "p scissors"]
)
def test_editing_invalidates_order(monkeypatch, tmp_path, capsys, edit):
    result, records = interactive_review(
        monkeypatch, tmp_path,
        ["g 12 0,2", "g 13 1", "order 12 13", edit, "a", "order 13 12", "a"],
    )
    output = capsys.readouterr().out
    assert "Ordering cleared before editing" in output
    assert "Instance masks overlap" in output
    assert any(
        record.get("action") == "overlap_order_invalidated" for record in records
    )
    assert result.depth_order == ["13", "12"]


def test_invalid_orders_and_fully_erased_tools_do_not_change_masks(
    monkeypatch, tmp_path, capsys
):
    panels = []
    result, _ = interactive_review(
        monkeypatch, tmp_path,
        [
            "g 12 0", "g 13 1", "order", "order 12 12", "order 99 12",
            "order 12", "order 12 13", "s",
        ],
        initial=[candidates()[0], candidates()[0]],
        show_tools=lambda image, parts, title, tools: panels.append(tools),
    )
    output = capsys.readouterr().out
    assert "List each Tool ID once" in output
    assert "Unknown Tool ID in order" in output
    assert "missing: 13" in output
    assert "Ordering would erase tool 13" in output
    for tool in panels[-1]:
        np.testing.assert_array_equal(tool.mask, candidates()[0].mask)
    assert result.decision == "skipped"


def test_order_only_requires_overlapping_tools_and_preserves_unrelated_masks(
    monkeypatch, tmp_path
):
    parts = candidates()
    unrelated = np.zeros_like(parts[0].mask)
    unrelated[0, 10] = True
    parts[2] = Candidate(unrelated, 0.9, "forceps", "candidate-2")
    result, _ = interactive_review(
        monkeypatch, tmp_path,
        ["g 12 0", "g 13 1", "g 14 2", "order 12 13", "a"],
        initial=parts,
        rows=[instance("a", "12"), instance("b", "13"), instance("c", "14")],
    )
    np.testing.assert_array_equal(result.selected[2].mask, unrelated)
    assert result.depth_order == ["12", "13"]


def test_three_way_order_preserves_union_and_unique_pixels(monkeypatch, tmp_path):
    masks = np.zeros((3, 10, 12), dtype=np.bool_)
    masks[0, 1:6, 1:6] = True
    masks[1, 3:8, 3:8] = True
    masks[2, 4:9, 4:9] = True
    result, records = interactive_review(
        monkeypatch, tmp_path,
        ["g 12 0", "g 13 1", "g 14 2", "order 12 13 14", "a"],
        initial=[
            Candidate(mask, 0.9, "tools", str(index))
            for index, mask in enumerate(masks)
        ],
        rows=[instance("a", "12"), instance("b", "13"), instance("c", "14")],
    )
    actual = np.stack([part.mask for part in result.selected])
    np.testing.assert_array_equal(actual[0], masks[0])
    np.testing.assert_array_equal(actual[1], masks[1] & ~masks[0])
    np.testing.assert_array_equal(actual[2], masks[2] & ~(masks[0] | masks[1]))
    np.testing.assert_array_equal(actual.any(axis=0), masks.any(axis=0))
    assert (actual.sum(axis=0) <= 1).all()
    ordered = next(
        record for record in records if record.get("action") == "overlap_order"
    )
    assert ordered["overlap_pixels"] == int(np.count_nonzero(masks.sum(axis=0) > 1))


def test_nonoverlapping_masks_need_no_order(monkeypatch, tmp_path, capsys):
    result, records = interactive_review(
        monkeypatch, tmp_path,
        ["g 12 0", "g 13 1", "order 12 13", "a"],
        initial=[candidates()[0], candidates()[2]],
    )
    assert "ordering is unnecessary" in capsys.readouterr().out
    assert result.decision == "accepted"
    assert result.depth_order == []
    assert not any(record.get("action") == "overlap_order" for record in records)


def test_impossible_order_rejects_image_without_saving(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    setup_manifest(tmp_path, [instance("a", "12"), instance("b", "13")])
    monkeypatch.setattr(sys, "argv", ["annotate.py", "images", "--data-dir", "."])

    class Backend:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, images, prompts):
            return [candidates()[:2] for _ in images]

    monkeypatch.setattr(annotate, "Sam3ImageBackend", Backend)
    monkeypatch.setattr("builtins.input", lambda _: "order impossible")
    monkeypatch.setattr(annotate, "show_candidates", lambda *_: None)
    annotate.main()
    assert not list(tmp_path.rglob("*.npz"))
    assert all(
        not row["Annotation Complete"] for row in read_csv("image_object_manifest.csv")
    )
    records = [
        json.loads(line) for line in Path("image_review.jsonl").read_text().splitlines()
    ]
    assert records[0]["action"] == "incompatible_depth_order"
    assert records[-1]["decision"] == "rejected"


def test_failed_reorder_retains_previous_result(monkeypatch, tmp_path, capsys):
    small = candidates()[0].mask
    large = small | candidates()[1].mask
    result, _ = interactive_review(
        monkeypatch,
        tmp_path,
        ["g 12 0", "g 13 1", "order 12 13", "order 13 12", "a"],
        initial=[
            Candidate(small, 0.9, "tools", "a"), Candidate(large, 0.9, "tools", "b")
        ],
    )
    assert "Ordering would erase tool 12" in capsys.readouterr().out
    assert result.depth_order == ["12", "13"]
    np.testing.assert_array_equal(result.selected[0].mask, small)
    np.testing.assert_array_equal(result.selected[1].mask, large & ~small)


def test_ordered_masks_recompute_boxes_and_preserve_ignore_regions(
    monkeypatch, tmp_path
):
    top = np.zeros((10, 12), dtype=np.bool_)
    back = top.copy()
    top[1:6, 1:5] = True
    back[1:6, 3:8] = True
    ignore = top.copy()
    ignore[9, 11] = True
    result, _ = interactive_review(
        monkeypatch,
        tmp_path,
        ["g 12 0", "g 13 1", "i 2", "order 12 13", "a"],
        initial=[
            Candidate(mask, 0.9, "tools", str(index))
            for index, mask in enumerate([top, back, ignore])
        ],
    )
    rows = annotate.resolved_image_rows(
        result.instances, result.selected, Path("masks.npz")
    )
    fields = ["BBox X", "BBox Y", "BBox Width", "BBox Height"]
    assert [rows[1][field] for field in fields] == [5, 1, 3, 5]
    np.testing.assert_array_equal(result.ignored[0].mask, ignore)


def test_inspect_opens_automatically_after_order_and_reopens_without_editing(
    monkeypatch, tmp_path
):
    inspections = []
    result, records = interactive_review(
        monkeypatch,
        tmp_path,
        ["g 12 0,2", "g 13 1", "order 13 12", "inspect", "a"],
        inspect_order=lambda image, tools, title: inspections.append(tools),
    )
    assert len(inspections) == 2
    assert result.decision == "accepted"
    assert result.depth_order == ["13", "12"]
    for previews in inspections:
        np.testing.assert_array_equal(previews[0].mask, result.selected[0].mask)
        np.testing.assert_array_equal(
            previews[0].original, candidates()[0].mask | candidates()[2].mask
        )
        assert previews[0].disputed.any()
    assert sum(record.get("action") == "inspect_order" for record in records) == 2
    assert not any(record.get("action") == "overlap_order_invalidated" for record in records)


def test_inspect_requires_an_order_and_never_accepts_by_itself(monkeypatch, tmp_path, capsys):
    inspections = []
    result, _ = interactive_review(
        monkeypatch,
        tmp_path,
        ["inspect", "g 12 0,2", "g 13 1", "order 12 13", "inspect", "q"],
        inspect_order=lambda *args: inspections.append(args),
    )
    assert "Use inspect after setting" in capsys.readouterr().out
    assert len(inspections) == 2
    assert result.decision == "quit"
