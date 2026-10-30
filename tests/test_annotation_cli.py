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
@pytest.mark.parametrize("data_dir", [".", "data"])
def test_image_cli_writes_local_outputs_and_resumes(
    tmp_path, monkeypatch, empty, data_dir
):
    dataset = tmp_path / data_dir
    dataset.mkdir(exist_ok=True)
    monkeypatch.chdir(dataset)
    argv = ["annotate.py", "images", "--auto-accept"]
    if data_dir == ".":
        argv.extend(["--data-dir", "."])
    monkeypatch.setattr(sys, "argv", argv)
    Path("images/7").mkdir(parents=True)
    Image.new("RGB", (12, 10)).save("images/7/tool.jpg")
    with Path("image_manifest.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["Filepath", "Image ID"])
        writer.writeheader()
        writer.writerow(
            {
                "Filepath": "images/7/tool.jpg",
                "Image ID": stable_media_id("image", "images/7/tool.jpg"),
            }
        )
    write_csv(
        "image_object_manifest.csv",
        [
            {
                "Image ID": stable_media_id("image", "images/7/tool.jpg"),
                "Instance ID": "declared-0",
                "Tool ID": "7",
                "Tool Name": "iris",
                "Tool Class": "scissors",
                "Confidence": "0.8",
            }
        ],
        IMAGE_OBJECT_FIELDS,
    )
    calls = []

    class Backend:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, images, prompts):
            calls.append(prompts)
            mask = np.zeros((10, 12), dtype=np.bool_)
            mask[2:8, 3:9] = True
            candidates = (
                []
                if empty
                else [annotate.Candidate(mask, 0.9, prompts[0], "instance-0")]
            )
            return [candidates]

    monkeypatch.setattr(annotate, "Sam3ImageBackend", Backend)
    with monkeypatch.context() as context:
        context.chdir(tmp_path)
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
    with monkeypatch.context() as context:
        context.chdir(tmp_path)
        annotate.main()
    assert calls == [["scissors"]]


@pytest.mark.parametrize("skipped_first", [False, True])
def test_skipped_priority_is_stable_across_batches(
    tmp_path, monkeypatch, skipped_first
):
    monkeypatch.chdir(tmp_path)
    argv = [
        "annotate.py",
        "images",
        "--data-dir",
        ".",
        "--single-tool",
        "--batch-size",
        "2",
    ]
    if skipped_first:
        argv.append("--skipped-first")
    monkeypatch.setattr(sys, "argv", argv)
    Path("images").mkdir()
    keys = ["skip-a", "new-a", "accepted", "skip-b", "new-b", "rejected", "missing"]
    rows = [{"Image ID": key, "Filepath": f"images/{key}.jpg"} for key in keys]
    for row in rows[:-1]:
        Image.new("RGB", (8, 8)).save(row["Filepath"])
    write_csv("image_manifest.csv", rows, ["Image ID", "Filepath"])
    objects = [
        {
            "Image ID": key,
            "Instance ID": "instance-0",
            "Tool ID": "7",
            "Tool Class": "scissors",
            "Tool Name": "iris",
            "Confidence": "1",
        }
        for key in keys
    ]
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
        ["skip-a", "skip-b", "new-a", "new-b"]
        if skipped_first
        else ["new-a", "new-b", "skip-a", "skip-b"]
    )
    assert seen == expected
    assert batches == [2, 2]
    assert [row["Image ID"] for row in read_csv("image_manifest.csv")] == keys


@pytest.mark.parametrize(
    "count, commands, selected_index",
    [
        (1, ["a"], 0),
        (2, ["a", "a 0,1", "a 1"], 1),
        (0, ["a", "p lineman pliers", "a"], 0),
        (0, ["p alternatives", "a 1"], 1),
    ],
)
def test_default_single_tool_review_never_requires_instance_commands(
    tmp_path, monkeypatch, capsys, count, commands, selected_index
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["annotate.py", "images", "--data-dir", "."])
    Path("images/7").mkdir(parents=True)
    Image.new("RGB", (12, 10)).save("images/7/tool.jpg")
    write_csv(
        "image_manifest.csv",
        [
            {
                "Image ID": "image-test",
                "Filepath": "images/7/tool.jpg",
            }
        ],
        ["Image ID", "Filepath"],
    )
    write_csv(
        "image_object_manifest.csv",
        [
            {
                "Image ID": "image-test",
                "Instance ID": "internal-tool",
                "Tool ID": "7",
                "Tool Class": "pliers",
                "Tool Name": "lineman",
                "Confidence": "1",
            }
        ],
        IMAGE_OBJECT_FIELDS,
    )
    masks = np.zeros((2, 10, 12), dtype=np.bool_)
    masks[0, 1:4, 2:5] = True
    masks[1, 5:8, 7:10] = True

    class Backend:
        def __init__(self, *args, **kwargs):
            assert not kwargs["single_tool"]

        def predict(self, images, prompts):
            size = {"lineman pliers": 1, "alternatives": 2}.get(prompts[0], count)
            return [
                [
                    annotate.Candidate(mask, 0.9, prompt, f"candidate-{index}")
                    for index, mask in enumerate(masks[:size])
                ]
                for prompt in prompts
            ]

    monkeypatch.setattr(annotate, "Sam3ImageBackend", Backend)
    monkeypatch.setattr(annotate, "show_candidates", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        annotate,
        "review_image_instances",
        lambda *args, **kwargs: pytest.fail(
            "Single-tool image entered instance assignment"
        ),
    )
    responses = iter(commands)
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    annotate.main()
    output = capsys.readouterr().out
    assert "Single-tool review:" in output
    assert "Declared tools" not in output
    assert "internal-tool" not in output
    if count == 2:
        assert "Choose one mask with 'a INDEX'" in output
        assert "Accept exactly one mask" in output
    bundle = load_mask_bundle("masks/images/7/tool.npz")
    assert bundle.identities == ("internal-tool",)
    np.testing.assert_array_equal(bundle.masks[0], masks[selected_index])
    assert annotate.ReviewLog(Path("image_review.jsonl")).completed("image-test")


def test_auto_accept_rejects_multiple_single_tool_predictions():
    masks = np.ones((10, 12), dtype=np.bool_)
    candidates = [
        annotate.Candidate(masks, 0.9, "pliers", f"candidate-{index}")
        for index in range(2)
    ]
    with pytest.raises(ValueError, match="Multiple masks predicted"):
        annotate.review_candidates(
            Image.new("RGB", (12, 10)), candidates, key="image-test", auto_accept=True
        )


@pytest.mark.parametrize("flag", ["--merge-masks", "--single-tool"])
def test_fragment_merging_remains_opt_in(monkeypatch, flag):
    monkeypatch.setattr(sys, "argv", ["annotate.py", "images", "--data-dir", ".", flag])
    assert annotate.parse_args().single_tool
    monkeypatch.setattr(sys, "argv", ["annotate.py", "images", "--data-dir", "."])
    assert not annotate.parse_args().single_tool


@pytest.mark.parametrize("multi_only", [False, True])
def test_default_cli_selects_review_per_image_in_mixed_batch(
    tmp_path, monkeypatch, multi_only
):
    from image_review import ImageReview

    monkeypatch.chdir(tmp_path)
    argv = ["annotate.py", "images", "--data-dir", "."]
    if multi_only:
        argv.append("--multi-tool")
    monkeypatch.setattr(sys, "argv", argv)
    Path("images").mkdir()
    images = []
    objects = []
    for key, tools in [("single", ["7"]), ("multi", ["7", "8"])]:
        filepath = f"images/{key}.jpg"
        Image.new("RGB", (12, 10)).save(filepath)
        images.append({"Image ID": key, "Filepath": filepath})
        objects.extend(
            {
                "Image ID": key,
                "Instance ID": f"internal-{tool}",
                "Tool ID": tool,
                "Tool Class": "pliers",
                "Tool Name": "lineman",
                "Confidence": "1",
            }
            for tool in tools
        )
    write_csv("image_manifest.csv", images, ["Image ID", "Filepath"])
    write_csv("image_object_manifest.csv", objects, IMAGE_OBJECT_FIELDS)

    class Backend:
        def __init__(self, *args, **kwargs):
            assert not kwargs["single_tool"]

        def predict(self, images, prompts):
            mask = np.ones((10, 12), dtype=np.bool_)
            return [
                [
                    annotate.Candidate(mask, 0.9, prompt, f"candidate-{index}")
                    for index in range(2)
                ]
                for prompt in prompts
            ]

    calls = []

    def single_review(image, candidates, *, key, **kwargs):
        calls.append((key, "simple"))
        assert len(candidates) == 2
        return "skipped", [], [], None, "clear"

    def multi_review(image, candidates, instances, *, key, **kwargs):
        calls.append((key, "explicit"))
        assert len(instances) == 2
        return ImageReview("skipped", instances, [], [])

    monkeypatch.setattr(annotate, "Sam3ImageBackend", Backend)
    monkeypatch.setattr(annotate, "review_candidates", single_review)
    monkeypatch.setattr(annotate, "review_image_instances", multi_review)
    annotate.main()
    expected = (
        [("multi", "explicit")]
        if multi_only
        else [("single", "simple"), ("multi", "explicit")]
    )
    assert calls == expected
    if multi_only:
        assert "single" not in annotate.ReviewLog(Path("image_review.jsonl")).decisions


def test_multi_tool_filter_with_no_matches_does_not_load_model(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["annotate.py", "images", "--data-dir", ".", "--multi-tool"],
    )
    write_csv(
        "image_manifest.csv",
        [{"Image ID": "single", "Filepath": "images/single.jpg"}],
        ["Image ID", "Filepath"],
    )
    write_csv(
        "image_object_manifest.csv",
        [
            {
                "Image ID": "single",
                "Instance ID": "instance-0",
                "Tool ID": "7",
                "Tool Class": "pliers",
                "Tool Name": "lineman",
                "Confidence": "1",
            }
        ],
        IMAGE_OBJECT_FIELDS,
    )
    monkeypatch.setattr(
        annotate,
        "Sam3ImageBackend",
        lambda *args, **kwargs: pytest.fail("No model should load"),
    )
    annotate.main()
    assert "No unfinished multi-tool images available" in capsys.readouterr().out
    assert not Path("image_review.jsonl").exists()
