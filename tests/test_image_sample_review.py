import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

import annotate
import image_sample_review as sample_review
from annotations import IMAGE_OBJECT_FIELDS, read_csv, write_csv
from image_review import Candidate
from masks import load_mask_bundle


def candidate(image, prompt="scissors"):
    mask = np.zeros((image.height, image.width), dtype=np.bool_)
    mask[1:4, 2:5] = True
    return Candidate(mask, 0.9, prompt, "predicted")


def sample_images(count=3):
    images = [Image.new("RGB", (12 + index, 10)) for index in range(count)]
    return [
        sample_review.SampleImage(
            image, f"images/{index}.jpg", [candidate(image)], "scissors"
        )
        for index, image in enumerate(images)
    ]


class Backend:
    def __init__(self):
        self.calls = []
        self.spatial_calls = []

    def predict(self, images, prompts):
        self.calls.append((list(images), list(prompts)))
        return [
            [candidate(image, prompt)]
            for image, prompt in zip(images, prompts, strict=True)
        ]

    def spatial(self, image, points, seed=None):
        self.spatial_calls.append((image, points, seed))
        return candidate(image, "spatial")


def run_review(monkeypatch, items, commands, backend=None, **kwargs):
    backend = backend or Backend()
    responses = iter(commands)
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    monkeypatch.setattr(sample_review, "show_sample", lambda *_: None)
    quit_requested = sample_review.review_sample(items, "7", backend, **kwargs)
    return backend, quit_requested


def test_prompt_sample_then_one_image_and_accept(monkeypatch):
    items = sample_images()
    backend, quit_requested = run_review(
        monkeypatch,
        items,
        ["p Iris Scissors", "p 1 fine scissors", "a"],
        normalize_prompt=str.lower,
        batch_size=2,
    )
    assert not quit_requested
    assert [len(images) for images, _ in backend.calls] == [2, 1, 1]
    assert backend.calls[0][1] == ["iris scissors", "iris scissors"]
    assert backend.calls[2] == ([items[1].image], ["fine scissors"])
    assert [item.prompt for item in items] == [
        "iris scissors", "fine scissors", "iris scissors"
    ]
    assert all(item.decision == "accepted" for item in items)


def test_bulk_accept_is_atomic_and_requires_explicit_ambiguity_resolution(
    monkeypatch, capsys
):
    items = sample_images()
    items[1].candidates.append(candidate(items[1].image))
    _, quit_requested = run_review(monkeypatch, items, ["a", "q"])
    assert quit_requested
    assert all(item.decision == "pending" for item in items)
    assert "exactly one" in capsys.readouterr().out
    run_review(monkeypatch, items, ["a 1 1", "a"])
    assert all(item.decision == "accepted" for item in items)
    assert items[1].selected == [items[1].candidates[1]]


def test_invalid_targets_and_empty_predictions_preserve_previous_candidates(
    monkeypatch,
):
    items = sample_images()
    previous = items[0].candidates
    items[1].candidates = []
    _, quit_requested = run_review(
        monkeypatch, items, ["p 3 scissors", "p -1 scissors", "p 0", "a", "a 1", "q"]
    )
    assert quit_requested
    assert items[0].candidates is previous
    assert all(item.decision == "pending" for item in items)


def test_image_decisions_ignore_and_visibility(monkeypatch):
    items = sample_images(4)
    items[0].candidates.append(candidate(items[0].image))
    run_review(monkeypatch, items, ["i 0 1", "v 0 partial 0", "e 1", "r 2", "s 3", "a"])
    assert [item.decision for item in items] == [
        "accepted", "accepted", "rejected", "skipped"
    ]
    assert items[0].selected[0].visibility == "partial"
    assert items[0].ignored == {1}
    assert items[1].selected == []


@pytest.mark.parametrize("command", ["clicks", "draw"])
def test_spatial_prompts_require_image_number_and_use_original_image(
    monkeypatch, command
):
    items = sample_images()
    mask = candidate(items[1].image).mask
    monkeypatch.setattr(
        sample_review, "collect_clicks", lambda image, *args: [(3.0, 2.0, 1)]
    )
    monkeypatch.setattr(sample_review, "draw_tool_mask", lambda image, *args: mask)
    backend, _ = run_review(monkeypatch, items, [command, f"{command} 1", "a"])
    assert len(backend.spatial_calls) == 1
    image, points, seed = backend.spatial_calls[0]
    assert image is items[1].image
    if command == "clicks":
        assert points == [(3.0, 2.0, 1)]
        assert seed is None
    else:
        assert points == []
        assert seed is mask
    assert [item.prompt for item in items] == ["scissors", "spatial", "scissors"]


def test_cancelled_and_failed_spatial_prompts_keep_previous_mask(monkeypatch):
    items = sample_images(1)
    original = items[0].candidates[0]
    responses = iter([None, [(3.0, 2.0, 1)]])
    monkeypatch.setattr(sample_review, "collect_clicks", lambda *args: next(responses))
    backend = Backend()

    def fail(*args):
        raise ValueError("No reliable mask")

    monkeypatch.setattr(backend, "spatial", fail)
    run_review(monkeypatch, items, ["clicks 0", "clicks 0", "a"], backend)
    assert items[0].selected[0] is original


def test_preview_numbers_images_and_mask_candidates(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda **kwargs: None)
    monkeypatch.setattr(plt, "pause", lambda *_: None)
    items = sample_images(5)
    sample_review.show_sample(items, "7")
    figure = plt.gcf()
    assert len(figure.axes) == 8
    for index, axis in enumerate(figure.axes[:5]):
        assert axis.get_title().startswith(f"Image {index} | pending")
        assert axis.texts[0].get_text().startswith("mask 0:")
    plt.close(figure)


@pytest.mark.parametrize(
    "commands",
    [
        ["p 1 revised scissors", "a 1", "q"],
        ["focus 1", "p revised scissors", "a", "q"],
    ],
)
def test_cli_groups_interleaved_sample_and_saves_partial_acceptance_on_quit(
    tmp_path, monkeypatch, commands
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["annotate.py", "images", "--data-dir", ".", "--batch-size", "1"],
    )
    rows, objects = [], []
    # The same sample crosses both manifest order and the inference batch limit.
    for index, sample_id in enumerate(["10", "11", "10"]):
        filepath = f"images/7/{index}.jpg"
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (12, 10)).save(filepath)
        key = f"image-{index}"
        rows.append({"Image ID": key, "Sample ID": sample_id, "Filepath": filepath})
        objects.append(
            {
                "Image ID": key,
                "Instance ID": f"declared-{index}",
                "Tool ID": "7",
                "Tool Class": "scissors",
                "Tool Name": "iris",
                "Confidence": "1",
            }
        )
    write_csv("image_manifest.csv", rows, ["Image ID", "Sample ID", "Filepath"])
    write_csv("image_object_manifest.csv", objects, IMAGE_OBJECT_FIELDS)
    backend = Backend()
    monkeypatch.setattr(annotate, "Sam3ImageBackend", lambda *args, **kwargs: backend)
    monkeypatch.setattr(sample_review, "show_sample", lambda *_: None)
    commands = iter(commands)
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    annotate.main()
    assert len(backend.calls) == 3
    assert backend.calls[-1][1] == ["revised scissors"]
    assert not Path("masks/images/7/0.npz").exists()
    assert not Path("masks/images/7/1.npz").exists()
    assert load_mask_bundle("masks/images/7/2.npz").identities == ("declared-2",)
    log = annotate.ReviewLog(Path("image_review.jsonl"))
    assert set(log.decisions) == {"image-2"}
    assert log.decisions["image-2"]["prompt"] == "revised scissors"
    assert read_csv("image_object_manifest.csv")[2]["Reviewed"] == "TRUE"
    # Resume skips the accepted image and processes the other samples.
    commands = iter(["a", "a"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    annotate.main()
    log = annotate.ReviewLog(Path("image_review.jsonl"))
    assert all(log.completed(row["Image ID"]) for row in rows)
    assert len(backend.calls) == 5


def test_cleanup_is_scoped_to_image_and_undo_restores_mask(monkeypatch):
    items = sample_images(2)
    items[0].candidates[0].mask[2, 3] = False
    original = items[0].candidates[0]
    untouched = items[1].candidates[0]
    snapshots = []
    commands = iter(["clean 0 holes 2", "unclean 0", "a"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    monkeypatch.setattr(
        sample_review,
        "show_sample",
        lambda images, _: snapshots.append(bool(images[0].candidates[0].mask[2, 3])),
    )
    sample_review.review_sample(items, "7", Backend())
    assert snapshots == [False, True, False]
    assert items[0].selected[0] is original
    assert items[1].selected[0] is untouched


def test_preview_shows_only_selected_mask_on_accepted_image(monkeypatch):
    items = sample_images(1)
    items[0].candidates.append(candidate(items[0].image))
    items[0].selected = [items[0].candidates[1]]
    items[0].decision = "accepted"
    monkeypatch.setattr(plt, "show", lambda **kwargs: None)
    monkeypatch.setattr(plt, "pause", lambda *_: None)
    sample_review.show_sample(items, "7")
    axis = plt.gcf().axes[0]
    assert len(axis.images) == 2  # Source image and the selected mask only.
    assert axis.texts[0].get_text().startswith("mask 1:")
    plt.close("all")


def test_grouping_keeps_sample_together_across_skipped_priority():
    # Queue already sorts new images ahead of skipped images. The skipped view
    # joins the first sample while unrelated samples retain their queue order.
    queue = [
        {"Image ID": "new-a", "Sample ID": "1"},
        {"Image ID": "new-b", "Sample ID": "2"},
        {"Image ID": "multi", "Sample ID": "3"},
        {"Image ID": "skip-a", "Sample ID": "1"},
        {"Image ID": "ungrouped", "Sample ID": ""},
    ]
    sources = {
        row["Image ID"]: [{}] * (2 if row["Image ID"] == "multi" else 1)
        for row in queue
    }
    batches = annotate.image_review_batches(queue, sources, 4)
    assert [(sample, [row["Image ID"] for row in rows]) for sample, rows in batches] == [
        ("1", ["new-a", "skip-a"]),
        ("2", ["new-b"]),
        (None, ["multi", "ungrouped"]),
    ]


def test_focus_edits_only_one_image_and_returns_to_sample(monkeypatch):
    items = sample_images()
    accepted = items[0].candidates[0]
    views = []
    backend = Backend()
    commands = iter(["a 0", "focus 1", "p fine scissors", "a", "q"])
    monkeypatch.setattr(
        sample_review, "show_sample",
        lambda images, sample, focus=None: views.append(focus),
    )
    quit_requested = sample_review.review_sample(
        items, "7", backend, read_command=lambda _: next(commands)
    )
    assert quit_requested
    assert views == [None, None, 1, 1, None]
    assert backend.calls == [([items[1].image], ["fine scissors"])]
    assert items[0].selected == [accepted]
    assert [item.decision for item in items] == ["accepted", "accepted", "pending"]


def test_sample_reprompt_preserves_accepted_and_skipped_images(monkeypatch):
    items = sample_images(4)
    accepted = items[0].candidates[0]
    skipped = items[1].candidates[0]
    backend, _ = run_review(
        monkeypatch, items, ["a 0", "s 1", "p fine scissors", "q"]
    )
    assert backend.calls == [
        ([items[2].image, items[3].image], ["fine scissors", "fine scissors"])
    ]
    assert items[0].selected == [accepted]
    assert items[1].candidates[0] is skipped
    assert [item.decision for item in items] == [
        "accepted", "skipped", "pending", "pending"
    ]


def test_focused_accept_uses_mask_number_and_back_keeps_edits(monkeypatch):
    items = sample_images()
    items[1].candidates.append(candidate(items[1].image))
    backend, _ = run_review(
        monkeypatch, items,
        ["focus 1", "a 1", "focus 2", "p revised", "back", "q"],
    )
    assert items[1].selected == [items[1].candidates[1]]
    assert items[2].prompt == "revised" and items[2].decision == "pending"
    assert len(backend.calls) == 1


def test_focus_preview_keeps_original_image_number(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda **kwargs: None)
    monkeypatch.setattr(plt, "pause", lambda *_: None)
    items = sample_images()
    sample_review.show_sample(items, "7", 2)
    figure = plt.gcf()
    assert len(figure.axes) == 1
    assert figure.axes[0].get_title().startswith("Image 2 | pending")
    plt.close(figure)
