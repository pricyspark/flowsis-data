import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

import annotate
from annotations import IMAGE_OBJECT_FIELDS, read_csv, write_csv
from image_refinement import refine_tool, tool_crop
from image_review import Candidate, review_image_instances
from image_ui import ToolPreview, show_tool_review
from masks import load_mask_bundle


def candidate(shape, box, identity="internal-tool"):
    mask = np.zeros(shape, dtype=np.bool_)
    left, top, right, bottom = box
    mask[top:bottom, left:right] = True
    return Candidate(mask, 0.9, "scissors", identity)


def test_refine_assigned_tool_uses_original_pixels_and_full_image_coordinates():
    pixels = np.arange(100 * 120 * 3, dtype=np.uint8).reshape(100, 120, 3)
    seed = candidate((100, 120), (70, 30, 90, 40))
    calls = []

    def predict(crop, prompt):
        calls.append((np.asarray(crop), prompt))
        return [candidate((18, 28), (3, 3, 25, 15))]

    result, message = refine_tool(
        Image.fromarray(pixels), seed, "iris scissors", predict
    )
    assert result is not None
    np.testing.assert_array_equal(calls[0][0], pixels[26:44, 66:94])
    assert calls[0][1] == "iris scissors"
    np.testing.assert_array_equal(
        result.mask, candidate((100, 120), (69, 29, 91, 41)).mask
    )
    assert result.identity == seed.identity
    assert "inspect" in message


@pytest.mark.parametrize("failure", ["empty", "ambiguous", "wrong-tool", "fragment"])
def test_unreliable_refinements_leave_assignment_intact(failure):
    seed = candidate((100, 120), (70, 30, 90, 40))
    matches = {
        "empty": [],
        "ambiguous": [candidate((18, 28), (3, 3, 25, 15))] * 2,
        "wrong-tool": [candidate((18, 28), (0, 0, 3, 3))],
        "fragment": [candidate((18, 28), (4, 4, 8, 8))],
    }
    before = seed.mask.copy()
    result, message = refine_tool(
        Image.new("RGB", (120, 100)), seed, "scissors", lambda *_: matches[failure]
    )
    assert result is None
    assert "previous mask retained" in message
    np.testing.assert_array_equal(seed.mask, before)


def test_refinement_selects_tool_by_mask_agreement_not_prediction_score():
    seed = candidate((100, 120), (70, 30, 90, 40))
    correct = replace(candidate((18, 28), (3, 3, 25, 15)), score=0.6)
    wrong = replace(candidate((18, 28), (0, 0, 3, 3)), score=0.99)
    result, _ = refine_tool(
        Image.new("RGB", (120, 100)), seed, "scissors", lambda *_: [wrong, correct]
    )
    assert result is not None
    assert result.score == 0.6


def test_refinement_expands_crop_when_mask_touches_interior_edge():
    seed = candidate((100, 120), (70, 30, 90, 40))
    calls = []

    def predict(crop, prompt):
        calls.append(crop.size)
        if len(calls) == 1:
            return [candidate((18, 28), (0, 3, 25, 15))]
        return [candidate((26, 36), (7, 7, 29, 19))]

    result, _ = refine_tool(Image.new("RGB", (120, 100)), seed, "scissors", predict)
    assert calls == [(28, 18), (36, 26)]
    assert result is not None
    np.testing.assert_array_equal(
        result.mask, candidate((100, 120), (69, 29, 91, 41)).mask
    )


def test_refinement_gives_up_when_all_expanded_crops_are_clipped():
    seed = candidate((100, 120), (70, 30, 90, 40))
    sizes = []

    def predict(crop, prompt):
        sizes.append(crop.size)
        margin = (crop.width - 20) // 2
        return [
            candidate((crop.height, crop.width), (0, margin, margin + 20, margin + 10))
        ]

    result, message = refine_tool(
        Image.new("RGB", (120, 100)), seed, "scissors", predict
    )
    assert result is None
    assert len(sizes) == 3
    assert "crop edge" in message


def test_real_image_boundary_is_allowed_and_bad_prediction_shape_is_rejected():
    seed = candidate((100, 120), (0, 30, 20, 40))
    result, _ = refine_tool(
        Image.new("RGB", (120, 100)),
        seed,
        "scissors",
        lambda *_: [candidate((18, 24), (0, 3, 21, 15))],
    )
    assert result is not None and result.mask[:, 0].any()
    with pytest.raises(ValueError, match="dimensions"):
        refine_tool(
            Image.new("RGB", (120, 100)),
            seed,
            "scissors",
            lambda *_: [candidate((10, 10), (1, 1, 3, 3))],
        )


def test_combined_preview_shows_overview_crops_and_before_refinement(monkeypatch):
    pixels = np.arange(100 * 120 * 3, dtype=np.uint8).reshape(100, 120, 3)
    seed = candidate((100, 120), (70, 30, 90, 40))
    current = candidate((100, 120), (69, 29, 91, 41))
    other = candidate((100, 120), (10, 20, 30, 40))
    monkeypatch.setattr(plt, "show", lambda **_: None)
    monkeypatch.setattr(plt, "pause", lambda _: None)
    try:
        show_tool_review(
            Image.fromarray(pixels),
            [seed, other],
            "image",
            [
                ToolPreview("0", "iris scissors", current.mask, seed.mask),
                ToolPreview("3", "rongeur", other.mask),
            ],
        )
        figure = plt.gcf()
        assert len(figure.axes) == 3
        assert "Overview" in figure.axes[0].get_title()
        assert any(text.get_text() == "0: scissors" for text in figure.axes[0].texts)
        assert "Tool 0" in figure.axes[1].get_title()
        assert "Tool 3" in figure.axes[2].get_title()
        bounds = tool_crop(seed.mask | current.mask)
        np.testing.assert_array_equal(
            figure.axes[1].images[0].get_array(),
            np.asarray(Image.fromarray(pixels).crop(bounds)),
        )
        assert not figure.axes[1].collections  # Colored pixels, without mask outlines.
    finally:
        plt.close("all")


def test_reviewer_refines_tools_independently_and_restores_original(monkeypatch):
    image = Image.new("RGB", (120, 100))
    first = candidate((100, 120), (10, 20, 30, 40), "candidate-0")
    second = candidate((100, 120), (70, 50, 90, 70), "candidate-1")
    rows = [
        {
            "Image ID": "image",
            "Instance ID": f"internal-{tool}",
            "Tool ID": tool,
            "Tool Class": "scissors",
            "Tool Name": "iris",
            "Confidence": "1",
        }
        for tool in ["0", "3"]
    ]
    commands = iter(["refine 0", "g 0 0", "g 3 1", "refine all", "restore 0", "a"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    calls, logs, panels = [], [], []

    def refine(seed, prompt):
        calls.append((seed.identity, prompt))
        return replace(seed, mask=np.roll(seed.mask, 1, axis=0)), "refined"

    result = review_image_instances(
        image,
        [first, second],
        rows,
        key="image",
        show=lambda *_: None,
        predict=lambda _: [],
        log=lambda *args, **kwargs: logs.append(kwargs),
        refine=refine,
        show_tools=lambda image, candidates, title, tools: panels.append(tools),
    )
    assert len(calls) == 2  # Unassigned refinement does not call the model.
    assert result.selected[0].identity == "internal-0"
    np.testing.assert_array_equal(result.selected[0].mask, first.mask)
    np.testing.assert_array_equal(
        result.selected[1].mask, np.roll(second.mask, 1, axis=0)
    )
    assert panels[-1][0].original is None
    assert panels[-1][1].original is not None
    assert any(
        log.get("action") == "crop_refinement" and log["tool_id"] == "3" for log in logs
    )


def test_cli_saves_refined_masks_in_original_coordinates(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv",
        ["annotate.py", "images", "--data-dir", ".", "--multi-tool"],
    )
    Path("images/multi").mkdir(parents=True)
    Image.new("RGB", (120, 100)).save("images/multi/tools.png")
    write_csv(
        "image_manifest.csv",
        [{"Image ID": "image", "Filepath": "images/multi/tools.png"}],
        ["Image ID", "Filepath"],
    )
    write_csv(
        "image_object_manifest.csv",
        [
            {
                "Image ID": "image",
                "Instance ID": identity,
                "Tool ID": tool,
                "Tool Class": tool_class,
                "Tool Name": name,
                "Confidence": "1",
            }
            for identity, tool, tool_class, name in [
                ("scissors-id", "0", "scissors", "iris"),
                ("rongeur-id", "3", "pliers", "rongeur"),
            ]
        ],
        IMAGE_OBJECT_FIELDS,
    )
    calls = []

    class Backend:
        def __init__(self, *args, **kwargs):
            self.refine_crop = kwargs["refine_crop"]

        def predict(self, images, prompts):
            calls.extend(
                (image.size, prompt, self.refine_crop)
                for image, prompt in zip(images, prompts)
            )
            output = []
            for image, prompt in zip(images, prompts):
                if image.size == (120, 100):
                    box = (10, 20, 30, 40) if prompt == "scissors" else (70, 50, 90, 70)
                else:
                    assert image.size == (28, 28)
                    box = (3, 3, 25, 25)
                output.append([candidate((image.height, image.width), box)])
            return output

    monkeypatch.setattr(annotate, "Sam3ImageBackend", Backend)
    monkeypatch.setattr(annotate, "show_tool_review", lambda *_: None)
    commands = iter(["g 0 0", "g 3 1", "refine all", "a"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    annotate.main()
    bundle = load_mask_bundle("masks/images/multi/tools.npz")
    assert bundle.identities == ("scissors-id", "rongeur-id")
    assert bundle.masks.shape == (2, 100, 120)
    np.testing.assert_array_equal(
        bundle.masks[0], candidate((100, 120), (9, 19, 31, 41)).mask
    )
    np.testing.assert_array_equal(
        bundle.masks[1], candidate((100, 120), (69, 49, 91, 71)).mask
    )
    assert [row["Reviewed"] for row in read_csv("image_object_manifest.csv")] == [
        "TRUE",
        "TRUE",
    ]
    assert [call[1] for call in calls[-2:]] == ["iris scissors", "rongeur pliers"]
    assert all(not call[2] for call in calls)


@pytest.mark.parametrize(
    "reset", [["g 0 0"], ["u 0", "g 0 0"], ["v 0 absent", "g 0 0"]]
)
def test_assignment_changes_discard_refinement(monkeypatch, reset):
    first = candidate((100, 120), (10, 20, 30, 40), "candidate-0")
    second = candidate((100, 120), (70, 50, 90, 70), "candidate-1")
    rows = [
        {
            "Image ID": "image",
            "Instance ID": f"internal-{tool}",
            "Tool ID": tool,
            "Tool Class": "scissors",
            "Tool Name": "iris",
            "Confidence": "1",
        }
        for tool in ["0", "3"]
    ]
    commands = iter(["g 0 0", "g 3 1", "refine 0", *reset, "a"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    result = review_image_instances(
        Image.new("RGB", (120, 100)),
        [first, second],
        rows,
        key="image",
        show=lambda *_: None,
        predict=lambda _: [],
        log=lambda *args, **kwargs: None,
        refine=lambda seed, prompt: (
            replace(seed, mask=np.roll(seed.mask, 1, axis=0)),
            "refined",
        ),
    )
    np.testing.assert_array_equal(result.selected[0].mask, first.mask)
    np.testing.assert_array_equal(result.selected[1].mask, second.mask)


def test_failed_retry_keeps_last_successful_refinement(monkeypatch):
    seed = candidate((100, 120), (10, 20, 30, 40))
    row = {
        "Image ID": "image",
        "Instance ID": "internal-tool",
        "Tool ID": "0",
        "Tool Class": "scissors",
        "Tool Name": "iris",
        "Confidence": "1",
    }
    commands = iter(["g 0 0", "refine 0", "refine 0", "a"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    refined = replace(seed, mask=np.roll(seed.mask, 1, axis=0))
    replies = iter([(refined, "refined"), (None, "ambiguous; previous mask retained")])
    result = review_image_instances(
        Image.new("RGB", (120, 100)),
        [seed],
        [row],
        key="image",
        show=lambda *_: None,
        predict=lambda _: [],
        log=lambda *args, **kwargs: None,
        refine=lambda *_: next(replies),
    )
    np.testing.assert_array_equal(result.selected[0].mask, refined.mask)


def test_unassigned_mask_is_visible_before_assignment_on_gray_background(monkeypatch):
    image = Image.new("RGB", (120, 100), (178, 178, 178))
    mask = candidate((100, 120), (20, 20, 60, 70)).mask
    mask[30:60, 30:50] = False  # A hole inside the bounding box must remain uncolored.
    prediction = Candidate(mask, 0.9, "iris scissors [unassigned]", "candidate-0")
    monkeypatch.setattr(plt, "show", lambda **_: None)
    monkeypatch.setattr(plt, "pause", lambda _: None)
    try:
        show_tool_review(
            image,
            [prediction],
            "image",
            [ToolPreview("0", "iris scissors", np.zeros_like(mask))],
        )
        overview, tool_panel = plt.gcf().axes
        pixels = np.asarray(overview.images[0].get_array())
        assert np.any(pixels[mask] != 178)
        assert np.all(pixels[~mask] == 178)
        assert np.any(pixels[mask, 0] != pixels[mask, 2])  # Colored rather than gray.
        assert not overview.collections  # Colored mask pixels, without an outline.
        assert overview.patches[0].get_linestyle() == ":"
        assert "0: iris scissors" in [text.get_text() for text in overview.texts]
        assert any(
            "No visible mask assigned" == text.get_text() for text in tool_panel.texts
        )
    finally:
        plt.close("all")


def test_candidate_palette_does_not_repeat_every_ten_or_twenty_masks():
    from image_ui import candidate_color

    colors = [candidate_color(index) for index in range(64)]
    assert len(set(colors)) == 64
    assert colors[0] != colors[10] and colors[0] != colors[20]
    assert all(0 <= channel <= 1 for color in colors for channel in color)


def test_tool_preview_color_matches_candidate_and_survives_other_removal(monkeypatch):
    from image_ui import candidate_color

    first = candidate((100, 120), (10, 20, 30, 40))
    second = candidate((100, 120), (70, 50, 90, 70))
    image = Image.new("RGB", (120, 100), (178, 178, 178))
    monkeypatch.setattr(plt, "show", lambda **_: None)
    monkeypatch.setattr(plt, "pause", lambda _: None)
    try:
        show_tool_review(
            image,
            [first, second],
            "image",
            [
                ToolPreview("0", "scissors", first.mask, candidate_index=0),
                ToolPreview("3", "pliers", second.mask, candidate_index=1),
            ],
        )
        before = np.array(plt.gcf().axes[0].images[0].get_array())
        np.testing.assert_allclose(
            plt.gcf().axes[2].title.get_color(), candidate_color(1)
        )
        overlay = np.asarray(plt.gcf().axes[2].images[1].get_array())
        np.testing.assert_array_equal(
            overlay[overlay[:, :, 3] > 0][0, :3],
            [round(value * 255) for value in candidate_color(1)],
        )
        plt.close("all")
        removed = replace(first, mask=np.zeros_like(first.mask), prompt="[removed]")
        show_tool_review(
            image,
            [removed, second],
            "image",
            [ToolPreview("3", "pliers", second.mask, candidate_index=1)],
        )
        after = np.asarray(plt.gcf().axes[0].images[0].get_array())
        np.testing.assert_array_equal(before[second.mask], after[second.mask])
        np.testing.assert_allclose(
            plt.gcf().axes[1].title.get_color(), candidate_color(1)
        )
    finally:
        plt.close("all")


def test_order_preview_outlines_disputes_without_hiding_resolved_masks(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda **_: None)
    monkeypatch.setattr(plt, "pause", lambda _: None)
    original = candidate((100, 120), (20, 20, 60, 60)).mask
    disputed = np.zeros_like(original)
    disputed[30:40, 50:60] = True
    resolved = original & ~disputed
    try:
        show_tool_review(
            Image.new("RGB", (120, 100)),
            [],
            "ordered",
            [
                ToolPreview(
                    "12", "scissors", resolved, original,
                    edit_note="Ordered mask — order reset to undo",
                    disputed=disputed,
                )
            ],
        )
        overview, panel = plt.gcf().axes
        assert len(overview.collections) == 1
        assert len(panel.collections) == 1
        assert any("Magenta outline" in text.get_text() for text in panel.texts)
        assert any("order reset" in text.get_text() for text in panel.texts)
        overlay = np.asarray(panel.images[1].get_array())
        assert np.count_nonzero(overlay[:, :, 3]) == np.count_nonzero(resolved)
    finally:
        plt.close("all")
