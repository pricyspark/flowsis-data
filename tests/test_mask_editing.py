import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.backend_bases import KeyEvent, MouseEvent
from PIL import Image

import annotate
from annotations import IMAGE_OBJECT_FIELDS, read_csv, write_csv
from image_review import Candidate, review_image_instances
from image_ui import draw_tool_mask, paint_polygon
from masks import load_mask_bundle


def example_masks():
    combined = np.zeros((100, 120), dtype=np.bool_)
    combined[20:80, 20:100] = True
    known = np.zeros_like(combined)
    known[20:80, 20:60] = True
    return combined, known


def review(
    monkeypatch, commands, candidates, *, tools=("0", "3"), draw=None, refine=None
):
    rows = [
        {
            "Image ID": "image",
            "Instance ID": f"internal-{tool}",
            "Tool ID": tool,
            "Tool Class": "scissors",
            "Tool Name": "iris",
            "Confidence": "1",
        }
        for tool in tools
    ]
    replies = iter(commands)
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    logs = []
    result = review_image_instances(
        Image.new("RGB", (120, 100)),
        candidates,
        rows,
        key="image",
        show=lambda *_: None,
        predict=lambda _: [],
        log=lambda *args, **kwargs: logs.append(kwargs),
        draw=draw,
        refine=refine,
    )
    return result, logs


def test_draw_then_subtract_partitions_combined_visible_pixels(monkeypatch):
    combined, known = example_masks()
    source = Candidate(combined, 0.9, "tools", "candidate-0")
    result, logs = review(
        monkeypatch,
        ["draw 3", "subtract 0 0 3", "x 0", "a"],
        [source],
        draw=lambda *_: known.copy(),
    )
    remainder, drawn = result.selected
    assert [mask.identity for mask in result.selected] == ["internal-0", "internal-3"]
    np.testing.assert_array_equal(remainder.mask, combined & ~known)
    np.testing.assert_array_equal(drawn.mask, known)
    assert not (remainder.mask & drawn.mask).any()
    np.testing.assert_array_equal(remainder.mask | drawn.mask, combined)
    np.testing.assert_array_equal(source.mask, combined)
    provenance = next(log for log in logs if log["action"] == "subtract")
    assert provenance["source_index"] == 0
    assert provenance["subtracted_tool_ids"] == ["3"]
    assert provenance["candidate_index"] == 2


def test_subtraction_uses_current_refined_masks_for_multiple_tools(monkeypatch):
    from dataclasses import replace

    combined, known = example_masks()
    third = np.zeros_like(combined)
    third[20:80, 60:80] = True
    refined = known.copy()
    refined[:, 20:22] = False
    candidates = [
        Candidate(mask, 0.9, "tools", f"candidate-{index}")
        for index, mask in enumerate([combined, known, third])
    ]
    result, _ = review(
        monkeypatch,
        ["g 3 1", "g 7 2", "refine 3", "subtract 0 0 3,7", "x 0", "a"],
        candidates,
        tools=("0", "3", "7"),
        refine=lambda seed, _: (replace(seed, mask=refined), "refined"),
    )
    np.testing.assert_array_equal(result.selected[0].mask, combined & ~refined & ~third)
    np.testing.assert_array_equal(result.selected[1].mask, refined)
    np.testing.assert_array_equal(result.selected[2].mask, third)


@pytest.mark.parametrize(
    "command, expected",
    [
        ("subtract 0 0 3", "Assign or draw a correct visible mask"),
        ("subtract 0 0 0", "Cannot subtract the target tool"),
        ("subtract 99 0 3", "Unknown Tool ID"),
        ("subtract 0 0 99", "Unknown Tool ID"),
        ("subtract 0 99 3", "valid comma-separated candidate indices"),
        ("subtract 0 0", "Use draw TOOL_ID or subtract"),
    ],
)
def test_invalid_subtraction_leaves_tools_unassigned(
    monkeypatch, capsys, command, expected
):
    combined, _ = example_masks()
    result, logs = review(
        monkeypatch, [command, "s"], [Candidate(combined, 0.9, "tools", "candidate-0")]
    )
    assert result.decision == "skipped"
    assert expected in capsys.readouterr().out
    assert not any(log["action"] == "subtract" for log in logs)


@pytest.mark.parametrize(
    "kind, expected",
    [("empty", "resulting mask is empty"), ("outside", "nothing was subtracted")],
)
def test_empty_or_unchanged_remainder_is_not_assigned(
    monkeypatch, capsys, kind, expected
):
    combined, _ = example_masks()
    known = combined.copy() if kind == "empty" else np.zeros_like(combined)
    if kind == "outside":
        known[1:5, 1:5] = True
    result, logs = review(
        monkeypatch,
        ["draw 3", "subtract 0 0 3", "s"],
        [Candidate(combined, 0.9, "tools", "candidate-0")],
        draw=lambda *_: known,
    )
    assert result.decision == "skipped"
    assert expected in capsys.readouterr().out
    assert not any(log["action"] == "subtract" for log in logs)


@pytest.mark.parametrize(
    "drawn",
    [None, np.zeros((100, 120), dtype=np.bool_), np.zeros((10, 10), dtype=np.bool_)],
)
def test_cancelled_or_invalid_drawing_does_not_assign(monkeypatch, drawn):
    combined, _ = example_masks()
    result, logs = review(
        monkeypatch,
        ["draw 3", "s"],
        [Candidate(combined, 0.9, "tools", "candidate-0")],
        draw=lambda *_: drawn,
    )
    assert result.decision == "skipped"
    assert not any(log["action"] == "draw" for log in logs)


def test_polygon_painting_supports_disconnected_parts_holes_and_clipping():
    mask = np.zeros((600, 120), dtype=np.bool_)
    paint_polygon(mask, [(-10, -10), (60, -10), (60, 550), (-10, 550)], add=True)
    assert mask[0, 0] and mask[500, 30]
    assert not mask[500, 90]
    paint_polygon(mask, [(10, 10), (30, 10), (30, 30), (10, 30)], add=False)
    assert not mask[20, 20]
    assert mask[40, 40]
    paint_polygon(mask, [(80, 560), (90, 560), (90, 580), (80, 580)], add=True)
    assert mask[570, 85] and not mask[570, 50]
    before = mask.copy()
    paint_polygon(mask, [(200, 700), (300, 700), (300, 800)], add=True)
    np.testing.assert_array_equal(mask, before)


def test_polygon_editor_adds_erases_and_undoes_with_real_mouse_events(monkeypatch):
    initial = np.zeros((100, 120), dtype=np.bool_)

    def interact(**kwargs):
        figure = plt.gcf()
        axis = figure.axes[0]
        figure.canvas.draw()

        def key(value):
            figure.canvas.callbacks.process(
                "key_press_event", KeyEvent("key_press_event", figure.canvas, key=value)
            )

        def polygon(vertices):
            for point in [*vertices, vertices[0]]:
                x, y = axis.transData.transform(point)
                for event in [
                    "motion_notify_event",
                    "button_press_event",
                    "button_release_event",
                ]:
                    figure.canvas.callbacks.process(
                        event, MouseEvent(event, figure.canvas, x, y, button=1)
                    )

        polygon([(10, 10), (40, 10), (40, 40), (10, 40)])
        polygon([(60, 60), (80, 60), (80, 80), (60, 80)])
        key("z")
        key("-")
        polygon([(20, 20), (30, 20), (30, 30), (20, 30)])
        key("enter")

    monkeypatch.setattr(plt, "show", interact)
    result = draw_tool_mask(Image.new("RGB", (120, 100)), initial)
    assert result is not None
    assert result[15, 15] and not result[25, 25] and not result[70, 70]
    assert not initial.any()


@pytest.mark.parametrize("close", [True, False])
def test_polygon_editor_cancel_preserves_input(monkeypatch, close):
    _, initial = example_masks()
    before = initial.copy()

    def interact(**kwargs):
        figure = plt.gcf()
        if close:
            plt.close(figure)
        else:
            figure.canvas.callbacks.process(
                "key_press_event",
                KeyEvent("key_press_event", figure.canvas, key="escape"),
            )

    monkeypatch.setattr(plt, "show", interact)
    assert draw_tool_mask(Image.new("RGB", (120, 100)), initial) is None
    np.testing.assert_array_equal(initial, before)


@pytest.mark.parametrize("method", ["draw", "click", "polygon"])
def test_cli_saves_drawn_and_subtracted_masks_with_stable_tool_identity(
    monkeypatch, tmp_path, method
):
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
                "Instance ID": f"internal-{tool}",
                "Tool ID": tool,
                "Tool Class": tool_class,
                "Tool Name": name,
                "Confidence": "0.9",
            }
            for tool, tool_class, name in [
                ("0", "scissors", "iris"),
                ("3", "pliers", "rongeur"),
            ]
        ],
        IMAGE_OBJECT_FIELDS,
    )
    combined, known = example_masks()

    spatial_calls = []

    class Backend:
        def __init__(self, *args, **kwargs):
            self.refine_crop = False

        def predict(self, images, prompts):
            return [
                (
                    [Candidate(combined.copy(), 0.9, prompt, "candidate-0")]
                    if prompt == "scissors"
                    else []
                )
                for prompt in prompts
            ]

        def spatial(self, image, points, seed):
            spatial_calls.append((points, seed))
            return Candidate(known.copy(), 0.8, "SAM3 spatial prompt", "spatial")

    monkeypatch.setattr(annotate, "Sam3ImageBackend", Backend)
    monkeypatch.setattr(annotate, "show_tool_review", lambda *_: None)
    monkeypatch.setattr(annotate, "draw_tool_mask", lambda *_: known.copy())
    monkeypatch.setattr(
        annotate, "collect_clicks", lambda *_: [(30.0, 30.0, 1), (80.0, 30.0, 0)]
    )
    commands = iter([f"{method} 3", "subtract 0 0 3", "x 0", "a"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    annotate.main()
    bundle = load_mask_bundle("masks/images/multi/tools.npz")
    assert bundle.identities == ("internal-0", "internal-3")
    np.testing.assert_array_equal(bundle.masks[0], combined & ~known)
    np.testing.assert_array_equal(bundle.masks[1], known)
    assert all(
        row["Reviewed"] == "TRUE" for row in read_csv("image_object_manifest.csv")
    )

    assert len(spatial_calls) == (0 if method == "draw" else 1)


def test_editor_requires_closing_an_unfinished_polygon_before_saving(
    monkeypatch, capsys
):
    def interact(**kwargs):
        figure = plt.gcf()
        figure.canvas.draw()
        x, y = figure.axes[0].transData.transform((20, 20))
        for event in [
            "motion_notify_event",
            "button_press_event",
            "button_release_event",
        ]:
            figure.canvas.callbacks.process(
                event, MouseEvent(event, figure.canvas, x, y, button=1)
            )
        figure.canvas.callbacks.process(
            "key_press_event", KeyEvent("key_press_event", figure.canvas, key="enter")
        )
        assert plt.fignum_exists(figure.number)
        figure.canvas.callbacks.process(
            "key_press_event", KeyEvent("key_press_event", figure.canvas, key="escape")
        )

    monkeypatch.setattr(plt, "show", interact)
    assert (
        draw_tool_mask(
            Image.new("RGB", (120, 100)), np.zeros((100, 120), dtype=np.bool_)
        )
        is None
    )
    assert "Close the current polygon" in capsys.readouterr().out


def test_remove_hides_candidates_without_renumbering_new_masks(monkeypatch):
    combined, known = example_masks()
    noise = np.zeros_like(combined)
    noise[:4, :4] = True
    candidates = [
        Candidate(mask, 0.9, "tools", f"candidate-{index}")
        for index, mask in enumerate([combined, known, noise])
    ]
    result, logs = review(
        monkeypatch,
        ["remove 2", "g 3 1", "subtract 0 0 3", "remove 0", "a"],
        candidates,
    )
    np.testing.assert_array_equal(result.selected[0].mask, combined & ~known)
    np.testing.assert_array_equal(result.selected[1].mask, known)
    final = logs[-1]
    assert final["removed"] == [0, 2]
    assert final["assignments"]["internal-0"][0]["index"] == 3
    assert final["assignments"]["internal-3"][0]["index"] == 1


def test_remove_assigned_fragment_clears_whole_assignment_and_refinement(
    monkeypatch, capsys
):
    from dataclasses import replace

    combined, known = example_masks()
    spare = np.zeros_like(combined)
    spare[1:4, 1:4] = True
    candidates = [
        Candidate(mask, 0.9, "tools", f"candidate-{index}")
        for index, mask in enumerate([known, combined & ~known, spare])
    ]
    result, logs = review(
        monkeypatch,
        ["g 0 0,2", "g 3 1", "refine 0", "remove 0", "a", "g 0 2", "a"],
        candidates,
        refine=lambda seed, _: (
            replace(seed, mask=np.roll(seed.mask, 1, axis=0)),
            "refined",
        ),
    )
    removal = next(log for log in logs if log["action"] == "remove_candidates")
    assert "internal-0" not in removal["assignments"]
    assert "internal-0" not in removal["visibility"]
    assert "internal-0" not in removal["refinements"]
    assert "internal-3" in removal["assignments"]
    assert "Cleared assignments for tools 0" in capsys.readouterr().out
    np.testing.assert_array_equal(result.selected[0].mask, spare)


@pytest.mark.parametrize(
    "command", ["g 0 2", "x 2", "i 2", "subtract 0 2 3", "remove 2"]
)
def test_removed_candidates_cannot_be_reused(monkeypatch, capsys, command):
    combined, known = example_masks()
    candidates = [
        Candidate(mask, 0.9, "tools", f"candidate-{index}")
        for index, mask in enumerate([combined, known, known])
    ]
    result, _ = review(monkeypatch, ["remove 2", command, "s"], candidates)
    assert result.decision == "skipped"
    assert "have been removed" in capsys.readouterr().out


def test_removing_ignored_candidate_removes_it_from_final_ignore_mask(monkeypatch):
    combined, known = example_masks()
    candidates = [
        Candidate(mask, 0.9, "tools", f"candidate-{index}")
        for index, mask in enumerate([known, combined & ~known, combined])
    ]
    result, _ = review(
        monkeypatch, ["i 2", "remove 2", "g 0 0", "g 3 1", "a"], candidates
    )
    assert result.decision == "accepted"
    assert not result.ignored


def test_invalid_batch_removal_does_not_partly_remove_valid_candidates(monkeypatch):
    combined, known = example_masks()
    candidates = [
        Candidate(mask, 0.9, "tools", f"candidate-{index}")
        for index, mask in enumerate([known, combined & ~known])
    ]
    result, logs = review(
        monkeypatch, ["remove 0,999", "g 0 0", "g 3 1", "a"], candidates
    )
    assert result.decision == "accepted"
    assert logs[-1]["removed"] == []


def test_removing_every_candidate_preserves_numbers_for_later_drawing(
    monkeypatch, capsys
):
    combined, known = example_masks()
    drawings = iter([known, combined & ~known])
    result, logs = review(
        monkeypatch,
        ["remove 0", "draw 3", "draw 0", "a"],
        [Candidate(combined, 0.9, "tools", "candidate-0")],
        draw=lambda *_: next(drawings),
    )
    assert result.decision == "accepted"
    assert "All candidates were removed" in capsys.readouterr().out
    assert logs[-1]["assignments"]["internal-3"][0]["index"] == 1
    assert logs[-1]["assignments"]["internal-0"][0]["index"] == 2
