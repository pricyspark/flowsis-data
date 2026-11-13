import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

import annotate
from annotations import IMAGE_OBJECT_FIELDS, write_csv
from image_review import Candidate, ClickPrompt, review_image_instances
from image_review_canvas import CommandInput, MultiToolCanvas
from image_review_state import ReviewHistory, ReviewState, decode_state, encode_state, state_path
from image_ui import ToolPreview
from masks import load_mask_bundle


def instances():
    return [
        {"Image ID": "image", "Instance ID": f"instance-{tool}", "Tool ID": tool,
         "Tool Name": name, "Tool Class": "forceps", "Confidence": "1"}
        for tool, name in [("0", "crile"), ("3", "adson")]
    ]


def candidates():
    masks = np.zeros((2, 20, 20), dtype=np.bool_)
    masks[0, 2:8, 2:8] = True
    masks[1, 11:17, 11:17] = True
    return [Candidate(mask, 0.9, "forceps", f"candidate-{index}") for index, mask in enumerate(masks)]


def review(monkeypatch, commands, history=None, **kwargs):
    replies = iter(commands)
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    return review_image_instances(
        Image.new("RGB", (20, 20)), kwargs.pop("candidates", candidates()), instances(),
        key="image", show=lambda *_: None, predict=lambda _: [], log=lambda *args, **kw: None,
        history=history, **kwargs,
    )


def test_checkpoint_roundtrip_and_prompt_aliases(tmp_path):
    source = candidates()
    prompt = ClickPrompt("instance-0", [(3.0, 3.0, 1)], source[0].mask, 0, True)
    state = ReviewState(source, instances(), shape=(20, 20),
                        assignments={"instance-0": [0]}, cleaned={"instance-0": source[0]},
                        active_prompts={"instance-0": prompt}, click_prompts={0: prompt},
                        ownership={"instance-0": source[0].mask.copy()}, locked={"instance-0"})
    encoded = encode_state(state)
    assert encoded == encode_state(state)
    restored = decode_state(encoded)
    assert restored.active_prompts["instance-0"] is restored.click_prompts[0]
    np.testing.assert_array_equal(restored.cleaned["instance-0"].mask, source[0].mask)
    path = tmp_path / "draft.npz"
    history = ReviewHistory(path, "pixels/model/identities")
    history.commit(state)
    state.assignments["instance-3"] = [1]
    history.commit(state)
    history.move(-1)
    resumed = ReviewHistory(path, "pixels/model/identities")
    assert resumed.current.assignments == {"instance-0": [0]}
    assert resumed.move(1).assignments["instance-3"] == [1]
    assert ReviewHistory(path, "changed-image").current is None
    with np.load(path, allow_pickle=False) as arrays:
        assert arrays["version"] == 1


def test_reset_retains_original_predictions_after_history_limit(monkeypatch, tmp_path):
    history = ReviewHistory(tmp_path / "draft.npz", "same")
    state = ReviewState(candidates(), instances(), shape=(20, 20))
    history.commit(state)
    for index in range(50):
        state.states["instance-0"] = str(index)
        history.commit(state)
    assert len(history.frames) == 40
    resumed = ReviewHistory(history.path, "same")
    review(monkeypatch, ["reset", "q"], resumed)
    assert resumed.current.states == {}
    assert len(resumed.current.candidates) == 2
    review(monkeypatch, ["undo", "q"], resumed)
    assert resumed.current.states["instance-0"] == "49"


def test_local_edit_undo_redo_and_reset_restore_full_state(monkeypatch):
    history = ReviewHistory()
    review(monkeypatch, ["g 0 0", "g 3 1", "lock 3", "brush 0 erase 1 4,4", "q"], history)
    edited = history.current.candidates[-1].mask.copy()
    assert not edited[4, 4]
    review(monkeypatch, ["undo", "redo", "reset", "q"], history)
    assert history.current.assignments == {}
    assert history.current.locked == set()
    assert len(history.current.candidates) == 2
    result = review(monkeypatch, ["undo", "a"], history)
    np.testing.assert_array_equal(result.selected[0].mask, edited)
    assert history.current.locked == {"instance-3"}


def test_add_piece_keeps_existing_pixels_and_correction_receives_mask_seed(monkeypatch):
    history = ReviewHistory()
    calls = []
    piece = np.zeros((20, 20), dtype=np.bool_)
    piece[2:4, 15:17] = True

    def spatial(points, seed):
        calls.append((points, None if seed is None else seed.copy()))
        mask = piece.copy() if seed is None else seed.copy()
        if seed is not None:
            mask[4, 4] = False
        return Candidate(mask, 0.8, "points", "spatial")

    result = review(monkeypatch, ["g 0 0", "g 3 1", "add 0 15,2,1", "correct 0 4,4,0", "a"], history, spatial=spatial)
    assert calls[0][1] is None
    assert calls[1][1][2, 15] and calls[1][1][3, 3]
    assert result.selected[0].mask[2, 15]
    assert not result.selected[0].mask[4, 4]
    assert result.selected[0].mask[3, 3]


def test_failed_and_cancelled_edits_do_not_change_locks_order_or_history(monkeypatch):
    first, second = candidates()
    second.mask[4:7, 4:12] = True
    history = ReviewHistory()
    review(monkeypatch, ["g 0 0", "g 3 1", "order 0 3", "q"], history, candidates=[first, second])
    before = encode_state(history.current)
    count = len(history.frames)
    review(monkeypatch, ["draw 0", "box 0 -1 0 8 8", "g 0 99", "q"], history,
           draw=lambda *_: None, box_spatial=lambda _: pytest.fail("Invalid box should not run SAM3"))
    assert encode_state(history.current) == before
    assert len(history.frames) == count


def test_local_crossings_can_have_different_owners_and_edits_keep_other_crossings(monkeypatch):
    first, second = candidates()
    second.mask[4:6, 4:6] = True
    first.mask[12:14, 12:14] = True
    history = ReviewHistory()
    result = review(monkeypatch,
        ["g 0 0", "g 3 1", "own 0 4 4", "own 3 12 12", "brush 0 erase 1 2,2", "a"],
        history, candidates=[first, second])
    assert result.selected[0].mask[4, 4] and not result.selected[1].mask[4, 4]
    assert result.selected[1].mask[12, 12] and not result.selected[0].mask[12, 12]
    assert not (result.selected[0].mask & result.selected[1].mask).any()
    # Reopening and undoing the boundary edit retains both crossing decisions.
    review(monkeypatch, ["undo", "q"], history)
    assert history.current.ownership["instance-3"][12, 12]


def test_locked_tools_cannot_be_edited_or_have_their_pixels_claimed(monkeypatch):
    history = ReviewHistory()
    result = review(monkeypatch,
        ["g 0 0", "g 3 1", "lock 3", "u 3", "remove 1", "e", "paint 0 add 10,10 18,10 18,18 10,18", "a"], history)
    np.testing.assert_array_equal(result.selected[1].mask, candidates()[1].mask)
    assert not (result.selected[0].mask & result.selected[1].mask).any()
    assert history.current.locked == {"instance-3"}


def test_unused_proposals_require_preview_and_explicit_discard(monkeypatch):
    history = ReviewHistory()
    extra = Candidate(np.eye(20, dtype=np.bool_), 0.6, "other", "candidate-2")
    review(monkeypatch, ["g 0 0", "g 3 1", "discard", "q"], history, candidates=[*candidates(), extra])
    assert 2 not in history.current.excluded
    result = review(monkeypatch, ["discard confirm", "a"], history)
    assert result.decision == "accepted"
    assert 2 in history.current.excluded
    review(monkeypatch, ["undo", "q"], history)
    assert 2 not in history.current.excluded


def test_corrupt_checkpoint_stops_without_overwriting_it(tmp_path):
    path = tmp_path / "draft.npz"
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Cannot restore"):
        ReviewHistory(path, "same")
    assert path.read_bytes() == b"corrupt"


def test_canvas_preserves_window_tool_selection_and_zoom_and_emits_local_commands(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda **kw: None)
    monkeypatch.setattr(plt, "pause", lambda *_: None)
    canvas = MultiToolCanvas()
    source = candidates()
    tools = [ToolPreview(str(tool), "forceps", item.mask, candidates=(index,)) for index, (tool, item) in enumerate(zip([0, 3], source))]
    image = Image.new("RGB", (20, 20))
    canvas.show(image, source, "image", tools)
    figure = canvas.figure
    canvas._select_tool("3: forceps")
    canvas.overview.set_xlim(8, 18)
    canvas.overview.set_ylim(18, 8)
    canvas.show(image, source, "image", tools)
    assert canvas.figure is figure and canvas.selected_tool == "3"
    assert canvas.overview.get_xlim() == (8, 18)
    canvas._set_mode("Correct mask")
    canvas.on_press(SimpleNamespace(inaxes=canvas.overview, xdata=12.0, ydata=12.0, button=3))
    canvas.apply()
    assert canvas.commands.get_nowait() == "correct 3 12.0,12.0,0"
    canvas.cancel()
    assert canvas.points == []
    canvas.buttons[4]._observers.process("clicked", None)
    assert canvas.commands.get_nowait() == "reset"
    canvas.close()


def test_cli_restores_draft_without_rerunning_predictions_and_clears_it_on_accept(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["annotate.py", "images", "--data-dir", ".", "--multi-tool"])
    path = Path("images/multi/tools.png")
    path.parent.mkdir(parents=True)
    Image.new("RGB", (20, 20)).save(path)
    write_csv("image_manifest.csv", [{"Image ID": "image", "Filepath": str(path)}], ["Image ID", "Filepath"])
    write_csv("image_object_manifest.csv", instances(), IMAGE_OBJECT_FIELDS)
    calls = []

    class Backend:
        def __init__(self, *args, **kwargs):
            self.refine_crop = False

        def predict(self, images, prompts):
            calls.append(prompts)
            return [[candidates()[0] if prompt.startswith("crile") else candidates()[1]] for prompt in prompts]

    monkeypatch.setattr(annotate, "Sam3ImageBackend", Backend)
    monkeypatch.setattr(MultiToolCanvas, "show", lambda *_: None)
    commands = iter(["g 0 0", "g 3 1", "brush 0 erase 1 4,4", "lock 3", "s"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    annotate.main()
    draft = state_path(Path("image_review.jsonl"), "image")
    assert draft.is_file() and len(calls) == 2
    commands = iter(["reset", "undo", "a"])
    monkeypatch.setattr("builtins.input", lambda _: next(commands))
    annotate.main()
    assert len(calls) == 2
    assert not draft.exists()
    bundle = load_mask_bundle("masks/images/multi/tools.npz")
    assert bundle.identities == ("instance-0", "instance-3")
    assert not bundle.masks[0, 4, 4]
    assert bundle.masks[1, 12, 12]


def test_split_transfers_only_selected_pixels_and_undo_restores_both_tools(monkeypatch):
    history = ReviewHistory()
    review(monkeypatch, ["g 0 0", "g 3 1", "split 0 3 2,2 5,2 5,5 2,5", "q"], history)
    first = history.current.candidates[history.current.assignments["instance-0"][0]].mask
    second = history.current.candidates[history.current.assignments["instance-3"][0]].mask
    assert not first[3, 3] and second[3, 3]
    assert first[6, 6] and second[12, 12]
    result = review(monkeypatch, ["undo", "a"], history)
    for expected, actual in zip(candidates(), result.selected, strict=True):
        np.testing.assert_array_equal(actual.mask, expected.mask)


def test_new_disputed_pixels_need_ownership_but_old_crossings_remain_resolved(monkeypatch):
    first, second = candidates()
    first.mask[12:14, 12:14] = True
    history = ReviewHistory()
    review(monkeypatch, ["g 0 0", "g 3 1", "own 3 12 12", "q"], history, candidates=[first, second])
    result = review(monkeypatch,
        ["paint 3 add 2,2 4,2 4,4 2,4", "a", "own 0 3 3", "a"], history)
    assert result.selected[1].mask[12, 12] and not result.selected[0].mask[12, 12]
    assert result.selected[0].mask[3, 3] and not result.selected[1].mask[3, 3]


def test_saved_click_prompts_reopen_with_their_points_and_seed(monkeypatch, tmp_path):
    history = ReviewHistory(tmp_path / "draft.npz", "same")
    seen = []

    def spatial(points, seed):
        seen.append((points, seed.copy()))
        return Candidate(seed.copy(), 0.8, "points", "spatial")

    review(monkeypatch, ["g 0 0", "correct 0 3,3,1", "q"], history, spatial=spatial)
    loaded = ReviewHistory(history.path, "same")
    initial_points = []

    def collect(image, mask, initial, title):
        initial_points.append(initial)
        return None

    review(monkeypatch, ["click 0", "q"], loaded, collect_points=collect, spatial=spatial)
    assert initial_points == [[(3.0, 3.0, 1)]]
    np.testing.assert_array_equal(loaded.current.active_prompts["instance-0"].seed, candidates()[0].mask)


def test_only_accepted_views_contribute_sample_prompts_and_references(tmp_path):
    log = annotate.ReviewLog(tmp_path / "review.jsonl")
    rows = []
    for index, sample in enumerate(["5", "5", "6"]):
        key = f"image-{index}"
        filepath = f"{index}.png"
        Image.new("RGB", (20, 20)).save(tmp_path / filepath)
        rows.append({"Image ID": key, "Sample ID": sample, "Filepath": filepath})
        log.append(key, "in_progress", action="resolved", assignments={"a": [{"prompt": "metal tweezers"}]})
    log.append("image-0", "accepted")
    log.append("image-2", "accepted")
    assert annotate.successful_sample_prompts(rows, log) == {"5": ["metal tweezers"], "6": ["metal tweezers"]}
    references = annotate.sample_references(tmp_path, rows, rows[1], log)
    assert len(references) == 1
    assert "accepted" in references[0].label and "0.png" in references[0].label


def test_pending_canvas_input_is_consumed_by_next_reviewer(monkeypatch):
    terminal = CommandInput()
    terminal.commands.put("p better prompt")
    monkeypatch.setattr("builtins.input", lambda _: pytest.fail("Duplicate stdin read"))
    assert terminal.read("sample> ") == "p better prompt"
    terminal.available = False
    assert terminal.read("sample> ") == "q"


def test_edit_cannot_restore_disputed_pixels_to_a_locked_tool(monkeypatch, capsys):
    first, second = candidates()
    second.mask[4:6, 4:6] = True
    history = ReviewHistory()
    review(monkeypatch, ["g 0 0", "g 3 1", "order 0 3", "lock 3", "q"],
           history, candidates=[first, second])
    before = encode_state(history.current)
    review(monkeypatch, ["brush 0 erase 1 4,4", "q"], history)
    assert "Edit would change locked tool 3" in capsys.readouterr().out
    assert encode_state(history.current) == before


def test_accepted_single_tool_prompts_are_reused(tmp_path):
    log = annotate.ReviewLog(tmp_path / "review.jsonl")
    log.append("image", "accepted", prompt="fine metal tweezers")
    rows = [{"Image ID": "image", "Sample ID": "5"}]
    assert annotate.successful_sample_prompts(rows, log) == {"5": ["fine metal tweezers"]}


def test_prompt_configuration_changes_invalidate_drafts():
    args = annotate.parse_args(["images"])
    image = Image.new("RGB", (20, 20))
    fingerprint = annotate.review_fingerprint(image, instances(), args)
    args.discovery_prompt = ["thin forceps"]
    assert annotate.review_fingerprint(image, instances(), args) != fingerprint


def test_backend_failure_keeps_the_previous_checkpoint(monkeypatch):
    history = ReviewHistory()
    review(monkeypatch, ["g 0 0", "g 3 1", "q"], history)
    before = encode_state(history.current)

    def fail(*_):
        raise RuntimeError("inference failed")

    with pytest.raises(RuntimeError, match="inference failed"):
        review(monkeypatch, ["click 0"], history, spatial=fail,
               collect_points=lambda *_: [(3.0, 3.0, 1)])
    assert encode_state(history.current) == before
