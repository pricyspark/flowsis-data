from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from annotate import Sam3ImageBackend
from image_review import Candidate, review_image_instances
from image_spatial import Sam3SpatialBackend


class Batch(dict):
    def to(self, device):
        return self


class Processor:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        result = Batch(original_sizes=torch.tensor([[8, 8]]))
        if kwargs["images"] is not None:
            result["pixel_values"] = torch.zeros((1, 3, 8, 8))
        if kwargs["input_points"] is not None:
            result["input_points"] = torch.tensor(kwargs["input_points"])
            result["input_labels"] = torch.tensor(kwargs["input_labels"])
        return result

    def post_process_masks(self, masks, sizes, binarize):
        assert binarize
        return [masks[0] > 0]


class Model:
    def __init__(self):
        self.prompt_encoder = SimpleNamespace(mask_input_size=(8, 8))
        self.calls = []
        self.moves = []
        self.embeddings = [torch.zeros((1, 1, 2, 2))]

    def to(self, device):
        self.moves.append(str(device))
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        masks = torch.full((1, 1, 3, 8, 8), -10.0)
        masks[0, 0, 0] = (
            10  # Higher score, but incorrectly covers the background click.
        )
        masks[0, 0, 1, :3, :3] = 10
        return SimpleNamespace(
            iou_scores=torch.tensor([[[0.99, 0.8, 0.2]]]),
            pred_masks=masks,
            image_embeddings=self.embeddings,
        )


def backend():
    instance = Sam3SpatialBackend("unused", torch.device("cpu"))
    instance.model, instance.processor = Model(), Processor()
    return instance


def test_spatial_points_choose_a_mask_satisfying_foreground_and_background():
    instance = backend()
    result = instance.segment(Image.new("RGB", (8, 8)), [(1, 1, 1), (5, 5, 0)])
    assert result.mask.shape == (8, 8)
    assert result.mask[1, 1] and not result.mask[5, 5]
    assert result.mask.sum() == 9
    assert result.score == pytest.approx(0.8)
    assert instance.processor.calls[0]["input_points"] == [[[[1, 1], [5, 5]]]]
    assert instance.processor.calls[0]["input_labels"] == [[[1, 0]]]


def test_polygon_is_passed_to_decoder_as_logits_and_is_not_the_final_mask():
    instance = backend()
    seed = np.zeros((8, 8), dtype=np.bool_)
    seed[:5, :5] = True
    result = instance.segment(Image.new("RGB", (8, 8)), [(1, 1, 1), (5, 5, 0)], seed)
    dense = instance.model.calls[0]["input_masks"]
    assert dense.shape == (1, 1, 8, 8)
    assert dense[0, 0, 0, 0] == 4
    assert dense[0, 0, 7, 7] == -4
    assert not np.array_equal(result.mask, seed)
    assert seed.sum() == 25


def test_spatial_cache_reuses_embedding_and_offload_clears_it():
    instance = backend()
    image = Image.new("RGB", (8, 8))
    instance.segment(image, [(1, 1, 1), (5, 5, 0)])
    instance.segment(image, [(1, 1, 1), (5, 5, 0), (2, 2, 1)])
    assert instance.processor.calls[1]["images"] is None
    assert "pixel_values" not in instance.model.calls[1]
    assert instance.model.calls[1]["image_embeddings"] is instance.model.embeddings
    instance.offload()
    assert instance.embeddings is None and instance.image is None
    assert not instance.active
    instance.segment(image, [(1, 1, 1), (5, 5, 0)])
    assert instance.processor.calls[-1]["images"] is image


@pytest.mark.parametrize(
    "points", [[], [(1, 1, 0)], [(-1, 1, 1)], [(1, float("nan"), 1)], [(1, 1, 2)]]
)
def test_invalid_spatial_prompts_fail_before_model_loading(points):
    instance = Sam3SpatialBackend("not-a-real-model", torch.device("cpu"))
    with pytest.raises(ValueError):
        instance.segment(Image.new("RGB", (8, 8)), points)
    assert instance.model is None


def test_mask_that_disobeys_clicks_is_not_returned():
    instance = backend()
    with pytest.raises(ValueError, match="satisfying your clicks"):
        instance.segment(Image.new("RGB", (8, 8)), [(7, 7, 1), (1, 1, 0)])


def test_switching_prompt_modes_offloads_the_other_model(monkeypatch):
    instance = object.__new__(Sam3ImageBackend)
    instance.model = Model()
    instance.device = torch.device("cpu")
    instance.model_name = "unused"
    instance._spatial_backend = backend()
    instance._spatial_active = False
    instance.refine_crop = False
    instance.single_tool = False
    monkeypatch.setattr(
        instance, "_predict", lambda images, prompts: [[] for _ in images]
    )
    image = Image.new("RGB", (8, 8))
    instance.spatial(image, [(1, 1, 1), (5, 5, 0)])
    assert instance._spatial_active
    assert instance.model.moves == ["cpu"]
    instance.predict([image], ["pliers"])
    assert not instance._spatial_active
    assert instance._spatial_backend.embeddings is None
    assert instance.model.moves == ["cpu", "cpu"]


def run_review(monkeypatch, commands, collect_points, spatial, draw=None):
    combined = np.ones((8, 8), dtype=np.bool_)
    replies = iter(commands)
    monkeypatch.setattr("builtins.input", lambda _: next(replies))
    logs = []
    rows = [
        {
            "Image ID": "image",
            "Instance ID": f"internal-{tool}",
            "Tool ID": tool,
            "Tool Class": cls,
            "Tool Name": name,
            "Confidence": "1",
        }
        for tool, cls, name in [("0", "scissors", "iris"), ("3", "pliers", "rongeur")]
    ]
    result = review_image_instances(
        Image.new("RGB", (8, 8)),
        [Candidate(combined, 0.9, "tools", "candidate-0")],
        rows,
        key="image",
        show=lambda *_: None,
        predict=lambda _: [],
        log=lambda *args, **kwargs: logs.append(kwargs),
        collect_points=collect_points,
        spatial=spatial,
        draw=draw,
    )
    return result, logs


def test_clicks_accumulate_and_result_is_used_for_subtraction(monkeypatch):
    point_calls = []
    seed = np.zeros((8, 8), dtype=np.bool_)
    seed[:3, :3] = True
    prompts = []

    def collect(image, mask, initial, title):
        point_calls.append(list(initial))
        assert "Tool 3" in title
        return [*initial, (1.0, 1.0, 1)] if not initial else [*initial, (5.0, 5.0, 0)]

    def spatial(points, polygon):
        prompts.append((list(points), polygon))
        return Candidate(seed.copy(), 0.8, "SAM3 point prompts", "spatial")

    result, logs = run_review(
        monkeypatch,
        ["click 3", "click 3", "subtract 0 0 3", "x 0", "a"],
        collect,
        spatial,
    )
    assert point_calls == [[], [(1.0, 1.0, 1)]]
    assert len(prompts[-1][0]) == 2
    np.testing.assert_array_equal(result.selected[0].mask, ~seed)
    np.testing.assert_array_equal(result.selected[1].mask, seed)
    assert result.selected[1].identity == "internal-3"
    assert [log["candidate_index"] for log in logs if log.get("action") == "click"] == [
        1,
        1,
    ]


def test_polygon_is_a_model_prompt_and_later_clicks_retain_it(monkeypatch):
    coarse = np.ones((8, 8), dtype=np.bool_)
    actual = coarse.copy()
    actual[3:, 3:] = False
    calls = []

    def spatial(points, seed):
        calls.append((list(points), seed.copy()))
        return Candidate(actual.copy(), 0.8, "SAM3 polygon prompt", "spatial")

    result, _ = run_review(
        monkeypatch,
        ["polygon 3", "click 3", "subtract 0 0 3", "x 0", "a"],
        lambda *args: [(1.0, 1.0, 1), (5.0, 5.0, 0)],
        spatial,
        draw=lambda *_: coarse.copy(),
    )
    assert calls[0][0] == []
    assert len(calls[1][0]) == 2
    np.testing.assert_array_equal(calls[1][1], coarse)
    np.testing.assert_array_equal(result.selected[1].mask, actual)
    assert not np.array_equal(result.selected[1].mask, coarse)


def test_cancelled_click_editor_does_not_call_model(monkeypatch):
    result, logs = run_review(
        monkeypatch,
        ["click 3", "s"],
        lambda *_: None,
        lambda *_: pytest.fail("Model should not run after cancellation"),
    )
    assert result.decision == "skipped"
    assert not any(log.get("action") == "click" for log in logs)


def test_click_reset_clears_both_polygon_and_point_prompts(monkeypatch):
    actual = np.zeros((8, 8), dtype=np.bool_)
    actual[:3, :3] = True
    seen_points, seen_seeds = [], []

    def collect(image, mask, points, title):
        seen_points.append(list(points))
        return [(1.0, 1.0, 1), (5.0, 5.0, 0)]

    def spatial(points, seed):
        seen_seeds.append(seed)
        return Candidate(actual.copy(), 0.8, "SAM3 spatial prompts", "spatial")

    result, _ = run_review(
        monkeypatch,
        ["polygon 3", "click 3", "click 3 reset", "subtract 0 0 3", "x 0", "a"],
        collect,
        spatial,
        draw=lambda *_: np.ones((8, 8), dtype=np.bool_),
    )
    assert seen_points == [[], []]
    assert seen_seeds[1] is not None and seen_seeds[2] is None
    np.testing.assert_array_equal(result.selected[1].mask, actual)


def test_failed_spatial_prediction_keeps_previous_assignment(monkeypatch):
    actual = np.zeros((8, 8), dtype=np.bool_)
    actual[:3, :3] = True
    attempts = []

    def spatial(points, seed):
        attempts.append(points)
        if len(attempts) == 2:
            raise ValueError("No mask satisfies your clicks")
        return Candidate(actual.copy(), 0.8, "SAM3 point prompts", "spatial")

    result, _ = run_review(
        monkeypatch,
        ["click 3", "click 3", "subtract 0 0 3", "x 0", "a"],
        lambda *args: [(1.0, 1.0, 1), (5.0, 5.0, 0)],
        spatial,
    )
    assert len(attempts) == 2
    np.testing.assert_array_equal(result.selected[1].mask, actual)


def test_removing_spatial_candidate_clears_click_history(monkeypatch):
    actual = np.zeros((8, 8), dtype=np.bool_)
    actual[:3, :3] = True
    seen = []

    def collect(image, mask, points, title):
        seen.append(list(points))
        return [(1.0, 1.0, 1), (5.0, 5.0, 0)]

    result, _ = run_review(
        monkeypatch,
        ["click 3", "remove 1", "click 3", "subtract 0 0 3", "remove 0", "a"],
        collect,
        lambda *_: Candidate(actual.copy(), 0.8, "SAM3 point prompts", "spatial"),
    )
    assert seen == [[], []]
    np.testing.assert_array_equal(result.selected[1].mask, actual)


def test_separate_click_prompts_keep_their_points_and_merge_visible_parts(monkeypatch):
    first = np.zeros((8, 8), dtype=np.bool_)
    first[:2, :2] = True
    second = np.zeros_like(first)
    second[6:, 6:] = True
    revised = first.copy()
    revised[2, :2] = True
    masks = iter([first, second, revised, revised])
    initial_points = []

    def collect(image, mask, initial, title):
        initial_points.append(list(initial))
        return [*initial, (float(len(initial) + 1), 1.0, 1)]

    result, logs = run_review(
        monkeypatch,
        [
            "click 3",
            "click 3 new",
            "click 3 1",
            "click 3",
            "subtract 0 0 3",
            "x 0",
            "a",
        ],
        collect,
        lambda *_: Candidate(next(masks).copy(), 0.8, "points", "spatial"),
    )
    assert initial_points == [[], [], [(1.0, 1.0, 1)], [(1.0, 1.0, 1), (2.0, 1.0, 1)]]
    assert [log["candidate_index"] for log in logs if log.get("action") == "click"] == [
        1,
        2,
        1,
        1,
    ]
    np.testing.assert_array_equal(result.selected[1].mask, revised | second)
    np.testing.assert_array_equal(result.selected[0].mask, ~(revised | second))


@pytest.mark.parametrize("option", ["new", "reset"])
def test_cancelled_new_or_reset_prompt_preserves_active_points(monkeypatch, option):
    mask = np.zeros((8, 8), dtype=np.bool_)
    mask[:2, :2] = True
    seen = []

    def collect(image, context, initial, title):
        seen.append(list(initial))
        if len(seen) == 2:
            return None
        return [*initial, (1.0, 1.0, 1)]

    result, logs = run_review(
        monkeypatch,
        ["click 3", f"click 3 {option}", "click 3", "subtract 0 0 3", "x 0", "a"],
        collect,
        lambda *_: Candidate(mask.copy(), 0.8, "points", "spatial"),
    )
    assert seen == [[], [], [(1.0, 1.0, 1)]]
    assert [log["candidate_index"] for log in logs if log.get("action") == "click"] == [
        1,
        1,
    ]
    np.testing.assert_array_equal(result.selected[1].mask, mask)


def test_failed_new_prompt_is_transactional_and_can_be_explicitly_retried(monkeypatch):
    first = np.zeros((8, 8), dtype=np.bool_)
    first[:2, :2] = True
    second = np.zeros_like(first)
    second[6:, 6:] = True
    seen = []
    attempts = []

    def collect(image, context, initial, title):
        seen.append(list(initial))
        return [*initial, (1.0, 1.0, 1)]

    def spatial(points, seed):
        attempts.append(points)
        if len(attempts) == 2:
            raise ValueError("No mask satisfies your clicks")
        return Candidate(
            first.copy() if len(attempts) == 1 else second.copy(),
            0.8,
            "points",
            "spatial",
        )

    result, logs = run_review(
        monkeypatch,
        ["click 3", "click 3 new", "click 3 new", "subtract 0 0 3", "x 0", "a"],
        collect,
        spatial,
    )
    assert seen == [[], [], []]
    assert [log["candidate_index"] for log in logs if log.get("action") == "click"] == [
        1,
        2,
    ]
    np.testing.assert_array_equal(result.selected[1].mask, first | second)


def test_reset_edits_only_active_portion_and_keeps_other_parts(monkeypatch):
    first = np.zeros((8, 8), dtype=np.bool_)
    first[:2, :2] = True
    second = np.zeros_like(first)
    second[6:, 6:] = True
    revised = second.copy()
    revised[5, 6:] = True
    masks = iter([first, second, revised])
    seen = []

    def collect(image, context, initial, title):
        seen.append(list(initial))
        return [(1.0, 1.0, 1)]

    result, logs = run_review(
        monkeypatch,
        ["click 3", "click 3 new", "click 3 reset", "subtract 0 0 3", "x 0", "a"],
        collect,
        lambda *_: Candidate(next(masks).copy(), 0.8, "points", "spatial"),
    )
    assert seen == [[], [], []]
    assert [log["candidate_index"] for log in logs if log.get("action") == "click"] == [
        1,
        2,
        2,
    ]
    np.testing.assert_array_equal(result.selected[1].mask, first | revised)


def test_prompt_selection_validates_candidate_and_tool_ownership(monkeypatch, capsys):
    mask = np.zeros((8, 8), dtype=np.bool_)
    mask[:2, :2] = True
    seen = []

    def collect(image, context, initial, title):
        seen.append(list(initial))
        return [*initial, (1.0, 1.0, 1)]

    result, _ = run_review(
        monkeypatch,
        [
            "click 3 0",
            "click 3",
            "click 0 1",
            "u 3",
            "g 0 1",
            "click 3 1",
            "u 0",
            "g 3 1",
            "click 3 1",
            "remove 1",
            "click 3 1",
            "s",
        ],
        collect,
        lambda *_: Candidate(mask.copy(), 0.8, "points", "spatial"),
    )
    assert seen == [[], [(1.0, 1.0, 1)]]
    assert result.decision == "skipped"
    output = capsys.readouterr().out
    assert "has no click prompt for tool" in output
    assert "assigned to another tool" in output
    assert "removed" in output.lower()


def test_editing_click_prompt_discards_stale_postprocessing(monkeypatch):
    mask = np.zeros((8, 8), dtype=np.bool_)
    mask[1:7, 1:7] = True
    mask[3, 3] = False
    result, logs = run_review(
        monkeypatch,
        ["click 3", "clean 3 holes 1", "click 3", "subtract 0 0 3", "x 0", "a"],
        lambda *_: [(1.0, 1.0, 1)],
        lambda *_: Candidate(mask.copy(), 0.8, "points", "spatial"),
    )
    assert any(
        log.get("action") == "cleanup" and log["added_pixels"] == 1 for log in logs
    )
    np.testing.assert_array_equal(result.selected[1].mask, mask)


def test_consecutive_mask_corrections_reuse_selected_decoder_logits():
    instance = backend()
    image = Image.new("RGB", (8, 8))
    first = instance.segment(image, [(1, 1, 1), (5, 5, 0)])
    instance.segment(image, [(5, 5, 0)], first.mask)
    logits = instance.model.calls[-1]["input_masks"]
    assert logits.shape == (1, 1, 8, 8)
    assert logits[0, 0, 1, 1] == 10
    assert logits[0, 0, 5, 5] == -10
    assert instance.processor.calls[-1]["images"] is None
    instance.offload()
    assert instance.previous_mask is None and instance.previous_logits is None


def test_box_prompt_is_forwarded_without_requiring_positive_clicks():
    instance = backend()
    instance.segment(Image.new("RGB", (8, 8)), [], box=(1.0, 1.0, 7.0, 7.0))
    assert instance.processor.calls[-1]["input_boxes"] == [[[1.0, 1.0, 7.0, 7.0]]]


@pytest.mark.parametrize("box", [(-1, 0, 4, 4), (3, 3, 2, 5), (0, 0, 9, 9), (0, 0, float("nan"), 5)])
def test_invalid_boxes_fail_before_loading_model(box):
    instance = Sam3SpatialBackend("unused", torch.device("cpu"))
    with pytest.raises(ValueError, match="Box"):
        instance.segment(Image.new("RGB", (8, 8)), [], box=box)
    assert instance.model is None


def test_failed_prompt_on_new_image_cannot_reuse_previous_image_logits():
    instance = backend()
    instance.segment(Image.new("RGB", (8, 8)), [(1, 1, 1), (5, 5, 0)])
    assert instance.previous_logits is not None
    with pytest.raises(ValueError, match="satisfying your clicks"):
        instance.segment(Image.new("RGB", (8, 8)), [(7, 7, 1), (1, 1, 0)])
    assert instance.previous_mask is None and instance.previous_logits is None
