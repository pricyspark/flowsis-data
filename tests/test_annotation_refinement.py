import numpy as np
import pytest
from PIL import Image

from annotate import Candidate, Sam3ImageBackend


def candidate(shape, box, identity="instance-0"):
    mask = np.zeros(shape, dtype=np.bool_)
    left, top, right, bottom = box
    mask[top:bottom, left:right] = True
    return Candidate(mask, 0.9, "scissors", identity)


def backend_with_predictions(
    monkeypatch, responses, *, enabled=True, single_tool=False
):
    backend = object.__new__(Sam3ImageBackend)
    backend.refine_crop = enabled
    backend.single_tool = single_tool
    calls = []
    responses = iter(responses)

    def predict(images, prompts):
        calls.append((images, prompts))
        return next(responses)

    monkeypatch.setattr(backend, "_predict", predict)
    return backend, calls


def test_refinement_crops_original_pixels_and_restores_coordinates(monkeypatch):
    pixels = np.arange(100 * 120 * 3, dtype=np.uint8).reshape(100, 120, 3)
    image = Image.fromarray(pixels)
    initial = candidate((100, 120), (70, 30, 90, 40), "instance-7")
    refined = candidate((18, 28), (3, 3, 25, 15))
    backend, calls = backend_with_predictions(
        monkeypatch, [[[initial]], [[refined]]]
    )

    result = backend.predict([image], ["scissors"])[0][0]

    assert len(calls) == 2
    np.testing.assert_array_equal(np.asarray(calls[1][0][0]), pixels[26:44, 66:94])
    assert calls[1][1] == ["scissors"]
    expected = candidate((100, 120), (69, 29, 91, 41)).mask
    np.testing.assert_array_equal(result.mask, expected)
    assert result.identity == "instance-7"


@pytest.mark.parametrize("failure", ["empty", "ambiguous", "clipped"])
def test_failed_refinement_keeps_original(monkeypatch, failure):
    initial = candidate((100, 120), (70, 30, 90, 40))
    refined = {
        "empty": [],
        "ambiguous": [
            candidate((18, 28), (3, 3, 10, 10)),
            candidate((18, 28), (15, 3, 25, 10)),
        ],
        "clipped": [candidate((18, 28), (0, 3, 25, 15))],
    }[failure]
    backend, _ = backend_with_predictions(monkeypatch, [[[initial]], [refined]])
    assert backend.predict([Image.new("RGB", (120, 100))], ["scissors"])[0] == [
        initial
    ]


@pytest.mark.parametrize("case", ["disabled", "empty", "ambiguous", "full_frame"])
def test_unnecessary_refinement_skips_second_pass(monkeypatch, case):
    initial = {
        "disabled": [candidate((100, 120), (70, 30, 90, 40))],
        "empty": [],
        "ambiguous": [
            candidate((100, 120), (10, 10, 20, 20)),
            candidate((100, 120), (70, 30, 90, 40)),
        ],
        "full_frame": [candidate((100, 120), (0, 0, 120, 100))],
    }[case]
    backend, calls = backend_with_predictions(
        monkeypatch, [[initial]], enabled=case != "disabled"
    )
    assert backend.predict([Image.new("RGB", (120, 100))], ["scissors"])[0] == initial
    assert len(calls) == 1


def test_batch_skips_empty_image_and_allows_mask_at_real_image_boundary(monkeypatch):
    initial = candidate((100, 120), (0, 30, 20, 40))
    refined = candidate((18, 24), (0, 3, 21, 15))
    backend, calls = backend_with_predictions(
        monkeypatch, [[[], [initial]], [[refined]]]
    )
    images = [Image.new("RGB", (120, 100)) for _ in range(2)]
    result = backend.predict(images, ["forceps", "scissors"])
    assert result[0] == []
    assert calls[1][1] == ["scissors"]
    np.testing.assert_array_equal(
        result[1][0].mask, candidate((100, 120), (0, 29, 21, 41)).mask
    )


def test_single_tool_merges_fragments_in_both_passes(monkeypatch):
    initial = [
        candidate((100, 120), (70, 30, 80, 40), "instance-7"),
        candidate((100, 120), (80, 30, 90, 40), "instance-8"),
    ]
    refined = [
        candidate((18, 28), (3, 3, 12, 15)),
        candidate((18, 28), (16, 3, 25, 15)),
    ]
    backend, calls = backend_with_predictions(
        monkeypatch, [[initial], [refined]], single_tool=True
    )
    result = backend.predict([Image.new("RGB", (120, 100))], ["scissors"])[0]
    assert calls[1][0][0].size == (28, 18)
    assert len(result) == 1
    expected = candidate((100, 120), (69, 29, 78, 41)).mask
    expected |= candidate((100, 120), (82, 29, 91, 41)).mask
    np.testing.assert_array_equal(result[0].mask, expected)
    assert result[0].identity == "instance-7"


def test_single_tool_merges_without_refinement_and_keeps_images_separate(monkeypatch):
    fragments = [
        candidate((100, 120), (10, 10, 20, 20)),
        candidate((100, 120), (70, 30, 90, 40)),
    ]
    backend, calls = backend_with_predictions(
        monkeypatch, [[fragments, []]], enabled=False, single_tool=True
    )
    result = backend.predict(
        [Image.new("RGB", (120, 100)) for _ in range(2)], ["scissors", "forceps"]
    )
    assert len(calls) == 1
    assert len(result[0]) == 1
    assert result[1] == []
    np.testing.assert_array_equal(
        result[0][0].mask, fragments[0].mask | fragments[1].mask
    )
