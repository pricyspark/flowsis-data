import cv2
import numpy as np
import pytest

from marker_tracking import MarkerTracker
from square_crop import MarkerDetectionError, find_marker_corners, find_markers


def frame(*, shift=0, margin=60, pale=False):
    image = np.full((1000, 1000, 3), 120, dtype=np.uint8)
    shape = np.zeros((48, 48), dtype=np.bool_)
    shape[:14, :] = True
    shape[:, :14] = True
    weak = cv2.cvtColor(np.uint8([[[110, 55, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]
    marker = np.full((48, 48, 3), 120, dtype=np.uint8)
    marker[shape] = weak if pale else (255, 85, 0)
    marker[:14, :14] = (255, 85, 0)
    far = 1000 - margin - 48
    for y, x, pixels in (
        (margin, margin + shift, marker),
        (margin, far + shift, np.fliplr(marker)),
        (far, margin + shift, np.flipud(marker)),
        (far, far + shift, np.flip(marker, axis=(0, 1))),
    ):
        image[y : y + 48, x : x + 48] = pixels
    return image


def distractor(image):
    # A second, perfectly blue L in the top-left quadrant, away from the tape.
    image[150:164, 350:398] = (255, 85, 0)
    image[150:198, 350:364] = (255, 85, 0)
    return image


def test_pale_connected_arms_retain_the_observed_inner_corners():
    expected = find_marker_corners(frame())
    np.testing.assert_allclose(find_marker_corners(frame(pale=True)), expected, atol=1)


def test_strict_candidate_survives_weak_pixels_filling_the_concavity():
    image = frame()
    weak = cv2.cvtColor(np.uint8([[[110, 55, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]
    image[74:108, 74:108] = weak
    np.testing.assert_allclose(
        find_marker_corners(image), find_marker_corners(frame()), atol=1
    )


def test_one_strong_pixel_cannot_seed_a_large_weak_candidate():
    image = frame(pale=True)
    weak = cv2.cvtColor(np.uint8([[[110, 55, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]
    image[60:74, 60:74] = weak
    image[60, 60] = (255, 85, 0)
    with pytest.raises(MarkerDetectionError, match="no blue marker.*tl"):
        find_markers(image)


def test_candidates_keep_separate_bounds_in_a_quadrant():
    image = frame()
    image[5:100, 300:450] = (255, 85, 0)
    matches = find_markers(image)
    np.testing.assert_array_equal(matches[0].bounds, [60, 60, 108, 108])
    np.testing.assert_allclose(matches[0].point, [74, 74], atol=1)


def test_ambiguous_initialization_is_rejected():
    with pytest.raises(MarkerDetectionError, match="ambiguous"):
        find_markers(distractor(frame()))


def test_local_search_ignores_a_second_marker_shaped_object():
    tracker = MarkerTracker()
    expected = tracker.find_corners(frame(), 0.0)
    actual = tracker.find_corners(distractor(frame(pale=True)), 0.1)
    np.testing.assert_allclose(actual, expected, atol=1)
    assert tracker.search_bounds is not None
    assert tracker.search_bounds[0, 2] < 350


@pytest.mark.parametrize("corner", range(4))
def test_missing_marker_cannot_be_replaced_and_failed_state_is_not_committed(corner):
    tracker = MarkerTracker()
    expected = tracker.find_corners(frame(), 0.0)
    before = tracker.matches
    hidden = distractor(frame())
    x, y = np.rint(expected[corner]).astype(int)
    cv2.rectangle(hidden, (x - 60, y - 60), (x + 60, y + 60), (120, 120, 120), -1)
    with pytest.raises(MarkerDetectionError):
        tracker.find_corners(hidden, 0.1)
    assert tracker.matches is before
    assert tracker.timestamp == 0.0
    np.testing.assert_array_equal(tracker.find_corners(frame(), 0.2), expected)


@pytest.mark.parametrize("missing_arm", ["horizontal", "vertical"])
def test_visible_strips_without_the_inner_l_are_rejected(missing_arm):
    image = frame()
    if missing_arm == "horizontal":
        image[60:74, 74:108] = 120
    else:
        image[74:108, 60:74] = 120
    with pytest.raises(MarkerDetectionError):
        find_markers(image)


def test_neighborhood_moves_beyond_its_initial_location():
    tracker = MarkerTracker()
    initial = tracker.find_corners(frame(margin=200), 0.0)
    for index in range(1, 21):
        actual = tracker.find_corners(frame(margin=200, shift=5 * index), index * 0.04)
    np.testing.assert_allclose(actual - initial, np.tile([100, 0], (4, 1)), atol=1)


def test_variable_timestamps_and_discontinuous_motion():
    tracker = MarkerTracker()
    tracker.find_corners(frame(margin=200), 0.0)
    tracker.find_corners(frame(margin=200, shift=5), 0.03)
    tracker.find_corners(frame(margin=200, shift=10), 0.2)
    with pytest.raises(ValueError, match="increase"):
        tracker.find_corners(frame(margin=200, shift=10), 0.2)
    with pytest.raises(MarkerDetectionError):
        tracker.find_corners(frame(margin=200, shift=150), 0.23)


def test_frame_size_changes_are_rejected():
    tracker = MarkerTracker()
    tracker.find_corners(frame(), 0.0)
    with pytest.raises(MarkerDetectionError, match="dimensions changed"):
        tracker.find_corners(cv2.resize(frame(), (800, 800)), 0.1)
