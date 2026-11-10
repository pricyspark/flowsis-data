import cv2
import numpy as np
import pytest

from square_crop import MarkerDetectionError, find_marker_corners


@pytest.mark.parametrize("size", [1440, 3024, 4284])
def test_marker_detection_tracks_the_same_corners_across_resolutions(size):
    # A 100px L on a 4284px photo, with proportionally smaller video markers.
    reference = np.full((4284, 4284, 3), 180, dtype=np.uint8)
    marker = np.zeros((100, 100), dtype=np.bool_)
    marker[:29, :] = True
    marker[:, :29] = True
    for y, x, shape in (
        (200, 200, marker),
        (200, 3984, np.fliplr(marker)),
        (3984, 200, np.flipud(marker)),
        (3984, 3984, np.flip(marker)),
    ):
        reference[y : y + 100, x : x + 100][shape] = (255, 85, 0)
    pixels = cv2.resize(reference, (size, size), interpolation=cv2.INTER_AREA)
    corners = find_marker_corners(pixels)
    expected = np.array([[229, 229], [4054, 229], [229, 4054], [4054, 4054]])
    np.testing.assert_allclose(corners / size, expected / 4284, atol=0.003)

    # Scale correction must still reject a genuinely missing marker.
    pixels[: size // 2, : size // 2] = 180
    with pytest.raises(MarkerDetectionError, match="no blue marker.*tl"):
        find_marker_corners(pixels)
