from dataclasses import dataclass
from itertools import product

import cv2
import numpy as np

BLUE_THRESHOLD = 0.25
DETECTION_MAX_DIMENSION = 1000
CROP_SIZE = 2160
MAX_TEMPLATE_ERROR = 0.35
# The existing 100-pixel kernel was calibrated on 4284x4284 raw photos.
MARKER_KERNEL_RATIO = 100 / 4284
CORNERS = ("tl", "tr", "bl", "br")
WEAK_BLUE_RATIO = 0.6


@dataclass(frozen=True)
class MarkerMatch:
    """One observed marker, in source-image coordinates; bounds are L, T, R, B."""

    point: np.ndarray
    bounds: np.ndarray
    error: float


class MarkerDetectionError(ValueError):
    """Raised when all four crop markers cannot be detected reliably."""


def blue_score(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    hue = hsv[..., 0]
    saturation = hsv[..., 1] / 255.0
    value = hsv[..., 2] / 255.0

    target_hue = 110.0
    hue_sigma = 10.0
    hue_distance = np.abs(hue - target_hue)
    hue_distance = np.minimum(hue_distance, 180.0 - hue_distance)

    hue_score = np.exp(-0.5 * (hue_distance / hue_sigma) ** 2)
    value_score = np.clip((value - 0.1) / 0.2, 0.0, 1.0)
    return hue_score * saturation * value_score


def find_tl_inner_corner(
    mask: np.ndarray,
    kernel_size: int,
    max_error: float = MAX_TEMPLATE_ERROR,
) -> np.ndarray:
    """Match a top-left L and return its concave corner as ``[x, y]``."""
    point, _ = match_tl_inner_corner(mask, kernel_size, max_error)
    return point


def match_tl_inner_corner(
    mask: np.ndarray,
    kernel_size: int,
    max_error: float = MAX_TEMPLATE_ERROR,
) -> tuple[np.ndarray, float]:
    if kernel_size < 3:
        raise ValueError("kernel size must be at least 3")
    if kernel_size > min(mask.shape):
        raise MarkerDetectionError("marker search area is smaller than the L kernel")

    arm_width = max(2, round(kernel_size * 2 / 7))
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    kernel[:arm_width, :] = 1.0
    kernel[:, :arm_width] = 1.0

    errors = cv2.matchTemplate(
        np.asarray(mask, dtype=np.float32),
        kernel,
        cv2.TM_SQDIFF_NORMED,
    )
    error, _, location, _ = cv2.minMaxLoc(errors)
    if not np.isfinite(error) or error > max_error:
        raise MarkerDetectionError(
            f"marker does not match the L kernel (error {error:.3f})"
        )

    left, top = location
    return np.array([left + arm_width, top + arm_width]), error


def find_inner_corner(mask: np.ndarray, kernel_size: int, corner: str) -> np.ndarray:
    point, _ = match_inner_corner(mask, kernel_size, corner)
    return point


def match_inner_corner(
    mask: np.ndarray, kernel_size: int, corner: str
) -> tuple[np.ndarray, float]:
    height, width = mask.shape
    if corner == "tl":
        adjusted_mask = mask
    elif corner == "tr":
        adjusted_mask = np.fliplr(mask)
    elif corner == "bl":
        adjusted_mask = np.flipud(mask)
    elif corner == "br":
        adjusted_mask = np.flip(mask, axis=(0, 1))
    else:
        raise ValueError(f"unknown corner: {corner!r}")

    point, error = match_tl_inner_corner(adjusted_mask, kernel_size)
    if corner in {"tr", "br"}:
        point[0] = width - 1 - point[0]
    if corner in {"bl", "br"}:
        point[1] = height - 1 - point[1]
    return point, error


def marker_candidates(
    score: np.ndarray,
    corner: str,
    kernel_size: int,
    threshold: float,
    minimum_area: int,
) -> list[MarkerMatch]:
    """Grow seeded blue components, then match each component independently."""
    minimum_seeds = max(3, round(kernel_size**2 * 0.02))
    matches = []
    reason = "incomplete blue marker"
    has_seeds = False
    # Keep strict candidates too: weak pixels can connect dark tape to nearby
    # blue noise. Growth should recover pale tape, not invalidate a clean L.
    for cutoff in (threshold * WEAK_BLUE_RATIO, threshold):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            (score > cutoff).astype(np.uint8), connectivity=8
        )
        seed_counts = np.bincount(labels[score > threshold], minlength=count)
        seeded = set(np.flatnonzero(seed_counts >= minimum_seeds)) - {0}
        has_seeds |= bool(seeded)
        for label in range(1, count):
            left, top, width, height, area = stats[label]
            if label not in seeded or area < minimum_area:
                continue
            if min(width, height) < kernel_size:
                continue
            # Padding is measured in kernel units, not fixed image locations.
            x0, y0 = max(0, left - kernel_size), max(0, top - kernel_size)
            x1 = min(score.shape[1], left + width + kernel_size)
            y1 = min(score.shape[0], top + height + kernel_size)
            component = labels[y0:y1, x0:x1] == label
            try:
                point, error = match_inner_corner(component, kernel_size, corner)
            except MarkerDetectionError as failure:
                reason = str(failure)
                continue
            matches.append(
                MarkerMatch(
                    point.astype(np.float64) + (x0, y0),
                    np.array([left, top, left + width, top + height], dtype=np.float64),
                    error,
                )
            )
    if not has_seeds:
        raise MarkerDetectionError(f"no blue marker found in the {corner} quadrant")
    if not matches:
        raise MarkerDetectionError(f"{corner} quadrant: {reason}")
    unique = []
    for match in sorted(matches, key=lambda candidate: candidate.error):
        if all(
            np.linalg.norm(match.point - other.point) > kernel_size / 2
            for other in unique
        ):
            unique.append(match)
    return unique


def find_marker_corners(
    image: np.ndarray,
    threshold: float = BLUE_THRESHOLD,
    max_dimension: int = DETECTION_MAX_DIMENSION,
) -> np.ndarray:
    """Return marker inner corners in TL, TR, BL, BR order."""
    return np.array(
        [match.point for match in find_markers(image, threshold, max_dimension)]
    )


def find_markers(
    image: np.ndarray,
    threshold: float = BLUE_THRESHOLD,
    max_dimension: int = DETECTION_MAX_DIMENSION,
    *,
    search_bounds: np.ndarray | None = None,
    expected_corners: np.ndarray | None = None,
) -> list[MarkerMatch]:
    """Find four observed markers; optional search bounds use source coordinates."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a three-channel BGR array")
    if max_dimension < 100:
        raise ValueError("max_dimension must be at least 100")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between 0 and 1")
    if search_bounds is not None and (
        search_bounds.shape != (4, 4) or not np.isfinite(search_bounds).all()
    ):
        raise ValueError("search_bounds must be a finite (4, 4) array")
    if expected_corners is not None and (
        expected_corners.shape != (4, 2) or not np.isfinite(expected_corners).all()
    ):
        raise ValueError("expected_corners must be a finite (4, 2) array")

    height, width = image.shape[:2]
    scale = min(1.0, max_dimension / max(height, width))
    if scale < 1.0:
        small = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        small = image

    score = blue_score(small)
    small_height, small_width = score.shape
    kernel_size = max(7, round(max(small_height, small_width) * MARKER_KERNEL_RATIO))
    middle_y = small_height // 2
    middle_x = small_width // 2
    quadrants = {
        "tl": (0, middle_y, 0, middle_x),
        "tr": (0, middle_y, middle_x, small_width),
        "bl": (middle_y, small_height, 0, middle_x),
        "br": (middle_y, small_height, middle_x, small_width),
    }

    minimum_area = max(20, round(score.size * 0.000125))
    candidates = []
    for index, corner in enumerate(CORNERS):
        top, bottom, left, right = quadrants[corner]
        if search_bounds is not None:
            x0, y0, x1, y1 = search_bounds[index] * scale
            left, top = max(left, int(np.floor(x0))), max(top, int(np.floor(y0)))
            right = min(right, int(np.ceil(x1)))
            bottom = min(bottom, int(np.ceil(y1)))
        if left >= right or top >= bottom:
            raise MarkerDetectionError(f"empty search region for the {corner} quadrant")
        local = marker_candidates(
            score[top:bottom, left:right], corner, kernel_size, threshold, minimum_area
        )
        matches = [
            MarkerMatch(
                (match.point + (left, top)) / scale,
                (match.bounds + (left, top, left, top)) / scale,
                match.error,
            )
            for match in local
        ]
        if expected_corners is not None:
            matches.sort(
                key=lambda match: match.error
                + 0.05 * np.linalg.norm(match.point - expected_corners[index])
                / max(kernel_size / scale, np.max(match.bounds[2:] - match.bounds[:2]))
            )
        else:
            matches.sort(key=lambda match: match.error)
        candidates.append(matches[:3])

    ranked = []
    for combination in product(*candidates):
        points = np.array([match.point for match in combination])
        outline = points[[0, 1, 3, 2]].astype(np.float32)
        if (
            not cv2.isContourConvex(outline)
            or abs(cv2.contourArea(outline)) < height * width * 0.25
        ):
            continue
        sizes = np.array(
            [np.max(match.bounds[2:] - match.bounds[:2]) for match in combination]
        )
        cost = sum(match.error for match in combination) + 0.2 * np.std(np.log(sizes))
        if expected_corners is not None:
            cost += 0.05 * np.sum(
                np.linalg.norm(points - expected_corners, axis=1) / sizes
            )
        ranked.append((float(cost), combination))
    if not ranked:
        raise MarkerDetectionError("detected markers do not form a valid crop area")
    ranked.sort(key=lambda item: item[0])
    if len(ranked) > 1 and ranked[1][0] - ranked[0][0] < 0.02:
        first = np.array([match.point for match in ranked[0][1]])
        second = np.array([match.point for match in ranked[1][1]])
        if np.max(np.linalg.norm(first - second, axis=1)) > 2 * kernel_size / scale:
            raise MarkerDetectionError("ambiguous blue marker candidates")
    return list(ranked[0][1])


def warp_quad_to_square(
    image: np.ndarray,
    corners: np.ndarray,
    size: int = CROP_SIZE,
    inset_fraction: float = 0.02,
) -> np.ndarray:
    if size < 1:
        raise ValueError("crop size must be at least 1")
    source = np.asarray(corners, dtype=np.float32)
    if source.shape != (4, 2):
        raise ValueError("corners must have shape (4, 2)")

    margin = round(size * inset_fraction)

    destination = np.array(
        [
            [-margin, -margin],
            [size - 1 + margin, -margin],
            [-margin, size - 1 + margin],
            [size - 1 + margin, size - 1 + margin],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(image, transform, (size, size))


def crop_marked_square(image: np.ndarray, size: int = CROP_SIZE) -> np.ndarray:
    corners = find_marker_corners(image)
    return warp_quad_to_square(image, corners, size)
