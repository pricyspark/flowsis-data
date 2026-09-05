import cv2
import numpy as np

BLUE_THRESHOLD = 0.25
DETECTION_MAX_DIMENSION = 1000
CROP_SIZE = 2160
MAX_TEMPLATE_ERROR = 0.35
CORNERS = ("tl", "tr", "bl", "br")


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
    return np.array([left + arm_width, top + arm_width])


def find_inner_corner(mask: np.ndarray, kernel_size: int, corner: str) -> np.ndarray:
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

    point = find_tl_inner_corner(adjusted_mask, kernel_size)
    if corner in {"tr", "br"}:
        point[0] = width - 1 - point[0]
    if corner in {"bl", "br"}:
        point[1] = height - 1 - point[1]
    return point


def marker_components(mask: np.ndarray, corner: str) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8),
        connectivity=8,
    )
    minimum_area = max(20, round(mask.size * 0.0005))
    candidates = [
        label
        for label in range(1, count)
        if stats[label, cv2.CC_STAT_AREA] >= minimum_area
        and stats[label, cv2.CC_STAT_WIDTH] >= 4
        and stats[label, cv2.CC_STAT_HEIGHT] >= 4
    ]
    if not candidates:
        raise MarkerDetectionError(f"no blue marker found in the {corner} quadrant")
    return np.isin(labels, candidates)


def marker_search_area(
    mask: np.ndarray,
    corner: str,
    kernel_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop a marker mask to its blue pixels plus room for the L kernel."""
    points = cv2.findNonZero(mask.astype(np.uint8))
    left, top, width, height = cv2.boundingRect(points)
    if width < kernel_size or height < kernel_size:
        raise MarkerDetectionError(f"incomplete blue marker in the {corner} quadrant")

    bottom = min(mask.shape[0], top + height + kernel_size)
    right = min(mask.shape[1], left + width + kernel_size)
    top = max(0, top - kernel_size)
    left = max(0, left - kernel_size)
    return mask[top:bottom, left:right], np.array([left, top])


def find_marker_corners(
    image: np.ndarray,
    threshold: float = BLUE_THRESHOLD,
    max_dimension: int = DETECTION_MAX_DIMENSION,
) -> np.ndarray:
    """Return marker inner corners in TL, TR, BL, BR order."""
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a three-channel BGR array")
    if max_dimension < 100:
        raise ValueError("max_dimension must be at least 100")

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

    mask = (blue_score(small) > threshold).astype(np.uint8)

    small_height, small_width = mask.shape
    middle_y = small_height // 2
    middle_x = small_width // 2
    quadrants = {
        "tl": (0, middle_y, 0, middle_x),
        "tr": (0, middle_y, middle_x, small_width),
        "bl": (middle_y, small_height, 0, middle_x),
        "br": (middle_y, small_height, middle_x, small_width),
    }

    points = []
    for corner in CORNERS:
        top, bottom, left, right = quadrants[corner]
        quadrant_mask = mask[top:bottom, left:right]
        quadrant_mask = marker_components(quadrant_mask, corner)
        kernel_size = max(7, round(100 * scale))
        search_mask, search_offset = marker_search_area(
            quadrant_mask,
            corner,
            kernel_size,
        )
        point = find_inner_corner(search_mask, kernel_size, corner).astype(np.float64)
        point += search_offset + (left, top)
        points.append(point)

    small_points = np.array(points)
    outline = small_points[[0, 1, 3, 2]].astype(np.float32)
    area = abs(cv2.contourArea(outline))
    if not cv2.isContourConvex(outline) or area < small_height * small_width * 0.25:
        raise MarkerDetectionError("detected markers do not form a valid crop area")

    return small_points / scale


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
