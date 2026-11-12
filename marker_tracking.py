"""Local video marker searches based only on previously validated observations."""

import math

import numpy as np

from square_crop import MarkerDetectionError, MarkerMatch, find_markers


class MarkerTracker:
    def __init__(self) -> None:
        self.matches: list[MarkerMatch] | None = None
        self.timestamp: float | None = None
        self.shape: tuple[int, int] | None = None
        self.velocity = np.zeros((4, 2), dtype=np.float64)
        self.search_bounds: np.ndarray | None = None

    def find_corners(self, image: np.ndarray, timestamp: float) -> np.ndarray:
        if not math.isfinite(timestamp):
            raise ValueError("marker timestamp must be finite")
        shape = image.shape[:2]
        if self.shape is not None and shape != self.shape:
            raise MarkerDetectionError("video frame dimensions changed")
        if self.timestamp is not None and timestamp <= self.timestamp:
            raise ValueError("marker timestamps must increase")

        expected = None
        elapsed = 0.0
        if self.matches is not None:
            assert self.timestamp is not None
            elapsed = timestamp - self.timestamp
            points = np.array([match.point for match in self.matches])
            bounds = np.array([match.bounds for match in self.matches])
            sizes = np.max(bounds[:, 2:] - bounds[:, :2], axis=1)
            step = self.velocity * elapsed
            # Limit extrapolation to an observed marker width. A large jump
            # should fail validation instead of searching unrelated scene areas.
            distance = np.linalg.norm(step, axis=1)
            step *= np.minimum(1.0, sizes / np.maximum(distance, 1e-9))[:, None]
            expected = points + step
            margin = 0.75 * sizes + np.minimum(2 * distance, sizes)
            self.search_bounds = bounds + np.tile(step, (1, 2))
            self.search_bounds += margin[:, None] * np.array([-1, -1, 1, 1])

        matches = find_markers(
            image,
            search_bounds=self.search_bounds,
            expected_corners=expected,
        )
        corners = np.array([match.point for match in matches])
        # State changes only after all four markers and their geometry validate.
        if self.matches is not None:
            previous = np.array([match.point for match in self.matches])
            self.velocity = 0.5 * self.velocity + 0.5 * (corners - previous) / elapsed
        self.matches = matches
        self.timestamp = timestamp
        self.shape = shape
        return corners
