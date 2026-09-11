from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

MASK_BUNDLE_VERSION = 2
MASK_BITORDER = "big"


@dataclass(frozen=True)
class MaskBundle:
    """All visible instance masks and the optional ignore region for one frame."""

    masks: NDArray[np.bool_]
    identities: tuple[str, ...]
    ignore: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        if self.masks.ndim != 3:
            raise ValueError("Bundle masks must have shape [N, H, W].")
        if len(self.identities) != self.masks.shape[0]:
            raise ValueError("Bundle identities must align with the mask axis.")
        if self.ignore is not None and self.ignore.shape != self.masks.shape[1:]:
            raise ValueError("The ignore mask must match the instance spatial shape.")


def _packed_width(height: int, width: int) -> int:
    return (height * width + 7) // 8


def save_mask_bundle(
    path: str | Path,
    masks: NDArray[np.bool_],
    identities: list[int | str] | tuple[int | str, ...],
    *,
    ignore: NDArray[np.bool_] | None = None,
) -> Path:
    """Atomically save a versioned, bit-packed bundle for one image/frame."""
    path = Path(path)
    masks = np.asarray(masks, dtype=np.bool_)
    if masks.ndim != 3:
        raise ValueError("masks must have shape [N, H, W].")
    if len(identities) != masks.shape[0]:
        raise ValueError("identities must have one value per mask.")
    _, height, width = masks.shape
    packed_masks = np.packbits(
        masks.reshape(masks.shape[0], height * width),
        axis=1,
        bitorder=MASK_BITORDER,
    )
    if packed_masks.shape[1:] != (_packed_width(height, width),):
        raise RuntimeError("Unexpected packed mask shape.")

    arrays: dict[str, NDArray] = {
        "format_version": np.asarray(MASK_BUNDLE_VERSION, dtype=np.int64),
        "mask_shape": np.asarray(masks.shape, dtype=np.int64),
        "bitorder": np.asarray(MASK_BITORDER),
        "identities": np.asarray([str(value) for value in identities], dtype=np.str_),
        "packed_masks": packed_masks,
        "has_ignore": np.asarray(ignore is not None, dtype=np.bool_),
    }
    if ignore is not None:
        ignore = np.asarray(ignore, dtype=np.bool_)
        if ignore.shape != (height, width):
            raise ValueError("ignore must match the mask spatial shape.")
        arrays["packed_ignore"] = np.packbits(
            ignore.reshape(-1), bitorder=MASK_BITORDER
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".npz"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return path


def load_mask_bundle(path: str | Path) -> MaskBundle:
    """Load a versioned, bit-packed multi-instance mask bundle."""
    with np.load(path, allow_pickle=False) as data:
        if "format_version" not in data:
            raise ValueError(f"Unversioned mask archive at {path} is unsupported.")

        version = int(data["format_version"])
        if version != MASK_BUNDLE_VERSION:
            raise ValueError(f"Unsupported mask bundle version {version}.")
        shape = tuple(int(value) for value in data["mask_shape"])
        if len(shape) != 3:
            raise ValueError("mask_shape must contain [N, H, W].")
        count, height, width = shape
        bitorder = str(data["bitorder"])
        packed_masks = data["packed_masks"]
        expected = (count, _packed_width(height, width))
        if packed_masks.shape != expected:
            raise ValueError(
                f"Packed masks have shape {packed_masks.shape}, expected {expected}."
            )
        masks = (
            np.unpackbits(packed_masks, axis=1, bitorder=bitorder)[:, : height * width]
            .reshape(shape)
            .astype(np.bool_, copy=False)
        )
        identities = tuple(str(value) for value in data["identities"].tolist())
        ignore = None
        if bool(data["has_ignore"]):
            if "packed_ignore" not in data:
                raise ValueError(
                    "Bundle declares an ignore mask but does not contain it."
                )
            ignore = (
                np.unpackbits(data["packed_ignore"], bitorder=bitorder)[
                    : height * width
                ]
                .reshape(height, width)
                .astype(np.bool_, copy=False)
            )
    return MaskBundle(masks, identities, ignore)


def mask2xywh(mask: NDArray) -> list[int] | None:
    rows, cols = np.nonzero(mask)
    if len(rows) == 0:
        return None

    x_min = cols.min()
    x_max = cols.max()
    y_min = rows.min()
    y_max = rows.max()

    width = x_max - x_min + 1
    height = y_max - y_min + 1

    return [
        int(x_min),
        int(y_min),
        int(width),
        int(height),
    ]
