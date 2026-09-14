from pathlib import Path

import pytest

from annotations import image_mask_path


@pytest.mark.parametrize("filepath", ["../a.jpg", "/images/a.jpg"])
def test_mask_path_rejects_unsafe_paths(filepath):
    with pytest.raises(ValueError):
        image_mask_path(Path("masks"), filepath)


def test_mask_path_preserves_nested_image_folders():
    assert image_mask_path(Path("masks/images"), "images/0/sample/a.jpg") == Path(
        "masks/images/0/sample/a.npz"
    )
