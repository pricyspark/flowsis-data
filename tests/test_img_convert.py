import json

import numpy as np
from PIL import Image
import pytest

import img_convert


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_crop_rotation_preserves_corner_positions(tmp_path, monkeypatch, rotation):
    pixels = np.zeros((32, 48, 3), dtype=np.uint8)
    pixels[:16, :24] = (240, 20, 20)
    pixels[:16, 24:] = (20, 240, 20)
    pixels[16:, :24] = (20, 20, 240)
    pixels[16:, 24:] = (240, 240, 20)
    source = tmp_path / "source.png"
    destination = tmp_path / "converted.jpg"
    Image.fromarray(pixels).save(source)
    monkeypatch.setattr(img_convert, "crop_marked_square", lambda image: image)

    img_convert.save_cropped_image(source, destination, rotation)

    with Image.open(destination) as image:
        actual = np.asarray(image)
    expected = np.rot90(pixels, -(rotation // 90))
    assert actual.shape == expected.shape
    # Compare interiors to avoid JPEG ringing at the color boundaries.
    for y in (8, actual.shape[0] - 8):
        for x in (8, actual.shape[1] - 8):
            np.testing.assert_allclose(actual[y, x], expected[y, x], atol=5)


def test_orientation_corrections_are_optional_and_load_paths(tmp_path):
    path = tmp_path / "orientations.json"
    assert img_convert.load_orientation_corrections(path) == {}
    path.write_text(json.dumps({"24/photo.jpg": 180, "multi/0/other.jpg": 90}))
    assert img_convert.load_orientation_corrections(path) == {
        "24/photo.jpg": 180,
        "multi/0/other.jpg": 90,
    }


@pytest.mark.parametrize(
    "value",
    [
        [],
        {"../photo.jpg": 180},
        {"/photo.jpg": 180},
        {"photo.jpg": 45},
        {"photo.jpg": "180"},
        {"photo.jpg": True},
    ],
)
def test_orientation_corrections_reject_invalid_entries(tmp_path, value):
    path = tmp_path / "orientations.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        img_convert.load_orientation_corrections(path)


def test_force_conversion_reapplies_correction_from_raw_without_double_rotation(
    tmp_path, monkeypatch
):
    raw = tmp_path / "raw-images"
    output = tmp_path / "images"
    (raw / "24").mkdir(parents=True)
    source = raw / "24/photo.png"
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    pixels[:16] = (240, 20, 20)
    pixels[16:] = (20, 20, 240)
    Image.fromarray(pixels).save(source)
    corrections = tmp_path / "image_orientation_corrections.json"
    corrections.write_text(json.dumps({"24/photo.jpg": 180}))
    monkeypatch.setattr(img_convert, "SOURCE_DIR", raw)
    monkeypatch.setattr(img_convert, "OUTPUT_DIR", output)
    monkeypatch.setattr(img_convert, "ORIENTATION_PATH", corrections)
    monkeypatch.setattr(img_convert, "crop_marked_square", lambda image: image)

    img_convert.main(force=True)
    destination = output / "24/photo.jpg"
    first = destination.read_bytes()
    with Image.open(destination) as image:
        assert image.getpixel((8, 8))[2] > 200
    img_convert.main(force=True)
    assert destination.read_bytes() == first
