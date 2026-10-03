"""
Offline unit tests for image preprocessing.

Verifies:
- RGB normalization (including alpha compositing without color loss)
- aspect-ratio preserving downscaling and the "never upscale by default" rule
- PNG/JPEG encoding of valid images and rejection of invalid input
- data URL construction

All images are generated in-memory with Pillow; no network is used.
"""

import base64
import io

import pytest
from PIL import Image

from backend.vision_agent.image_preprocessor import (
    MIN_MAX_IMAGE_DIMENSION,
    ImagePreprocessingError,
    PreparedImage,
    downscale,
    encode_image,
    load_image,
    prepare_image,
    to_data_url,
    to_rgb,
)


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class TestLoadImage:
    def test_loads_from_bytes(self):
        raw = _png_bytes(Image.new("RGB", (20, 10), (10, 20, 30)))
        assert load_image(raw).size == (20, 10)

    def test_loads_from_path(self, tmp_path):
        path = tmp_path / "asset.png"
        path.write_bytes(_png_bytes(Image.new("RGB", (8, 4), (0, 0, 0))))
        assert load_image(str(path)).size == (8, 4)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(ImagePreprocessingError):
            load_image(str(tmp_path / "missing.png"))

    def test_empty_bytes_raise(self):
        with pytest.raises(ImagePreprocessingError):
            load_image(b"")

    def test_non_image_bytes_raise(self):
        with pytest.raises(ImagePreprocessingError):
            load_image(b"this is not an image")


class TestRgbConversion:
    def test_rgba_is_composited_over_white(self):
        image = Image.new("RGBA", (4, 4), (0, 0, 0, 0))  # fully transparent
        converted = to_rgb(image)
        assert converted.mode == "RGB"
        assert converted.size == (4, 4)
        assert converted.getpixel((0, 0)) == (255, 255, 255)

    def test_grayscale_is_converted(self):
        converted = to_rgb(Image.new("L", (4, 4), 128))
        assert converted.mode == "RGB"
        assert converted.getpixel((0, 0)) == (128, 128, 128)

    def test_palette_image_is_converted(self):
        assert to_rgb(Image.new("P", (4, 4))).mode == "RGB"

    def test_rgb_passes_through_unchanged(self):
        image = Image.new("RGB", (3, 3), (1, 2, 3))
        assert to_rgb(image) is image


class TestDownscale:
    def test_aspect_ratio_is_preserved(self):
        assert downscale(Image.new("RGB", (2000, 1000)), 1000).size == (1000, 500)

    def test_portrait_image_is_capped_on_the_long_side(self):
        assert downscale(Image.new("RGB", (600, 1800)), 900).size == (300, 900)

    def test_image_within_limit_is_untouched(self):
        image = Image.new("RGB", (100, 50))
        assert downscale(image, 1600) is image

    def test_too_small_cap_is_rejected(self):
        with pytest.raises(ImagePreprocessingError):
            downscale(Image.new("RGB", (100, 50)), MIN_MAX_IMAGE_DIMENSION - 1)


class TestPrepareImage:
    def test_png_default_preserves_native_resolution(self):
        prepared = prepare_image(_png_bytes(Image.new("RGB", (320, 240))))
        assert isinstance(prepared, PreparedImage)
        assert prepared.image_format == "PNG"
        assert prepared.mime_type == "image/png"
        assert (prepared.width, prepared.height) == (320, 240)
        assert (prepared.original_width, prepared.original_height) == (320, 240)
        assert prepared.resized is False
        assert prepared.size_bytes == len(prepared.data) > 0
        assert Image.open(io.BytesIO(prepared.data)).mode == "RGB"

    def test_downscales_and_reports_resizing(self):
        prepared = prepare_image(
            _png_bytes(Image.new("RGB", (4000, 2000))), max_dimension=1000
        )
        assert (prepared.width, prepared.height) == (1000, 500)
        assert (prepared.original_width, prepared.original_height) == (4000, 2000)
        assert prepared.resized is True

    def test_small_images_are_not_upscaled_by_default(self):
        prepared = prepare_image(_png_bytes(Image.new("RGB", (50, 40))), max_dimension=1600)
        assert (prepared.width, prepared.height) == (50, 40)
        assert prepared.resized is False

    def test_upscaling_is_not_offered_at_all(self):
        # Interpolating a small figure upward would fabricate visual detail, so
        # the API deliberately exposes no upscale switch.
        import inspect

        assert "allow_upscale" not in inspect.signature(prepare_image).parameters

    def test_jpeg_encoding_is_supported(self):
        prepared = prepare_image(
            _png_bytes(Image.new("RGB", (60, 30), (255, 0, 0))),
            image_format="JPEG",
            jpeg_quality=85,
        )
        assert prepared.image_format == "JPEG"
        assert prepared.mime_type == "image/jpeg"
        reopened = Image.open(io.BytesIO(prepared.data))
        assert reopened.format == "JPEG"
        assert reopened.mode == "RGB"

    def test_rgba_input_is_normalized_for_jpeg(self):
        prepared = prepare_image(
            _png_bytes(Image.new("RGBA", (30, 30), (0, 0, 255, 128))), image_format="JPEG"
        )
        assert Image.open(io.BytesIO(prepared.data)).mode == "RGB"

    def test_data_url_property(self):
        prepared = prepare_image(_png_bytes(Image.new("RGB", (10, 10))))
        assert prepared.data_url.startswith("data:image/png;base64,")


class TestEncodingErrors:
    def test_unsupported_format_is_rejected(self):
        with pytest.raises(ImagePreprocessingError):
            encode_image(Image.new("RGB", (10, 10)), image_format="BMP")

    def test_invalid_input_bytes_raise(self):
        with pytest.raises(ImagePreprocessingError):
            prepare_image(b"definitely not an image")

    def test_truncated_png_raises(self):
        truncated = _png_bytes(Image.new("RGB", (64, 64)))[:20]
        with pytest.raises(ImagePreprocessingError):
            prepare_image(truncated)

    def test_data_url_from_empty_bytes_raises(self):
        with pytest.raises(ImagePreprocessingError):
            to_data_url(b"", "image/png")

    def test_data_url_round_trip(self):
        raw = _png_bytes(Image.new("RGB", (6, 6), (7, 8, 9)))
        url = to_data_url(raw, "image/png")
        assert url.startswith("data:image/png;base64,")
        assert base64.b64decode(url.split(",", 1)[1]) == raw