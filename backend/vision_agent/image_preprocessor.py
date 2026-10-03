"""
Lightweight image preprocessing for the Vision Agent.

Responsibilities
----------------
- Load an extracted asset region into a PIL image
- Normalize to RGB (transparency is composited onto white, never dropped)
- Downscale to a configurable maximum dimension while preserving aspect ratio
- Encode as PNG (default) or JPEG

Deliberate non-goals
--------------------
Research figures are evidence, so this module never enhances, recolors,
sharpens, crops away labels, removes text, or invents pixels. It performs no
upscaling by default: an image smaller than the cap is passed through at its
native resolution so small axis labels are not resampled unnecessarily.
"""

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

from PIL import Image

SUPPORTED_FORMATS = ("PNG", "JPEG")

_MIME_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg"}

# Below this longest-side value a configurable cap is treated as a mistake: the
# user would be trading away the legibility the Vision Agent depends on.
MIN_MAX_IMAGE_DIMENSION = 64


class ImagePreprocessingError(Exception):
    """Raised when an image cannot be loaded, normalized, or encoded."""
    pass


@dataclass
class PreparedImage:
    """A normalized, encoded image ready to be sent to a multimodal model."""

    data: bytes
    image_format: str
    mime_type: str
    width: int
    height: int
    original_width: int
    original_height: int
    resized: bool

    @property
    def size_bytes(self) -> int:
        return len(self.data)

    @property
    def data_url(self) -> str:
        """Base64 data URL suitable for OpenAI-compatible vision payloads."""
        return to_data_url(self.data, self.mime_type)


def load_image(source: Union[str, Path, bytes, bytearray]) -> Image.Image:
    """Load an image from a path or raw bytes into a PIL image."""
    try:
        if isinstance(source, (bytes, bytearray)):
            if not source:
                raise ImagePreprocessingError("Image byte content is empty")
            return Image.open(io.BytesIO(bytes(source)))
        path = Path(source)
        if not path.exists():
            raise ImagePreprocessingError(f"Image file not found: {path}")
        return Image.open(str(path))
    except ImagePreprocessingError:
        raise
    except Exception as exc:
        raise ImagePreprocessingError(f"Unable to load image: {exc}") from exc


def to_rgb(image: Image.Image) -> Image.Image:
    """
    Convert an image to RGB without changing its visual content.

    Images with an alpha channel are composited over a white background, which
    preserves the rendered appearance of a PDF region while producing a format
    every multimodal endpoint accepts.
    """
    try:
        if image.mode == "RGB":
            return image
        if image.mode in ("RGBA", "LA", "PA") or "transparency" in image.info:
            rgba = image.convert("RGBA")
            canvas = Image.new("RGB", rgba.size, (255, 255, 255))
            canvas.paste(rgba, mask=rgba.split()[-1])
            return canvas
        return image.convert("RGB")
    except Exception as exc:
        raise ImagePreprocessingError(f"Unable to convert image to RGB: {exc}") from exc


def downscale(image: Image.Image, max_dimension: int) -> Image.Image:
    """
    Downscale so the longest side fits ``max_dimension``.

    Aspect ratio is preserved exactly (a single scale factor is applied to both
    axes) and images already within the limit are returned unchanged.
    """
    if max_dimension < MIN_MAX_IMAGE_DIMENSION:
        raise ImagePreprocessingError(
            f"max_dimension must be at least {MIN_MAX_IMAGE_DIMENSION}px, "
            f"got {max_dimension}px"
        )

    width, height = image.size
    longest = max(width, height)
    if longest <= max_dimension:
        return image

    scale = max_dimension / float(longest)
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    try:
        return image.resize(target, resampling)
    except Exception as exc:
        raise ImagePreprocessingError(f"Unable to resize image: {exc}") from exc


def encode_image(
    image: Image.Image,
    image_format: str = "PNG",
    jpeg_quality: int = 90,
) -> bytes:
    """Encode a PIL image into bytes using a supported format."""
    normalized_format = (image_format or "PNG").strip().upper()
    if normalized_format == "JPG":
        normalized_format = "JPEG"
    if normalized_format not in SUPPORTED_FORMATS:
        raise ImagePreprocessingError(
            f"Unsupported image format '{image_format}'. "
            f"Supported formats: {', '.join(SUPPORTED_FORMATS)}"
        )

    buffer = io.BytesIO()
    try:
        if normalized_format == "JPEG":
            image.save(buffer, format="JPEG", quality=max(1, min(100, int(jpeg_quality))))
        else:
            image.save(buffer, format="PNG")
    except Exception as exc:
        raise ImagePreprocessingError(f"Unable to encode image: {exc}") from exc

    data = buffer.getvalue()
    if not data:
        raise ImagePreprocessingError("Image encoding produced no data")
    return data


def prepare_image(
    source: Union[str, Path, bytes, bytearray],
    max_dimension: int = 1600,
    image_format: str = "PNG",
    jpeg_quality: int = 90,
) -> PreparedImage:
    """
    Load, normalize, cap resolution, and encode an image for the vision model.

    There is deliberately no upscaling option: interpolating a small figure
    upward would fabricate visual detail that is not present in the paper, and
    the Vision Agent must never analyse invented pixels.
    """
    image = load_image(source)
    original_width, original_height = image.size
    normalized = to_rgb(image)

    resized = False
    longest = max(original_width, original_height)
    if longest > max_dimension:
        normalized = downscale(normalized, max_dimension)
        resized = True

    final_format = (image_format or "PNG").strip().upper()
    if final_format == "JPG":
        final_format = "JPEG"
    data = encode_image(normalized, final_format, jpeg_quality)

    width, height = normalized.size
    return PreparedImage(
        data=data,
        image_format=final_format,
        mime_type=_MIME_TYPES.get(final_format, "image/png"),
        width=width,
        height=height,
        original_width=original_width,
        original_height=original_height,
        resized=resized,
    )


def to_data_url(image_bytes: bytes, mime_type: Optional[str] = None) -> str:
    """Encode raw image bytes as a base64 data URL."""
    if not image_bytes:
        raise ImagePreprocessingError("Cannot build a data URL from empty bytes")
    resolved_mime = mime_type or "image/png"
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{resolved_mime};base64,{encoded}"