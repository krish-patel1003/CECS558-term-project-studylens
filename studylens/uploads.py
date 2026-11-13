"""Safe handling of user-uploaded images.

Policy (also documented in README "Data handling"):
* Accepted: PNG, JPEG, WebP, BMP, GIF (first frame), TIFF (first page), and HEIC/HEIF if the
  optional ``pillow-heif`` package is installed. The format is detected by decoding the bytes
  with Pillow; the file extension and client-supplied content type are not trusted.
* Normalisation: EXIF orientation is applied (phone photos), transparency is flattened onto
  white, palette / greyscale / CMYK / 16-bit images are converted to 8-bit RGB, and very small
  images are upscaled so the vision model gets enough pixels.
* Size limit MAX_UPLOAD_MB (default 10 MB) and pixel limit MAX_IMAGE_PIXELS (default 40 MP,
  guards against decompression bombs). Large images are downscaled in memory before inference.
* Uploads are processed in memory and are NEVER written to disk by the server. The image is
  discarded when the request finishes; the server keeps no per-user state.
"""

from __future__ import annotations

import io

from PIL import Image, ImageOps, UnidentifiedImageError

from .config import settings

try:  # optional HEIC/HEIF (iPhone photos)
    from pillow_heif import register_heif_opener  # type: ignore

    register_heif_opener()
    HEIF = True
except ImportError:  # pragma: no cover
    HEIF = False

ALLOWED_FORMATS = {"PNG", "JPEG", "MPO", "WEBP", "BMP", "GIF", "TIFF"} | ({"HEIF", "HEIC"} if HEIF else set())
MIN_SIDE = 64


class UploadError(ValueError):
    pass


def to_rgb(img: Image.Image) -> Image.Image:
    """Flatten any mode (RGBA, LA, P with transparency, I;16, CMYK, ...) to RGB on white."""
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        img = img.point(lambda v: v * (1 / 256)).convert("L")
    if img.mode == "P":
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img.convert("RGBA"), mask=img.convert("RGBA").getchannel("A"))
        return bg
    return img.convert("RGB")


def load_upload(data: bytes) -> Image.Image:
    if not data:
        raise UploadError("Empty upload.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise UploadError(f"File too large (limit {settings.max_upload_mb} MB).")
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels
    try:
        probe = Image.open(io.BytesIO(data))
        fmt = probe.format
        probe.verify()  # integrity check; the object is unusable afterwards
        img = Image.open(io.BytesIO(data))
        if getattr(img, "n_frames", 1) > 1:
            img.seek(0)  # first frame of an animated GIF/WebP or multi-page TIFF
        img.load()
    except Image.DecompressionBombError as exc:
        raise UploadError("Image has too many pixels.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise UploadError("File is not a readable image (supported: " + ", ".join(sorted(ALLOWED_FORMATS)) + ").") from exc
    if fmt not in ALLOWED_FORMATS:
        raise UploadError(f"Unsupported image format {fmt}; supported: {', '.join(sorted(ALLOWED_FORMATS))}.")
    if img.width * img.height > settings.max_image_pixels:
        raise UploadError("Image has too many pixels.")
    img = ImageOps.exif_transpose(img)
    img = to_rgb(img)
    if min(img.size) < MIN_SIDE:
        f = MIN_SIDE / min(img.size)
        img = img.resize((max(MIN_SIDE, round(img.width * f)), max(MIN_SIDE, round(img.height * f))), Image.LANCZOS)
    return img
