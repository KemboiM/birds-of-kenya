"""Approximate camera-to-bird distance from one photo.

Pinhole camera model:   distance = real_size x focal_length_px / size_in_pixels

  real_size        species body length (or wingspan for a bird in flight with
                   wings spread), as a range from species_info
  focal_length_px  from the photo's EXIF "35 mm equivalent focal length",
                   scaled to the image's pixel diagonal (35 mm frame diagonal = 43.27 mm)
  size_in_pixels   longest side of the bird's detection box

The answer is a range and deliberately approximate (often +/-30-50 %): pose,
partial views, digital zoom, cropped photos and box tightness all add error.
"""

from PIL import Image

FULL_FRAME_DIAGONAL_MM = 43.27
DEFAULT_PHONE_FOCAL_35MM = 26.0      # typical phone main camera, used when EXIF is missing
EXIF_FOCAL_35MM = 0xA405
EXIF_IFD = 0x8769
POSTURE_MARGIN = 0.15                # widen the range for pose and box error


def focal_length_35mm(image_path):
    """Return (focal length in mm, 35 mm equivalent; True if read from EXIF)."""
    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            value = exif.get_ifd(EXIF_IFD).get(EXIF_FOCAL_35MM) or exif.get(EXIF_FOCAL_35MM)
            if value:
                return float(value), True
    except Exception:
        pass
    return DEFAULT_PHONE_FOCAL_35MM, False


def _range(values):
    if not values:
        return None
    if isinstance(values, (int, float)):
        return float(values), float(values)
    lo, hi = float(min(values)), float(max(values))
    return lo, hi


def estimate_distance(det, image_size, focal_35mm, sizes):
    """Return a dict with an approximate distance range in metres, or None.

    det         Detection box (pixels)
    image_size  (width, height) in pixels
    sizes       species_info["size"]["adult"], e.g. {"length_cm": [63, 75], "wingspan_cm": [175, 240]}
    """
    if not sizes:
        return None
    w, h = image_size
    diag_px = (w ** 2 + h ** 2) ** 0.5
    focal_px = focal_35mm * diag_px / FULL_FRAME_DIAGONAL_MM
    size_px = max(det.width, det.height)
    if size_px < 4:
        return None

    # A box much wider than tall usually means a flying bird with wings spread.
    wings_spread = det.width > 1.6 * det.height
    basis = "wingspan" if wings_spread and sizes.get("wingspan_cm") else "body length"
    real = _range(sizes.get("wingspan_cm") if basis == "wingspan" else sizes.get("length_cm"))
    if not real:
        return None

    near = real[0] / 100 * focal_px / size_px * (1 - POSTURE_MARGIN)
    far = real[1] / 100 * focal_px / size_px * (1 + POSTURE_MARGIN)
    return {"min_m": round(near, 1), "max_m": round(far, 1), "based_on": basis}


def describe(dist, focal_from_exif):
    if not dist:
        return "distance unknown (species size not recorded yet)"
    lo, hi = dist["min_m"], dist["max_m"]
    fmt = (lambda v: f"{v:.0f}") if hi >= 10 else (lambda v: f"{v:.1f}")
    note = "" if focal_from_exif else ", camera focal length assumed (no EXIF)"
    return f"about {fmt(lo)}-{fmt(hi)} m away (approximate, from {dist['based_on']}{note})"
