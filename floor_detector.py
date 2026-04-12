"""Floor boundary detection using color histogram similarity.

Scans from the bottom of the image upward, comparing each row strip's
color distribution to a floor reference (bottom 10% of the image).
Where the similarity drops below a threshold, the floor ends.

Returns the floor boundary in normalized image coords (0=top, 1=bottom).
Waypoints above this boundary are not on the floor.

Uses only Pillow + numpy (no OpenCV dependency).
"""

import logging

import numpy as np
from PIL import Image

logger = logging.getLogger("qBc_Ai.floor")

# How much of the bottom of the image to use as floor reference
FLOOR_REF_FRACTION = 0.10

# Histogram similarity threshold (0=identical, 1=different)
SIMILARITY_THRESHOLD = 0.55

# Scan in strips of this many rows for speed
STRIP_HEIGHT = 16

# Minimum floor boundary (safety fallback — never report floor above this)
MIN_FLOOR_Y = 0.25

# Histogram bins per channel
HIST_BINS = 16


def _rgb_histogram(pixels: np.ndarray) -> np.ndarray:
    """Compute a normalized RGB color histogram from pixel array.

    Args:
        pixels: (N, 3) array of RGB values.

    Returns:
        Normalized histogram vector.
    """
    hist = np.zeros(HIST_BINS ** 3, dtype=np.float64)
    # Quantize each channel to HIST_BINS levels
    quantized = (pixels * (HIST_BINS / 256.0)).astype(np.int32)
    quantized = np.clip(quantized, 0, HIST_BINS - 1)
    # Flatten to 1D index
    indices = (quantized[:, 0] * HIST_BINS * HIST_BINS +
               quantized[:, 1] * HIST_BINS +
               quantized[:, 2])
    np.add.at(hist, indices, 1)
    # Normalize
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _bhattacharyya_distance(h1: np.ndarray, h2: np.ndarray) -> float:
    """Compute Bhattacharyya distance between two normalized histograms.

    Returns 0 for identical distributions, 1 for completely different.
    """
    bc = np.sum(np.sqrt(h1 * h2))
    bc = min(bc, 1.0)
    return float(np.sqrt(1.0 - bc))


def detect_floor_boundary(image_path: str) -> float:
    """Detect where the floor ends in the image.

    Scans from bottom upward comparing color histograms to the floor
    reference region.

    Args:
        image_path: Path to the camera frame.

    Returns:
        Normalized y coordinate of the floor boundary (0-1).
        Returns MIN_FLOOR_Y on failure (conservative fallback).
    """
    try:
        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        pixels = np.array(img)  # (h, w, 3)

        # Build floor reference histogram from bottom strip
        ref_top = int(h * (1.0 - FLOOR_REF_FRACTION))
        ref_pixels = pixels[ref_top:h, :, :].reshape(-1, 3)
        ref_hist = _rgb_histogram(ref_pixels)

        # Scan upward in strips
        boundary_row = 0
        y = ref_top - STRIP_HEIGHT

        while y >= 0:
            strip_pixels = pixels[y:y + STRIP_HEIGHT, :, :].reshape(-1, 3)
            strip_hist = _rgb_histogram(strip_pixels)

            dist = _bhattacharyya_distance(ref_hist, strip_hist)

            if dist > SIMILARITY_THRESHOLD:
                boundary_row = y + STRIP_HEIGHT
                break

            y -= STRIP_HEIGHT

        boundary_norm = max(boundary_row / h, MIN_FLOOR_Y)
        logger.info(
            "Floor boundary detected at y=%.2f (row %d/%d)",
            boundary_norm, boundary_row, h,
        )
        return boundary_norm

    except Exception as e:
        logger.error("Floor detection failed: %s", e)
        return MIN_FLOOR_Y
