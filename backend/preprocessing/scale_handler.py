"""
SIH26166 — Scale Handling and Resolution Alignment (Module 06).

Handles cross-sensor multi-resolution scaling (e.g. matching OHRC at 0.21 m/px
to TMC-2 at 4.27 m/px or IIRS at 82.7 m/px) via anti-aliased Gaussian pyramids.
"""

from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger("sih26166.preprocessing.scale_handler")

# Standard instrument ground sampling distances (meters per pixel)
DEFAULT_RESOLUTIONS: dict[str, float] = {
    "OHRC": 0.21,
    "TMC2": 4.27,
    "TMC-2": 4.27,
    "TMC": 4.27,
    "IIRS": 82.7,
}


def compute_scale_ratio(res_source: float, res_target: float) -> float:
    """Compute spatial resolution scaling ratio between source and target.

    Parameters
    ----------
    res_source : float
        Pixel resolution of source image in meters/pixel (e.g. 0.21 for OHRC).
    res_target : float
        Pixel resolution of target image in meters/pixel (e.g. 4.27 for TMC-2).

    Returns
    -------
    float
        Scale factor ratio (res_source / res_target).
        Values < 1.0 indicate the source is higher resolution and should be downsampled.
    """
    if res_source <= 0 or res_target <= 0:
        raise ValueError(
            f"Resolutions must be positive floats, got res_source={res_source}, res_target={res_target}"
        )
    return float(res_source / res_target)


def downsample_to_match(
    source: np.ndarray,
    scale_ratio: float,
) -> np.ndarray:
    """Downsample source image according to scale_ratio using anti-aliased pyramid filtering.

    Parameters
    ----------
    source : np.ndarray
        2D or 3D input image array.
    scale_ratio : float
        Scale factor (res_source / res_target). If scale_ratio >= 1.0, the image
        is returned unchanged.

    Returns
    -------
    np.ndarray
        Downsampled image array in float32.
    """
    if scale_ratio >= 0.999:
        return source.astype(np.float32)

    if scale_ratio <= 0:
        raise ValueError(f"scale_ratio must be positive, got {scale_ratio}")

    orig_h, orig_w = source.shape[:2]
    target_w = max(1, int(round(orig_w * scale_ratio)))
    target_h = max(1, int(round(orig_h * scale_ratio)))

    if target_w == orig_w and target_h == orig_h:
        return source.astype(np.float32)

    # Process 2D or 3D
    if source.ndim == 2:
        return _downsample_2d(source, target_w, target_h)
    elif source.ndim == 3:
        # Downsample along spatial dimensions (bands, H, W) or (H, W, C)
        # Check if bands-first or bands-last
        if source.shape[0] < min(source.shape[1], source.shape[2]):
            # (bands, H, W)
            bands = source.shape[0]
            out = np.zeros((bands, target_h, target_w), dtype=np.float32)
            for b in range(bands):
                out[b] = _downsample_2d(source[b], target_w, target_h)
            return out
        else:
            # (H, W, C)
            c = source.shape[2]
            out = np.zeros((target_h, target_w, c), dtype=np.float32)
            for ch in range(c):
                out[:, :, ch] = _downsample_2d(source[:, :, ch], target_w, target_h)
            return out
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {source.shape}")


def _downsample_2d(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    """Apply pyrDown iteratively followed by INTER_AREA resize for 2D image."""
    curr = np.nan_to_num(img, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    curr_h, curr_w = curr.shape

    # Apply cv2.pyrDown iteratively while current size is more than 2x target size
    while (curr_w >= 2 * target_w) and (curr_h >= 2 * target_h) and (curr_w > 4) and (curr_h > 4):
        curr = cv2.pyrDown(curr)
        curr_h, curr_w = curr.shape

    # Final anti-aliased resize to exact target dimensions
    if curr_w != target_w or curr_h != target_h:
        curr = cv2.resize(curr, (target_w, target_h), interpolation=cv2.INTER_AREA)

    return curr.astype(np.float32)


def build_gaussian_pyramid(image: np.ndarray, levels: int) -> list[np.ndarray]:
    """Build a multi-resolution Gaussian pyramid of the input image.

    Parameters
    ----------
    image : np.ndarray
        2D input array.
    levels : int
        Number of pyramid levels (level 0 is original).

    Returns
    -------
    list[np.ndarray]
        List of length `levels` with progressively halved resolution images (float32).
    """
    if levels < 1:
        raise ValueError(f"Pyramid levels must be >= 1, got {levels}")

    curr = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    pyramid = [curr]

    for _ in range(1, levels):
        if curr.shape[0] < 2 or curr.shape[1] < 2:
            break
        curr = cv2.pyrDown(curr)
        pyramid.append(curr.astype(np.float32))

    return pyramid


def get_resolution(metadata: dict[str, Any] | None, instrument: str) -> float:
    """Extract or infer pixel resolution in meters/pixel for a given instrument.

    Parameters
    ----------
    metadata : dict, optional
        Metadata dictionary from PDS4 loader or upload metadata extractor.
    instrument : str
        Instrument identifier string ("OHRC", "TMC2", "TMC-2", "IIRS").

    Returns
    -------
    float
        Pixel resolution in meters per pixel.
    """
    norm_inst = instrument.upper().replace(" ", "").replace("-", "")

    # Try extracting from metadata dict if available
    if metadata and isinstance(metadata, dict):
        # 1. Direct key
        if "pixel_resolution" in metadata and metadata["pixel_resolution"] is not None:
            try:
                return float(metadata["pixel_resolution"])
            except (ValueError, TypeError):
                pass

        # 2. PDS4 isda_product_params dict
        isda_pp = metadata.get("pds4_isda_product_params") or metadata.get("isda_product_params")
        if isinstance(isda_pp, dict):
            pr_val = isda_pp.get("pixel_resolution")
            if isinstance(pr_val, dict) and "value" in pr_val:
                try:
                    return float(pr_val["value"])
                except (ValueError, TypeError):
                    pass
            elif pr_val is not None:
                try:
                    return float(pr_val)
                except (ValueError, TypeError):
                    pass

        # 3. PDS4 tmc2_product_params dict
        tmc_pp = metadata.get("pds4_tmc2_product_params") or metadata.get("tmc2_product_params")
        if isinstance(tmc_pp, dict):
            pr_val = tmc_pp.get("pixel_resolution")
            if isinstance(pr_val, dict) and "value" in pr_val:
                try:
                    return float(pr_val["value"])
                except (ValueError, TypeError):
                    pass

    # Default fallback per instrument specification
    for key, val in DEFAULT_RESOLUTIONS.items():
        if key.replace("-", "") == norm_inst:
            return val

    logger.warning("Unrecognized instrument '%s', defaulting resolution to 1.0 m/px.", instrument)
    return 1.0
