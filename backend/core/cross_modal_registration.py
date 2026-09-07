"""
SIH26166 — Standalone Cross-Modal Registration via Phase Congruency.

Provides resolution-normalized, illumination-invariant image registration
for cross-modal lunar orbital pairs (specifically TMC-2 visible vs IIRS infrared).

Algorithmic architecture:
1. Validates resolution ordering (fine_res < coarse_res).
2. Downsamples the higher-resolution fine image (e.g. TMC-2 at 4.27 m/px) to
   approximately match the coarse image resolution (e.g. IIRS at 82.70 m/px)
   using anti-aliased Gaussian pyramid filtering (`scale_handler.downsample_to_match`).
3. Computes frequency-domain phase congruency maps (`illumination.compute_phase_congruency`)
   on both downsampled fine and coarse images to yield contrast- and illumination-invariant
   structural edge maps.
4. Normalizes and converts float32 [0.0, 1.0] phase congruency maps into uint8 [0, 255]
   representations for standard gradient-based feature extraction.
5. Runs the classical feature pipeline (SIFT keypoints & descriptors, FLANN kNN
   matching, Lowe ratio test, MAGSAC++ robust estimation) on the phase-congruency
   representations to compute the transformation in coarse-resolution coordinates.
6. Analytically composes the coarse-space transformation matrix with the known
   downsampling scale factor using `compose_scale_transform()`, producing a full-resolution
   transformation matrix mapping original fine coordinates directly into coarse coordinates.

SCOPE & LIMITATIONS (WHAT THIS DOES AND DOES NOT DO):
- Uses phase congruency (not MIND or RIFT) as the illumination/modality-invariant representation.
- Uses the existing classical SIFT/FLANN/MAGSAC++ chain on the phase congruency map.
- Validated on synthetic modality-simulating transforms (with genuine contrast/intensity inversions)
  and real-data pipeline-completion sanity checks.
- NOT validated on real overlapping TMC-2/IIRS pairs because no overlapping fixtures
  currently exist in the project repository.
- Standalone and unwired: NOT called by `registration_service.py`, `cli.py`, or `api/register.py`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.core.cross_scale_registration import compose_scale_transform
from backend.features.models import FilteredMatch, SIFTFeatures
from backend.features.sift import extract_sift
from backend.geometry.estimation import estimate_transform
from backend.geometry.models import ErrorMetrics, EstimatorMethod, GeometricResult, TransformModel
from backend.matching.flann import flann_knn_match
from backend.matching.ratio_test import apply_ratio_test
from backend.preprocessing.datamodel import FeatureImage, RawImage
from backend.preprocessing.grayscale import to_feature_image
from backend.preprocessing.illumination import compute_phase_congruency
from backend.preprocessing.scale_handler import compute_scale_ratio, downsample_to_match

logger = logging.getLogger("sih26166.core.cross_modal_registration")


# ===================================================================
# Data Model
# ===================================================================

@dataclass
class CrossModalRegistrationResult:
    """Result of cross-modal image registration via phase congruency.

    Attributes
    ----------
    success : bool
        Whether geometric transformation estimation succeeded with sufficient inliers.
    scale_ratio : float
        Computed spatial resolution scale ratio (fine_res / coarse_res < 1.0).
    downsampled_fine_shape : tuple[int, int]
        Dimensions (height, width) of the fine image after anti-aliased downsampling.
    pc_map_fine : np.ndarray | None
        Normalized phase congruency map of the downsampled fine image (2D float32 in [0, 1]).
    pc_map_coarse : np.ndarray | None
        Normalized phase congruency map of the coarse image (2D float32 in [0, 1]).
    coarse_transform_matrix : np.ndarray | None
        Estimated transformation matrix mapping downsampled fine PC coordinates
        to coarse image PC coordinates (2x3 for affine).
    full_resolution_transform_matrix : np.ndarray | None
        Composed transformation matrix mapping original full-resolution fine image
        pixel coordinates directly to coarse image pixel coordinates.
    inlier_count : int
        Number of geometrically verified inlier feature correspondences.
    total_correspondences : int
        Total number of candidate correspondences that passed the Lowe ratio test.
    inlier_rmse : float
        Root mean square reprojection error of inliers, measured strictly in
        **coarse-image pixel units** (not fine-image pixel units).
    inlier_median_error : float
        Median reprojection error of inliers in coarse-image pixel units.
    inlier_max_error : float
        Maximum reprojection error of inliers in coarse-image pixel units.
    all_rmse : float
        Root mean square reprojection error across all candidate correspondences.
    all_median_error : float
        Median reprojection error across all candidate correspondences.
    per_match_errors : np.ndarray | None
        Per-correspondence Euclidean reprojection errors in coarse pixel units.
    matches : list[FilteredMatch]
        Filtered correspondences that passed the ratio test.
    inlier_mask : np.ndarray | None
        Boolean mask indicating which matches are inliers under the estimated model.
    transform_model : str
        Transformation model used: ``'affine'``.
    failure_reason : str
        Description of failure mode if success is False, otherwise empty string.
    """

    success: bool
    scale_ratio: float
    downsampled_fine_shape: tuple[int, int]
    pc_map_fine: np.ndarray | None = None
    pc_map_coarse: np.ndarray | None = None
    coarse_transform_matrix: np.ndarray | None = None
    full_resolution_transform_matrix: np.ndarray | None = None
    inlier_count: int = 0
    total_correspondences: int = 0
    inlier_rmse: float = 0.0
    inlier_median_error: float = 0.0
    inlier_max_error: float = 0.0
    all_rmse: float = 0.0
    all_median_error: float = 0.0
    per_match_errors: np.ndarray | None = None
    matches: list[FilteredMatch] = field(default_factory=list)
    inlier_mask: np.ndarray | None = None
    transform_model: str = "affine"
    failure_reason: str = ""


# ===================================================================
# Helper Functions
# ===================================================================

def _pc_to_uint8(pc_map: np.ndarray) -> np.ndarray:
    """Convert float32 phase congruency map [0.0, 1.0] to uint8 [0, 255] for SIFT.

    Applies min-max contrast stretching across finite values to maximize dynamic
    range in uint8 space, followed by clipping to [0, 255].
    """
    clean = np.nan_to_num(pc_map, nan=0.0, posinf=0.0, neginf=0.0)
    p_min = float(clean.min())
    p_max = float(clean.max())
    if p_max - p_min > 1e-6:
        scaled = (clean - p_min) / (p_max - p_min) * 255.0
    else:
        scaled = clean * 255.0
    return np.clip(scaled, 0.0, 255.0).astype(np.uint8)


def _to_feature_img(img: np.ndarray | RawImage | FeatureImage) -> FeatureImage:
    """Ensure input is converted to a standardized 2D float32 FeatureImage."""
    if isinstance(img, FeatureImage):
        return img
    if isinstance(img, RawImage):
        return to_feature_image(img)
    if isinstance(img, np.ndarray):
        raw = RawImage(
            data=img,
            width=img.shape[1],
            height=img.shape[0],
            num_bands=1 if img.ndim == 2 else img.shape[2],
            dtype=img.dtype,
            source_format="ndarray",
            metadata={},
        )
        return to_feature_image(raw)
    raise TypeError(f"Expected np.ndarray, RawImage, or FeatureImage, got {type(img)}")


# ===================================================================
# Main Cross-Modal Registration Function
# ===================================================================

def register_cross_modal(
    fine_image: np.ndarray,
    fine_resolution_m_per_px: float,
    coarse_image: np.ndarray,
    coarse_resolution_m_per_px: float,
    transform_model: str = "affine",
    nscale: int = 4,
    norient: int = 6,
    ratio_threshold: float = 0.75,
    reproj_threshold: float = 3.0,
    confidence: float = 0.999,
) -> CrossModalRegistrationResult:
    """Register a fine-resolution image against a coarse-resolution image cross-modally.

    Uses anti-aliased scale-normalized downsampling, illumination-invariant phase
    congruency representation conversion, standard SIFT/FLANN/MAGSAC++ matching,
    and analytical affine transform composition.

    Parameters
    ----------
    fine_image : np.ndarray
        Higher-resolution image array (smaller ground sampling distance, e.g. TMC-2).
    fine_resolution_m_per_px : float
        Resolution of fine_image in meters/pixel (must be > 0 and < coarse_resolution).
    coarse_image : np.ndarray
        Lower-resolution image array (larger ground sampling distance, e.g. IIRS).
    coarse_resolution_m_per_px : float
        Resolution of coarse_image in meters/pixel (must be > fine_resolution).
    transform_model : str
        Transformation model: ``'affine'`` (default).
    nscale : int, default 4
        Number of wavelet scales for phase congruency log-Gabor filters.
    norient : int, default 6
        Number of filter orientations for phase congruency.
    ratio_threshold : float, default 0.75
        Lowe ratio test threshold (0.0 to 1.0).
    reproj_threshold : float, default 3.0
        MAGSAC++ reprojection threshold in coarse pixel units.
    confidence : float, default 0.999
        Estimation confidence for robust fitting.

    Returns
    -------
    CrossModalRegistrationResult
        Detailed outcome with coarse-space and composed full-resolution matrices.

    Raises
    ------
    ValueError
        If resolutions are non-positive, if fine_resolution >= coarse_resolution,
        or if transform_model is not supported.
    """
    # 1. Validate resolutions
    if fine_resolution_m_per_px <= 0 or coarse_resolution_m_per_px <= 0:
        raise ValueError(
            f"Resolutions must be positive floats: fine={fine_resolution_m_per_px}, "
            f"coarse={coarse_resolution_m_per_px}."
        )
    if fine_resolution_m_per_px >= coarse_resolution_m_per_px:
        raise ValueError(
            f"fine_image must have strictly higher resolution (smaller m/px) than coarse_image. "
            f"Got fine_resolution_m_per_px={fine_resolution_m_per_px}, "
            f"coarse_resolution_m_per_px={coarse_resolution_m_per_px}. "
            f"Caller must provide the higher-resolution image as fine_image."
        )

    # 2. Validate transform model
    try:
        model_enum = TransformModel(transform_model)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported transform model: '{transform_model}'. Use 'affine'."
        ) from exc

    if model_enum != TransformModel.AFFINE:
        return CrossModalRegistrationResult(
            success=False,
            scale_ratio=compute_scale_ratio(fine_resolution_m_per_px, coarse_resolution_m_per_px),
            downsampled_fine_shape=(0, 0),
            transform_model=transform_model,
            failure_reason=(
                f"Unsupported transform model '{transform_model}' for cross-modal registration. "
                "Cross-modal registration currently only supports 'affine'."
            ),
        )

    # 3. Compute scale ratio and downsample fine image
    scale_ratio = compute_scale_ratio(fine_resolution_m_per_px, coarse_resolution_m_per_px)
    downsampled_fine = downsample_to_match(fine_image, scale_ratio)
    ds_shape = (downsampled_fine.shape[0], downsampled_fine.shape[1])

    # 4. Compute phase congruency maps on both images
    try:
        pc_fine, _ = compute_phase_congruency(downsampled_fine, nscale=nscale, norient=norient)
        pc_coarse, _ = compute_phase_congruency(coarse_image, nscale=nscale, norient=norient)
    except Exception as exc:
        return CrossModalRegistrationResult(
            success=False,
            scale_ratio=scale_ratio,
            downsampled_fine_shape=ds_shape,
            transform_model=transform_model,
            failure_reason=f"Phase congruency computation failed: {exc}",
        )

    # 5. Convert PC maps to uint8 representations for SIFT feature extraction
    u8_fine = _pc_to_uint8(pc_fine)
    u8_coarse = _pc_to_uint8(pc_coarse)

    ref_feat = _to_feature_img(u8_fine)
    tgt_feat = _to_feature_img(u8_coarse)

    # 6. Extract SIFT features on phase congruency maps
    try:
        ref_sift = extract_sift(ref_feat)
        tgt_sift = extract_sift(tgt_feat)
    except Exception as exc:
        return CrossModalRegistrationResult(
            success=False,
            scale_ratio=scale_ratio,
            downsampled_fine_shape=ds_shape,
            pc_map_fine=pc_fine,
            pc_map_coarse=pc_coarse,
            transform_model=transform_model,
            failure_reason=f"SIFT extraction on PC maps failed: {exc}",
        )

    if ref_sift.num_keypoints == 0 or tgt_sift.num_keypoints == 0:
        return CrossModalRegistrationResult(
            success=False,
            scale_ratio=scale_ratio,
            downsampled_fine_shape=ds_shape,
            pc_map_fine=pc_fine,
            pc_map_coarse=pc_coarse,
            transform_model=transform_model,
            failure_reason=(
                f"Insufficient SIFT features on PC maps: downsampled_fine={ref_sift.num_keypoints}, "
                f"coarse={tgt_sift.num_keypoints}. Need >= 1 in each image."
            ),
        )

    # 7. FLANN kNN Matching
    try:
        raw_matches = flann_knn_match(ref_sift, tgt_sift)
    except Exception as exc:
        return CrossModalRegistrationResult(
            success=False,
            scale_ratio=scale_ratio,
            downsampled_fine_shape=ds_shape,
            pc_map_fine=pc_fine,
            pc_map_coarse=pc_coarse,
            transform_model=transform_model,
            failure_reason=f"FLANN matching failed: {exc}",
        )

    # 8. Lowe ratio test
    try:
        match_result = apply_ratio_test(
            raw_matches, ref_sift, tgt_sift, ratio_threshold=ratio_threshold
        )
    except Exception as exc:
        return CrossModalRegistrationResult(
            success=False,
            scale_ratio=scale_ratio,
            downsampled_fine_shape=ds_shape,
            pc_map_fine=pc_fine,
            pc_map_coarse=pc_coarse,
            transform_model=transform_model,
            failure_reason=f"Ratio test failed: {exc}",
        )

    # 9. Robust geometric estimation
    try:
        geo_result = estimate_transform(
            match_result.matches,
            model=model_enum,
            reproj_threshold=reproj_threshold,
            confidence=confidence,
        )
    except Exception as exc:
        return CrossModalRegistrationResult(
            success=False,
            scale_ratio=scale_ratio,
            downsampled_fine_shape=ds_shape,
            pc_map_fine=pc_fine,
            pc_map_coarse=pc_coarse,
            total_correspondences=match_result.accepted,
            matches=match_result.matches,
            transform_model=transform_model,
            failure_reason=f"Geometric estimation error: {exc}",
        )

    if not geo_result.success or geo_result.transform_matrix is None:
        return CrossModalRegistrationResult(
            success=False,
            scale_ratio=scale_ratio,
            downsampled_fine_shape=ds_shape,
            pc_map_fine=pc_fine,
            pc_map_coarse=pc_coarse,
            total_correspondences=geo_result.total_correspondences,
            inlier_count=geo_result.inlier_count,
            matches=match_result.matches,
            inlier_mask=geo_result.inlier_mask,
            transform_model=transform_model,
            failure_reason=geo_result.failure_reason or "Geometric estimation failed.",
        )

    # 10. Compose full-resolution transform matrix
    m_coarse = geo_result.transform_matrix
    m_full = compose_scale_transform(m_coarse, scale_ratio, model_enum)

    # Extract real error metrics from geo_result.error_metrics (not fabricated)
    em = geo_result.error_metrics
    inlier_rmse = em.inlier_rmse if em else 0.0
    inlier_median = em.inlier_median_error if em else 0.0
    inlier_max = em.inlier_max_error if em else 0.0
    all_rmse = em.all_rmse if em else 0.0
    all_median = em.all_median_error if em else 0.0
    per_match = em.per_match_errors if em else None

    logger.info(
        "Cross-modal registration succeeded: scale=%.4f, inliers=%d/%d, RMSE=%.3f coarse px",
        scale_ratio,
        geo_result.inlier_count,
        geo_result.total_correspondences,
        inlier_rmse,
    )

    return CrossModalRegistrationResult(
        success=True,
        scale_ratio=scale_ratio,
        downsampled_fine_shape=ds_shape,
        pc_map_fine=pc_fine,
        pc_map_coarse=pc_coarse,
        coarse_transform_matrix=m_coarse,
        full_resolution_transform_matrix=m_full,
        inlier_count=geo_result.inlier_count,
        total_correspondences=geo_result.total_correspondences,
        inlier_rmse=inlier_rmse,
        inlier_median_error=inlier_median,
        inlier_max_error=inlier_max,
        all_rmse=all_rmse,
        all_median_error=all_median,
        per_match_errors=per_match,
        matches=match_result.matches,
        inlier_mask=geo_result.inlier_mask,
        transform_model=transform_model,
        failure_reason="",
    )
