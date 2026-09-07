"""
SIH26166 — Standalone Cross-Scale Registration Module.

Orchestrates multi-resolution cross-sensor image registration (e.g. matching
high-resolution OHRC at ~0.21–0.25 m/px to coarse-resolution TMC-2 at ~4.27–5.0 m/px)
by combining:
1. Anti-aliased Gaussian pyramid downsampling of the fine image to match coarse resolution.
2. Classical SIFT feature extraction, FLANN matching, Lowe ratio filtering, and robust
   geometric estimation (MAGSAC++ / RANSAC) between downsampled fine (reference)
   and coarse (target) images.
3. Analytical composition of the coarse-space transformation with the downsampling scale
   factor to recover the full-resolution transform matrix mapping original fine coordinates
   directly to coarse image coordinates.

Scope and Architectural Boundaries
----------------------------------
This module implements the **resolution-normalization and transform-composition**
foundational stage of cross-scale registration.  It does NOT implement:
- Multi-level hierarchical pyramid feature tracking / refinement
- Bounding-box / ROI backward projection and sub-pixel fine refinement on unscaled OHRC
- Crater detection / morphological anchoring
- Illumination invariant Phase Congruency or multimodal descriptor mapping (MIND/RIFT)

This module is currently standalone and unwired into the primary registration pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

import cv2
import numpy as np

from backend.features.models import FilteredMatch
from backend.features.sift import extract_sift
from backend.geometry.estimation import estimate_transform
from backend.geometry.models import GeometricResult, TransformModel
from backend.matching.flann import flann_knn_match
from backend.matching.ratio_test import apply_ratio_test
from backend.preprocessing.datamodel import FeatureImage, RawImage
from backend.preprocessing.grayscale import to_feature_image
from backend.preprocessing.scale_handler import compute_scale_ratio, downsample_to_match

logger = logging.getLogger("sih26166.core.cross_scale_registration")


@dataclass
class CrossScaleRegistrationResult:
    """Result of cross-scale image registration.

    Attributes
    ----------
    success : bool
        Whether geometric transformation estimation succeeded with sufficient inliers.
    scale_ratio : float
        Computed spatial resolution scale ratio (fine_res / coarse_res < 1.0).
    downsampled_fine_shape : tuple[int, int]
        Dimensions (height, width) of the fine image after anti-aliased downsampling.
    coarse_transform_matrix : np.ndarray | None
        Estimated transformation matrix mapping downsampled fine pixel coordinates
        to coarse image pixel coordinates (2x3 for affine, 3x3 for homography).
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
    transform_model : str
        Transformation model used: ``'affine'`` or ``'homography'``.
    failure_reason : str
        Description of failure mode if success is False, otherwise empty string.
    """

    success: bool
    scale_ratio: float
    downsampled_fine_shape: tuple[int, int]
    coarse_transform_matrix: np.ndarray | None = None
    full_resolution_transform_matrix: np.ndarray | None = None
    inlier_count: int = 0
    total_correspondences: int = 0
    inlier_rmse: float = 0.0
    transform_model: str = "affine"
    failure_reason: str = ""
    inlier_median_error: float = 0.0
    inlier_max_error: float = 0.0
    all_rmse: float = 0.0
    all_median_error: float = 0.0
    per_match_errors: np.ndarray | None = None
    matches: list[FilteredMatch] = field(default_factory=list)
    inlier_mask: np.ndarray | None = None


def compose_scale_transform(
    matrix: np.ndarray,
    scale_ratio: float,
    model: str | TransformModel = TransformModel.AFFINE,
) -> np.ndarray:
    """Compose coarse-space transformation matrix with downsampling scale.

    Parameters
    ----------
    matrix : np.ndarray
        Estimated transformation matrix mapping downsampled fine coordinates
        to coarse image coordinates (2x3 for affine, 3x3 for homography).
    scale_ratio : float
        Resolution ratio (fine_res / coarse_res), equivalent to the downsampling factor.
    model : str or TransformModel
        "affine" or "homography".

    Returns
    -------
    np.ndarray
        Composed transformation matrix mapping full-resolution fine image
        coordinates to coarse image coordinates (2x3 for affine, 3x3 for homography).

    Derivation
    ----------
    Let p_full = [x_full, y_full]^T be coordinates in the full-resolution fine image.
    Let p_ds = [x_ds, y_ds]^T be coordinates in the downsampled fine image.
    Since downsampling scales spatial coordinates from the origin by scale_ratio s:
        p_ds = s * p_full
    In homogeneous coordinates:
        [x_ds, y_ds, 1]^T = S * [x_full, y_full, 1]^T,
    where S is the 3x3 scaling matrix:
        S = [[s, 0, 0],
             [0, s, 0],
             [0, 0, 1]].

    The estimated transformation M_coarse maps p_ds -> p_coarse:
        p_coarse = M_coarse * [x_ds, y_ds, 1]^T.
    Substituting p_ds in terms of p_full:
        p_coarse = M_coarse * (S * [x_full, y_full, 1]^T)
                 = (M_coarse * S) * [x_full, y_full, 1]^T
                 = M_full * [x_full, y_full, 1]^T.

    For Affine (2x3 matrix M_coarse = [[a00, a01, tx], [a10, a11, ty]]):
        M_coarse_3x3 = [[a00, a01, tx],
                        [a10, a11, ty],
                        [  0,   0,  1]]
        M_full_3x3 = M_coarse_3x3 @ S
                   = [[a00*s, a01*s, tx],
                      [a10*s, a11*s, ty],
                      [    0,     0,  1]]
        Taking the top 2 rows yields the 2x3 affine matrix:
        M_full = [[a00*s, a01*s, tx],
                  [a10*s, a11*s, ty]].

    For Homography (3x3 matrix H_coarse):
        M_full = H_coarse @ S
               = [[h00*s, h01*s, h02],
                  [h10*s, h11*s, h12],
                  [h20*s, h21*s, h22]].
    """
    m = TransformModel(model) if isinstance(model, str) else model
    s = float(scale_ratio)

    S = np.array([
        [s, 0.0, 0.0],
        [0.0, s, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    if m == TransformModel.AFFINE:
        if matrix.shape == (2, 3):
            m_3x3 = np.vstack([matrix.astype(np.float64), [0.0, 0.0, 1.0]])
        elif matrix.shape == (3, 3):
            m_3x3 = matrix.astype(np.float64)
        else:
            raise ValueError(f"Expected 2x3 or 3x3 matrix for affine, got shape {matrix.shape}")
        m_full_3x3 = m_3x3 @ S
        return m_full_3x3[:2, :]
    elif m == TransformModel.HOMOGRAPHY:
        if matrix.shape != (3, 3):
            raise ValueError(f"Expected 3x3 matrix for homography, got shape {matrix.shape}")
        return (matrix.astype(np.float64) @ S)
    else:
        raise ValueError(f"Unsupported transform model: {m}")


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


def register_cross_scale(
    fine_image: np.ndarray,
    fine_resolution_m_per_px: float,
    coarse_image: np.ndarray,
    coarse_resolution_m_per_px: float,
    transform_model: str = "affine",
    ratio_threshold: float = 0.75,
    reproj_threshold: float = 3.0,
    confidence: float = 0.999,
) -> CrossScaleRegistrationResult:
    """Register a fine-resolution image against a coarse-resolution image.

    Parameters
    ----------
    fine_image : np.ndarray
        Higher-resolution image array (smaller ground sampling distance, e.g. OHRC).
    fine_resolution_m_per_px : float
        Resolution of fine_image in meters/pixel (must be > 0 and < coarse_resolution).
    coarse_image : np.ndarray
        Lower-resolution image array (larger ground sampling distance, e.g. TMC-2).
    coarse_resolution_m_per_px : float
        Resolution of coarse_image in meters/pixel (must be > fine_resolution).
    transform_model : str
        Transformation model: ``'affine'`` (default) or ``'homography'``.
    ratio_threshold : float
        Lowe ratio test threshold (default 0.75).
    reproj_threshold : float
        MAGSAC++ reprojection threshold in pixels (default 3.0).
    confidence : float
        Estimation confidence (default 0.999).

    Returns
    -------
    CrossScaleRegistrationResult
        Detailed registration outcome with coarse and composed full-resolution matrices.

    Raises
    ------
    ValueError
        If resolutions are non-positive or if fine_resolution >= coarse_resolution.
    """
    # 1. Validate resolutions
    if fine_resolution_m_per_px <= 0 or coarse_resolution_m_per_px <= 0:
        raise ValueError(
            f"Resolutions must be positive floats: fine={fine_resolution_m_per_px}, "
            f"coarse={coarse_resolution_m_per_px}."
        )
    if fine_resolution_m_per_px >= coarse_resolution_m_per_px:
        raise ValueError(
            f"fine_resolution_m_per_px ({fine_resolution_m_per_px}) must be strictly smaller "
            f"than coarse_resolution_m_per_px ({coarse_resolution_m_per_px}). "
            f"Caller must provide the higher-resolution image as fine_image."
        )

    # 2. Validate transform model
    try:
        model_enum = TransformModel(transform_model)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported transform model: '{transform_model}'. Use 'affine' or 'homography'."
        ) from exc

    # 3. Compute scale ratio and downsample fine image
    scale_ratio = compute_scale_ratio(fine_resolution_m_per_px, coarse_resolution_m_per_px)
    downsampled_fine = downsample_to_match(fine_image, scale_ratio)
    ds_shape = (downsampled_fine.shape[0], downsampled_fine.shape[1])

    # 4. Convert images to FeatureImage representation
    ref_feat = _to_feature_img(downsampled_fine)
    tgt_feat = _to_feature_img(coarse_image)

    # 5. Extract SIFT features
    try:
        ref_sift = extract_sift(ref_feat)
        tgt_sift = extract_sift(tgt_feat)
    except Exception as exc:
        return CrossScaleRegistrationResult(
            success=False,
            scale_ratio=scale_ratio,
            downsampled_fine_shape=ds_shape,
            transform_model=transform_model,
            failure_reason=f"SIFT extraction failed: {exc}",
        )

    if ref_sift.num_keypoints == 0 or tgt_sift.num_keypoints == 0:
        return CrossScaleRegistrationResult(
            success=False,
            scale_ratio=scale_ratio,
            downsampled_fine_shape=ds_shape,
            transform_model=transform_model,
            failure_reason=(
                f"Insufficient SIFT features: downsampled_fine={ref_sift.num_keypoints}, "
                f"coarse={tgt_sift.num_keypoints}. Need >= 1 in each image."
            ),
        )

    # 6. FLANN kNN Matching
    try:
        raw_matches = flann_knn_match(ref_sift, tgt_sift)
    except Exception as exc:
        return CrossScaleRegistrationResult(
            success=False,
            scale_ratio=scale_ratio,
            downsampled_fine_shape=ds_shape,
            transform_model=transform_model,
            failure_reason=f"FLANN matching failed: {exc}",
        )

    # 7. Lowe ratio test
    try:
        match_result = apply_ratio_test(
            raw_matches, ref_sift, tgt_sift, ratio_threshold=ratio_threshold
        )
    except Exception as exc:
        return CrossScaleRegistrationResult(
            success=False,
            scale_ratio=scale_ratio,
            downsampled_fine_shape=ds_shape,
            transform_model=transform_model,
            failure_reason=f"Ratio test failed: {exc}",
        )

    # 8. Robust geometric estimation
    try:
        geo_result = estimate_transform(
            match_result.matches,
            model=model_enum,
            reproj_threshold=reproj_threshold,
            confidence=confidence,
        )
    except Exception as exc:
        return CrossScaleRegistrationResult(
            success=False,
            scale_ratio=scale_ratio,
            downsampled_fine_shape=ds_shape,
            total_correspondences=match_result.accepted,
            transform_model=transform_model,
            failure_reason=f"Geometric estimation error: {exc}",
        )

    if not geo_result.success or geo_result.transform_matrix is None:
        return CrossScaleRegistrationResult(
            success=False,
            scale_ratio=scale_ratio,
            downsampled_fine_shape=ds_shape,
            total_correspondences=geo_result.total_correspondences,
            inlier_count=geo_result.inlier_count,
            transform_model=transform_model,
            failure_reason=geo_result.failure_reason or "Geometric estimation failed.",
        )

    # 9. Compose full-resolution transform matrix
    m_coarse = geo_result.transform_matrix
    m_full = compose_scale_transform(m_coarse, scale_ratio, model_enum)

    em = geo_result.error_metrics
    inlier_rmse = em.inlier_rmse if em else 0.0
    inlier_median = em.inlier_median_error if em else 0.0
    inlier_max = em.inlier_max_error if em else 0.0
    all_rmse = em.all_rmse if em else 0.0
    all_median = em.all_median_error if em else 0.0
    per_match = em.per_match_errors if em else None

    return CrossScaleRegistrationResult(
        success=True,
        scale_ratio=scale_ratio,
        downsampled_fine_shape=ds_shape,
        coarse_transform_matrix=m_coarse,
        full_resolution_transform_matrix=m_full,
        inlier_count=geo_result.inlier_count,
        total_correspondences=geo_result.total_correspondences,
        inlier_rmse=inlier_rmse,
        transform_model=transform_model,
        failure_reason="",
        inlier_median_error=inlier_median,
        inlier_max_error=inlier_max,
        all_rmse=all_rmse,
        all_median_error=all_median,
        per_match_errors=per_match,
        matches=match_result.matches if match_result else [],
        inlier_mask=geo_result.inlier_mask,
    )
