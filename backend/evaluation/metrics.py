"""
SIH26166 — Structured evaluation metrics.

Consumes ``GeometricResult`` and ``RegistrationResult`` from the
geometry and registration layers to produce a compact, serializable
evaluation summary.

This module **reuses** the error metrics already computed by
``backend.geometry.estimation`` rather than re-computing them.
No competing RMSE calculations exist.

The ``MatchQualitySummary`` is the primary evaluation output,
suitable for later API serialization or frontend display.

This module is part of the **CLASSICAL REGISTRATION BASELINE**.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.geometry.models import (
    ErrorMetrics,
    EstimatorMethod,
    GeometricResult,
    TransformModel,
)

logger = logging.getLogger("sih26166.evaluation.metrics")


@dataclass
class MatchQualitySummary:
    """Compact, serializable summary of registration quality.

    All fields are plain Python types (no NumPy arrays) so this
    structure can be directly serialized to JSON.

    Attributes
    ----------
    success : bool
        Whether geometric estimation succeeded.
    total_correspondences : int
        Total filtered matches (pre-geometry).
    inlier_count : int
        Geometrically consistent correspondences.
    outlier_count : int
        Rejected correspondences.
    inlier_ratio : float
        ``inlier_count / total_correspondences``.
    inlier_rmse : float
        RMSE over inlier correspondences (primary accuracy metric).
    all_rmse : float
        RMSE over all correspondences.
    inlier_median_error : float
        Median reprojection error among inliers.
    inlier_max_error : float
        Maximum reprojection error among inliers.
    spatial_entropy : float
        Normalized spatial entropy of inlier distribution.
        0.0 = highly concentrated, 1.0 = perfectly uniform.
        -1.0 if not computed.
    transform_model : str
        Transformation model used (e.g. 'affine', 'homography').
    estimator_method : str
        Robust estimator actually used (e.g. 'USAC_MAGSAC', 'RANSAC').
    transform_matrix : list[list[float]] | None
        Transformation matrix as nested lists (JSON-serializable).
    failure_reason : str
        Explanation if estimation failed.
    quality_grade : str
        Quality classification or diagnostic flag (e.g.
        'not_computed_for_composed_registration').
    """

    success: bool
    total_correspondences: int
    inlier_count: int
    outlier_count: int
    inlier_ratio: float
    inlier_rmse: float
    all_rmse: float
    inlier_median_error: float
    inlier_max_error: float
    spatial_entropy: float
    transform_model: str
    estimator_method: str
    transform_matrix: list[list[float]] | None
    failure_reason: str = ""
    quality_grade: str = ""


def build_quality_summary(
    geo_result: GeometricResult,
    *,
    spatial_entropy: float = -1.0,
) -> MatchQualitySummary:
    """Build a ``MatchQualitySummary`` from a ``GeometricResult``.

    This reuses the error metrics already computed during geometric
    estimation — no duplicate RMSE calculation is performed.

    Parameters
    ----------
    geo_result : GeometricResult
        Result from ``geometry.estimation.estimate_transform()``.
    spatial_entropy : float
        Pre-computed spatial entropy.  -1.0 if not yet computed.

    Returns
    -------
    MatchQualitySummary
        Compact evaluation summary.
    """
    # Extract error metrics (may be None on failure)
    em = geo_result.error_metrics
    inlier_rmse = em.inlier_rmse if em is not None else 0.0
    all_rmse = em.all_rmse if em is not None else 0.0
    inlier_median = em.inlier_median_error if em is not None else 0.0
    inlier_max = em.inlier_max_error if em is not None else 0.0

    # Transform matrix → JSON-serializable nested list
    mat_list = None
    if geo_result.transform_matrix is not None:
        mat_list = geo_result.transform_matrix.tolist()

    # Estimator method string
    method_str = (
        geo_result.estimator_method.value
        if geo_result.estimator_method is not None
        else "none"
    )

    return MatchQualitySummary(
        success=geo_result.success,
        total_correspondences=geo_result.total_correspondences,
        inlier_count=geo_result.inlier_count,
        outlier_count=geo_result.outlier_count,
        inlier_ratio=geo_result.inlier_ratio,
        inlier_rmse=inlier_rmse,
        all_rmse=all_rmse,
        inlier_median_error=inlier_median,
        inlier_max_error=inlier_max,
        spatial_entropy=spatial_entropy,
        transform_model=geo_result.transform_model.value,
        estimator_method=method_str,
        transform_matrix=mat_list,
        failure_reason=geo_result.failure_reason,
    )
