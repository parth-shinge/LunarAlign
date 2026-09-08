"""
SIH26166 — Classical registration pipeline orchestration service.

Thin orchestrator that composes existing modules into the complete
classical same-modality registration pipeline:

    load images
    → preprocess (grayscale conversion)
    → SIFT feature extraction
    → FLANN kNN matching
    → Lowe ratio test
    → MAGSAC++ geometric estimation
    → image warping (registration)
    → evaluation metrics + spatial distribution
    → diagnostic visualizations

This module contains NO algorithmic logic.  Every processing step
delegates to the corresponding backend module.

Pipeline mode: ``classical_sift``

Transform direction:
    - Source image: reference
    - Matrix maps: reference → target
    - Output: reference warped into target coordinate frame
    - No inversion is performed
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.core.cross_modal_registration import register_cross_modal
from backend.core.cross_scale_registration import register_cross_scale
from backend.core.extreme_scale_registration import register_extreme_scale
from backend.evaluation.metrics import MatchQualitySummary, build_quality_summary
from backend.evaluation.spatial import compute_spatial_distribution, SpatialDistribution
from backend.evaluation.visualization import (
    draw_filtered_matches,
    draw_inlier_outlier_matches,
    draw_registration_overlay,
)
from backend.features.sift import extract_sift
from backend.geometry.estimation import estimate_transform
from backend.geometry.models import ErrorMetrics, EstimatorMethod, GeometricResult, TransformModel
from backend.matching.flann import flann_knn_match
from backend.matching.ratio_test import apply_ratio_test
from backend.preprocessing.datamodel import FeatureImage, RawImage
from backend.preprocessing.grayscale import to_feature_image
from backend.preprocessing.io import load_image
from backend.preprocessing.pds4 import extract_pds4_metadata, is_pds4_label
from backend.preprocessing.scale_handler import get_resolution
from backend.registration.models import WarpConfig
from backend.registration.warping import warp_image
from backend.routing.pair_classifier import PairType, classify_pair

logger = logging.getLogger("sih26166.core.registration_service")

PIPELINE_MODE = "classical_sift"


# ===================================================================
# Result container
# ===================================================================

@dataclass
class RegistrationPipelineResult:
    """Complete result from the classical registration pipeline.

    Holds all artifacts and metrics needed by the API layer.
    """

    result_id: str
    success: bool
    pipeline_mode: str

    # Metrics
    quality_summary: MatchQualitySummary | None = None
    spatial: SpatialDistribution | None = None

    # Images (NumPy arrays, not saved to disk yet)
    registered_image: np.ndarray | None = None
    match_visualization: np.ndarray | None = None
    inlier_visualization: np.ndarray | None = None
    overlay_visualization: np.ndarray | None = None

    # Timing (seconds)
    timings: dict[str, float] = field(default_factory=dict)

    # Failure
    failure_reason: str = ""
    failure_stage: str = ""

    # Routing / Classification metadata
    pair_type: str = ""

    # Additive metadata for cross-scale registration
    scale_ratio: float | None = None


# ===================================================================
# Helpers
# ===================================================================

def _extract_resolution(path: str | Path, instrument: str) -> float:
    """Extract resolution using PDS4 metadata when available, falling back gracefully."""
    p = Path(path)
    meta_dict = None
    if is_pds4_label(p):
        try:
            pds_meta = extract_pds4_metadata(p)
            meta_dict = {"isda_product_params": pds_meta.isda_product_params}
        except Exception as exc:
            logger.warning("Could not extract PDS4 metadata from %s: %s", p, exc)
    return get_resolution(meta_dict, instrument)


def _run_cross_scale_registration(
    ref_path: str | Path,
    tgt_path: str | Path,
    *,
    classification: Any,
    transform_model: str,
    ratio_threshold: float,
    reproj_threshold: float,
    confidence: float,
    result_id: str,
    total_start: float,
) -> RegistrationPipelineResult:
    """Execute cross-scale registration for OHRC<->TMC-2 pairs.

    Resolution-normalized downsampling is performed on the fine (OHRC) image,
    matched against the coarse (TMC-2) image, and the resulting affine transform
    is composed to map full-resolution fine coordinates to coarse coordinates.
    The original fine image is then warped into the coarse frame.
    All error metrics and RMSE are reported in coarse-image pixel units.
    """
    pipeline_mode = classification.recommended_pipeline_stage
    timings: dict[str, float] = {}

    # 1. Transform model check: cross-scale currently only supports affine
    if transform_model != "affine":
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=pipeline_mode,
            failure_reason=(
                f"Unsupported transform model '{transform_model}' for cross-scale registration. "
                "Cross-scale registration currently only supports affine "
                "(composition math has only been proven for affine)."
            ),
            failure_stage="configuration",
            pair_type=classification.pair_type.value,
        )

    # 2. Load images
    t0 = time.perf_counter()
    try:
        ref_raw = load_image(ref_path)
        tgt_raw = load_image(tgt_path)
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=pipeline_mode,
            failure_reason=f"Failed to load images: {exc}",
            failure_stage="load",
            pair_type=classification.pair_type.value,
        )
    timings["load"] = time.perf_counter() - t0

    # 3. Identify fine (OHRC) vs coarse (TMC-2) inputs using classification
    if classification.instrument_a == "OHRC":
        fine_raw = ref_raw
        coarse_raw = tgt_raw
        fine_path = ref_path
        coarse_path = tgt_path
        fine_inst = "OHRC"
        coarse_inst = "TMC2"
    else:
        fine_raw = tgt_raw
        coarse_raw = ref_raw
        fine_path = tgt_path
        coarse_path = ref_path
        fine_inst = "OHRC"
        coarse_inst = "TMC2"

    fine_res = _extract_resolution(fine_path, fine_inst)
    coarse_res = _extract_resolution(coarse_path, coarse_inst)

    # 4. Execute standalone cross-scale registration
    t0 = time.perf_counter()
    try:
        cross_res = register_cross_scale(
            fine_image=fine_raw.data,
            fine_resolution_m_per_px=fine_res,
            coarse_image=coarse_raw.data,
            coarse_resolution_m_per_px=coarse_res,
            transform_model="affine",
            ratio_threshold=ratio_threshold,
            reproj_threshold=reproj_threshold,
            confidence=confidence,
        )
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=pipeline_mode,
            failure_reason=f"Cross-scale registration failed: {exc}",
            failure_stage="cross_scale_registration",
            pair_type=classification.pair_type.value,
            timings=timings,
        )
    timings["cross_scale_registration"] = time.perf_counter() - t0

    if not cross_res.success:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=pipeline_mode,
            failure_reason=f"Cross-scale registration failed: {cross_res.failure_reason}",
            failure_stage="cross_scale_registration",
            pair_type=classification.pair_type.value,
            scale_ratio=cross_res.scale_ratio,
            timings=timings,
        )

    # 5. Warp original full-resolution fine image into coarse target frame
    t0 = time.perf_counter()
    try:
        reg_result = warp_image(
            fine_raw.data,
            cross_res.full_resolution_transform_matrix,
            TransformModel.AFFINE,
            target_width=coarse_raw.width,
            target_height=coarse_raw.height,
        )
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=pipeline_mode,
            failure_reason=f"Warp failed: {exc}",
            failure_stage="warp",
            pair_type=classification.pair_type.value,
            scale_ratio=cross_res.scale_ratio,
            timings=timings,
        )
    timings["warp"] = time.perf_counter() - t0

    if not reg_result.success:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=pipeline_mode,
            failure_reason=f"Warp failed: {reg_result.failure_reason}",
            failure_stage="warp",
            pair_type=classification.pair_type.value,
            scale_ratio=cross_res.scale_ratio,
            timings=timings,
        )

    # 6. Evaluation metrics using existing evaluation modules
    t0 = time.perf_counter()
    spatial = compute_spatial_distribution(
        matches=cross_res.matches,
        inlier_mask=cross_res.inlier_mask,
        image_width=coarse_raw.width,
        image_height=coarse_raw.height,
    )

    em = ErrorMetrics(
        per_match_errors=cross_res.per_match_errors
        if cross_res.per_match_errors is not None else np.array([], dtype=np.float64),
        inlier_rmse=cross_res.inlier_rmse,
        all_rmse=cross_res.all_rmse,
        inlier_median_error=cross_res.inlier_median_error,
        inlier_max_error=cross_res.inlier_max_error,
        all_median_error=cross_res.all_median_error,
    )

    geo_res = GeometricResult(
        success=cross_res.success,
        transform_matrix=cross_res.full_resolution_transform_matrix,
        transform_model=TransformModel.AFFINE,
        estimator_method=EstimatorMethod.MAGSAC,
        inlier_mask=cross_res.inlier_mask,
        inlier_count=cross_res.inlier_count,
        outlier_count=max(0, cross_res.total_correspondences - cross_res.inlier_count),
        total_correspondences=cross_res.total_correspondences,
        inlier_ratio=(cross_res.inlier_count / cross_res.total_correspondences)
        if cross_res.total_correspondences > 0 else 0.0,
        error_metrics=em,
        failure_reason="",
    )

    summary = build_quality_summary(
        geo_res,
        spatial_entropy=spatial.normalized_entropy,
    )
    timings["evaluation"] = time.perf_counter() - t0

    # 7. Visualizations
    t0 = time.perf_counter()
    coarse_display = _to_display(coarse_raw.data)
    overlay_vis = draw_registration_overlay(
        coarse_display, _to_display(reg_result.registered_image),
    )
    timings["visualization"] = time.perf_counter() - t0

    # 8. Total timing and log
    timings["total"] = time.perf_counter() - total_start

    logger.info(
        "Cross-scale registration complete: id=%s, inliers=%d/%d, RMSE=%.3f coarse px, total=%.3f s",
        result_id,
        summary.inlier_count,
        summary.total_correspondences,
        summary.inlier_rmse,
        timings["total"],
    )

    return RegistrationPipelineResult(
        result_id=result_id,
        success=True,
        pipeline_mode=pipeline_mode,
        quality_summary=summary,
        spatial=spatial,
        registered_image=reg_result.registered_image,
        overlay_visualization=overlay_vis,
        timings=timings,
        pair_type=classification.pair_type.value,
        scale_ratio=cross_res.scale_ratio,
    )


def _run_cross_modal_registration(
    ref_path: str | Path,
    tgt_path: str | Path,
    *,
    classification: Any,
    transform_model: str,
    ratio_threshold: float,
    reproj_threshold: float,
    confidence: float,
    result_id: str,
    total_start: float,
) -> RegistrationPipelineResult:
    """Execute cross-modal registration for TMC-2<->IIRS pairs.

    Resolution-normalized downsampling is performed on the fine (TMC-2) image,
    phase congruency maps are extracted for contrast/illumination invariance,
    matched against the coarse (IIRS) image, and the resulting affine transform
    is composed to map full-resolution fine coordinates to coarse coordinates.
    The original fine image is then warped into the coarse frame.
    All error metrics and RMSE are reported in coarse-image pixel units.
    """
    pipeline_mode = classification.recommended_pipeline_stage
    timings: dict[str, float] = {}

    # 1. Transform model check: cross-modal currently only supports affine
    if transform_model != "affine":
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=pipeline_mode,
            failure_reason=(
                f"Unsupported transform model '{transform_model}' for cross-modal registration. "
                "Cross-modal registration currently only supports affine "
                "(composition math has only been proven for affine)."
            ),
            failure_stage="configuration",
            pair_type=classification.pair_type.value,
        )

    # 2. Load images
    t0 = time.perf_counter()
    try:
        ref_raw = load_image(ref_path)
        tgt_raw = load_image(tgt_path)
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=pipeline_mode,
            failure_reason=f"Failed to load images: {exc}",
            failure_stage="load",
            pair_type=classification.pair_type.value,
        )
    timings["load"] = time.perf_counter() - t0

    # 3. Identify fine (TMC-2) vs coarse (IIRS) inputs using classification
    if classification.instrument_a == "TMC2":
        fine_raw = ref_raw
        coarse_raw = tgt_raw
        fine_path = ref_path
        coarse_path = tgt_path
        fine_inst = "TMC2"
        coarse_inst = "IIRS"
    else:
        fine_raw = tgt_raw
        coarse_raw = ref_raw
        fine_path = tgt_path
        coarse_path = ref_path
        fine_inst = "TMC2"
        coarse_inst = "IIRS"

    fine_res = _extract_resolution(fine_path, fine_inst)
    coarse_res = _extract_resolution(coarse_path, coarse_inst)

    # 4. Execute standalone cross-modal registration
    t0 = time.perf_counter()
    try:
        cross_res = register_cross_modal(
            fine_image=fine_raw.data,
            fine_resolution_m_per_px=fine_res,
            coarse_image=coarse_raw.data,
            coarse_resolution_m_per_px=coarse_res,
            transform_model="affine",
            ratio_threshold=ratio_threshold,
            reproj_threshold=reproj_threshold,
            confidence=confidence,
        )
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=pipeline_mode,
            failure_reason=f"Cross-modal registration failed: {exc}",
            failure_stage="cross_modal_registration",
            pair_type=classification.pair_type.value,
            timings=timings,
        )
    timings["cross_modal_registration"] = time.perf_counter() - t0

    if not cross_res.success:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=pipeline_mode,
            failure_reason=f"Cross-modal registration failed: {cross_res.failure_reason}",
            failure_stage="cross_modal_registration",
            pair_type=classification.pair_type.value,
            scale_ratio=cross_res.scale_ratio,
            timings=timings,
        )

    # 5. Warp original full-resolution fine image into coarse target frame
    t0 = time.perf_counter()
    try:
        reg_result = warp_image(
            fine_raw.data,
            cross_res.full_resolution_transform_matrix,
            TransformModel.AFFINE,
            target_width=coarse_raw.width,
            target_height=coarse_raw.height,
        )
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=pipeline_mode,
            failure_reason=f"Warp failed: {exc}",
            failure_stage="warp",
            pair_type=classification.pair_type.value,
            scale_ratio=cross_res.scale_ratio,
            timings=timings,
        )
    timings["warp"] = time.perf_counter() - t0

    if not reg_result.success:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=pipeline_mode,
            failure_reason=f"Warp failed: {reg_result.failure_reason}",
            failure_stage="warp",
            pair_type=classification.pair_type.value,
            scale_ratio=cross_res.scale_ratio,
            timings=timings,
        )

    # 6. Evaluation metrics using existing evaluation modules
    t0 = time.perf_counter()
    spatial = compute_spatial_distribution(
        matches=cross_res.matches,
        inlier_mask=cross_res.inlier_mask,
        image_width=coarse_raw.width,
        image_height=coarse_raw.height,
    )

    em = ErrorMetrics(
        per_match_errors=cross_res.per_match_errors
        if cross_res.per_match_errors is not None else np.array([], dtype=np.float64),
        inlier_rmse=cross_res.inlier_rmse,
        all_rmse=cross_res.all_rmse,
        inlier_median_error=cross_res.inlier_median_error,
        inlier_max_error=cross_res.inlier_max_error,
        all_median_error=cross_res.all_median_error,
    )

    geo_res = GeometricResult(
        success=cross_res.success,
        transform_matrix=cross_res.full_resolution_transform_matrix,
        transform_model=TransformModel.AFFINE,
        estimator_method=EstimatorMethod.MAGSAC,
        inlier_mask=cross_res.inlier_mask,
        inlier_count=cross_res.inlier_count,
        outlier_count=max(0, cross_res.total_correspondences - cross_res.inlier_count),
        total_correspondences=cross_res.total_correspondences,
        inlier_ratio=(cross_res.inlier_count / cross_res.total_correspondences)
        if cross_res.total_correspondences > 0 else 0.0,
        error_metrics=em,
        failure_reason="",
    )

    summary = build_quality_summary(
        geo_res,
        spatial_entropy=spatial.normalized_entropy,
    )
    timings["evaluation"] = time.perf_counter() - t0

    # 7. Visualizations
    t0 = time.perf_counter()
    coarse_display = _to_display(coarse_raw.data)
    overlay_vis = draw_registration_overlay(
        coarse_display, _to_display(reg_result.registered_image),
    )
    timings["visualization"] = time.perf_counter() - t0

    # 8. Total timing and log
    timings["total"] = time.perf_counter() - total_start

    logger.info(
        "Cross-modal registration complete: id=%s, inliers=%d/%d, RMSE=%.3f coarse px, total=%.3f s",
        result_id,
        summary.inlier_count,
        summary.total_correspondences,
        summary.inlier_rmse,
        timings["total"],
    )

    return RegistrationPipelineResult(
        result_id=result_id,
        success=True,
        pipeline_mode=pipeline_mode,
        quality_summary=summary,
        spatial=spatial,
        registered_image=reg_result.registered_image,
        overlay_visualization=overlay_vis,
        timings=timings,
        pair_type=classification.pair_type.value,
        scale_ratio=cross_res.scale_ratio,
    )


def _run_extreme_scale_registration(
    ref_path: str | Path,
    tgt_path: str | Path,
    *,
    classification: Any,
    tmc2_bridge_path: str | Path,
    transform_model: str,
    ratio_threshold: float,
    reproj_threshold: float,
    confidence: float,
    result_id: str,
    total_start: float,
) -> RegistrationPipelineResult:
    """Execute extreme-scale composed registration for OHRC<->IIRS pairs via TMC-2 bridge.

    Calls register_extreme_scale() which internally chains:
      Stage A: OHRC -> TMC-2 (cross-scale, ~20x)
      Stage B: TMC-2 -> IIRS (cross-modal, ~19.4x)
      Stage C: Algebraic composition M_composed = M_B @ M_A
    """
    pipeline_mode = classification.recommended_pipeline_stage
    timings: dict[str, float] = {}

    # 1. Transform model check
    if transform_model != "affine":
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=pipeline_mode,
            failure_reason=(
                f"Unsupported transform model '{transform_model}' for extreme-scale registration. "
                "Extreme-scale composed registration only supports affine "
                "(composition math is affine-only)."
            ),
            failure_stage="configuration",
            pair_type=classification.pair_type.value,
        )

    # 2. Load all three images: OHRC, TMC-2 bridge, IIRS
    t0 = time.perf_counter()
    try:
        ref_raw = load_image(ref_path)
        tgt_raw = load_image(tgt_path)
        tmc2_raw = load_image(tmc2_bridge_path)
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=pipeline_mode,
            failure_reason=f"Failed to load images: {exc}",
            failure_stage="load",
            pair_type=classification.pair_type.value,
        )
    timings["load"] = time.perf_counter() - t0

    # 3. Identify OHRC vs IIRS using classification instrument ordering
    if classification.instrument_a == "OHRC":
        ohrc_raw, iirs_raw = ref_raw, tgt_raw
        ohrc_path, iirs_path = ref_path, tgt_path
    else:
        ohrc_raw, iirs_raw = tgt_raw, ref_raw
        ohrc_path, iirs_path = tgt_path, ref_path

    ohrc_res = _extract_resolution(ohrc_path, "OHRC")
    iirs_res = _extract_resolution(iirs_path, "IIRS")
    tmc2_res = _extract_resolution(tmc2_bridge_path, "TMC2")

    # 4. Execute extreme-scale composed registration
    t0 = time.perf_counter()
    try:
        extreme_res = register_extreme_scale(
            ohrc_image=ohrc_raw.data,
            ohrc_resolution_m_per_px=ohrc_res,
            iirs_image=iirs_raw.data,
            iirs_resolution_m_per_px=iirs_res,
            tmc2_image=tmc2_raw.data,
            tmc2_resolution_m_per_px=tmc2_res,
            tmc2_bridge_path=str(tmc2_bridge_path),
            transform_model="affine",
            ratio_threshold=ratio_threshold,
            reproj_threshold=reproj_threshold,
            confidence=confidence,
        )
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=pipeline_mode,
            failure_reason=f"Extreme-scale composed registration failed: {exc}",
            failure_stage="extreme_scale_registration",
            pair_type=classification.pair_type.value,
            timings=timings,
        )
    timings["extreme_scale_registration"] = time.perf_counter() - t0
    timings.update(extreme_res.timings)

    if not extreme_res.success:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=pipeline_mode,
            failure_reason=(
                f"Extreme-scale composed registration failed at {extreme_res.failure_stage}: "
                f"{extreme_res.failure_reason}"
            ),
            failure_stage="extreme_scale_registration",
            pair_type=classification.pair_type.value,
            scale_ratio=extreme_res.total_scale_ratio,
            timings=timings,
        )

    # 5. Warp OHRC into IIRS frame using composed transform
    t0 = time.perf_counter()
    try:
        reg_result = warp_image(
            ohrc_raw.data,
            extreme_res.composed_transform_matrix,
            TransformModel.AFFINE,
            target_width=iirs_raw.width,
            target_height=iirs_raw.height,
        )
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=pipeline_mode,
            failure_reason=f"Warp failed: {exc}",
            failure_stage="warp",
            pair_type=classification.pair_type.value,
            scale_ratio=extreme_res.total_scale_ratio,
            timings=timings,
        )
    timings["warp"] = time.perf_counter() - t0

    if not reg_result.success:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=pipeline_mode,
            failure_reason=f"Warp failed: {reg_result.failure_reason}",
            failure_stage="warp",
            pair_type=classification.pair_type.value,
            scale_ratio=extreme_res.total_scale_ratio,
            timings=timings,
        )

    # 6. Build quality summary from Stage B metrics (IIRS-space)
    # (Direct correspondences do not exist for composed registration;
    #  correspondences exist only in intermediate Stage A and Stage B spaces)
    t0 = time.perf_counter()
    stage_b_total = (
        extreme_res.intermediate.stage_b_result.total_correspondences
        if extreme_res.intermediate and extreme_res.intermediate.stage_b_result
        else extreme_res.inlier_count_stage_b
    )
    inlier_count = extreme_res.inlier_count_stage_b
    outlier_count = max(0, stage_b_total - inlier_count)
    inlier_ratio = (inlier_count / stage_b_total) if stage_b_total > 0 else 0.0
    composed_mat = (
        extreme_res.composed_transform_matrix.tolist()
        if extreme_res.composed_transform_matrix is not None
        else None
    )

    summary = MatchQualitySummary(
        success=True,
        total_correspondences=stage_b_total,
        inlier_count=inlier_count,
        outlier_count=outlier_count,
        inlier_ratio=inlier_ratio,
        inlier_rmse=extreme_res.inlier_rmse_stage_b,
        all_rmse=extreme_res.inlier_rmse_stage_b,
        inlier_median_error=0.0,
        inlier_max_error=0.0,
        spatial_entropy=-1.0,  # Not computed for composed registration
        transform_model="affine",
        estimator_method="MAGSAC_composed",
        transform_matrix=composed_mat,
        failure_reason="",
        quality_grade="not_computed_for_composed_registration",
    )
    timings["evaluation"] = time.perf_counter() - t0

    # 7. Visualization
    t0 = time.perf_counter()
    iirs_display = _to_display(iirs_raw.data)
    overlay_vis = draw_registration_overlay(
        iirs_display, _to_display(reg_result.registered_image),
    )
    timings["visualization"] = time.perf_counter() - t0

    # 8. Total timing
    timings["total"] = time.perf_counter() - total_start

    logger.info(
        "Extreme-scale composed registration complete: id=%s, "
        "stage_a inliers=%d (RMSE=%.3f TMC-2 px), "
        "stage_b inliers=%d (RMSE=%.3f IIRS px), total=%.3f s",
        result_id,
        extreme_res.inlier_count_stage_a,
        extreme_res.inlier_rmse_stage_a,
        extreme_res.inlier_count_stage_b,
        extreme_res.inlier_rmse_stage_b,
        timings["total"],
    )

    return RegistrationPipelineResult(
        result_id=result_id,
        success=True,
        pipeline_mode=pipeline_mode,
        quality_summary=summary,
        registered_image=reg_result.registered_image,
        overlay_visualization=overlay_vis,
        timings=timings,
        pair_type=classification.pair_type.value,
        scale_ratio=extreme_res.total_scale_ratio,
    )


# ===================================================================
# Pipeline orchestrator
# ===================================================================

def run_classical_registration(
    ref_path: str | Path,
    tgt_path: str | Path,
    *,
    transform_model: str = "affine",
    ratio_threshold: float = 0.75,
    reproj_threshold: float = 3.0,
    confidence: float = 0.999,
    tmc2_bridge_path: str | Path | None = None,
) -> RegistrationPipelineResult:
    """Execute the full classical registration pipeline.

    Parameters
    ----------
    ref_path : str | Path
        Path to the reference image on disk.
    tgt_path : str | Path
        Path to the target image on disk.
    transform_model : str
        ``'affine'`` or ``'homography'``.
    ratio_threshold : float
        Lowe ratio test threshold (0–1).
    reproj_threshold : float
        MAGSAC++ reprojection threshold in pixels.
    confidence : float
        Estimation confidence (0–1).
    tmc2_bridge_path : str | Path | None
        Path to TMC-2 bridge image for extreme-scale OHRC<->IIRS registration.
        Required when pair classification yields CROSS_MODAL_EXTREME_SCALE_OHRC_IIRS.
        Ignored for all other pair types.

    Returns
    -------
    RegistrationPipelineResult
        Complete pipeline result with metrics and image artifacts.
    """
    result_id = uuid.uuid4().hex[:16]
    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    # --- Validate transform model ---
    try:
        model = TransformModel(transform_model)
    except ValueError:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=f"Unsupported transform model: '{transform_model}'. Use 'affine' or 'homography'.",
            failure_stage="configuration",
        )

    # --- 0. Pair classification pre-flight guard ---
    classification = classify_pair(ref_path, tgt_path)
    if not classification.is_implemented:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=f"Unsupported pair type '{classification.pair_type.value}': {classification.reason}",
            failure_stage="pair_classification",
            pair_type=classification.pair_type.value,
        )

    # --- 0b. Dedicated cross-scale branch for OHRC <-> TMC-2 ---
    if classification.pair_type == PairType.CROSS_SCALE_OHRC_TMC2 and classification.is_implemented:
        return _run_cross_scale_registration(
            ref_path=ref_path,
            tgt_path=tgt_path,
            classification=classification,
            transform_model=transform_model,
            ratio_threshold=ratio_threshold,
            reproj_threshold=reproj_threshold,
            confidence=confidence,
            result_id=result_id,
            total_start=total_start,
        )

    # --- 0c. Dedicated cross-modal branch for TMC-2 <-> IIRS ---
    if classification.pair_type == PairType.CROSS_MODAL_TMC2_IIRS and classification.is_implemented:
        return _run_cross_modal_registration(
            ref_path=ref_path,
            tgt_path=tgt_path,
            classification=classification,
            transform_model=transform_model,
            ratio_threshold=ratio_threshold,
            reproj_threshold=reproj_threshold,
            confidence=confidence,
            result_id=result_id,
            total_start=total_start,
        )

    # --- 0d. Dedicated extreme-scale branch for OHRC <-> IIRS via TMC-2 bridge ---
    if classification.pair_type == PairType.CROSS_MODAL_EXTREME_SCALE_OHRC_IIRS and classification.is_implemented:
        if tmc2_bridge_path is None:
            return RegistrationPipelineResult(
                result_id=result_id,
                success=False,
                pipeline_mode=classification.recommended_pipeline_stage,
                failure_reason=(
                    "Extreme-scale OHRC<->IIRS registration requires a TMC-2 bridge image "
                    "(tmc2_bridge_path argument). No bridge image was provided."
                ),
                failure_stage="configuration",
                pair_type=classification.pair_type.value,
            )
        return _run_extreme_scale_registration(
            ref_path=ref_path,
            tgt_path=tgt_path,
            classification=classification,
            tmc2_bridge_path=tmc2_bridge_path,
            transform_model=transform_model,
            ratio_threshold=ratio_threshold,
            reproj_threshold=reproj_threshold,
            confidence=confidence,
            result_id=result_id,
            total_start=total_start,
        )

    # --- 1. Load images ---
    t0 = time.perf_counter()
    try:
        ref_raw = load_image(ref_path)
        tgt_raw = load_image(tgt_path)
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=f"Failed to load images: {exc}",
            failure_stage="load",
        )
    timings["load"] = time.perf_counter() - t0

    # --- 2. Preprocess (grayscale conversion) ---
    t0 = time.perf_counter()
    try:
        ref_feat = to_feature_image(ref_raw)
        tgt_feat = to_feature_image(tgt_raw)
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=f"Preprocessing failed: {exc}",
            failure_stage="preprocess",
        )
    timings["preprocess"] = time.perf_counter() - t0

    # --- 3. SIFT feature extraction ---
    t0 = time.perf_counter()
    try:
        ref_sift = extract_sift(ref_feat)
        tgt_sift = extract_sift(tgt_feat)
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=f"SIFT extraction failed: {exc}",
            failure_stage="sift",
        )
    timings["sift"] = time.perf_counter() - t0

    if ref_sift.num_keypoints == 0 or tgt_sift.num_keypoints == 0:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=(
                f"Insufficient SIFT features: reference={ref_sift.num_keypoints}, "
                f"target={tgt_sift.num_keypoints}. Need ≥1 keypoint in each image."
            ),
            failure_stage="sift",
            timings=timings,
        )

    # --- 4. FLANN matching ---
    t0 = time.perf_counter()
    try:
        raw_matches = flann_knn_match(ref_sift, tgt_sift)
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=f"FLANN matching failed: {exc}",
            failure_stage="matching",
            timings=timings,
        )
    timings["matching"] = time.perf_counter() - t0

    # --- 5. Lowe ratio test ---
    t0 = time.perf_counter()
    try:
        match_result = apply_ratio_test(
            raw_matches, ref_sift, tgt_sift,
            ratio_threshold=ratio_threshold,
        )
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=f"Ratio test failed: {exc}",
            failure_stage="ratio_test",
            timings=timings,
        )
    timings["ratio_test"] = time.perf_counter() - t0

    if match_result.accepted == 0:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=(
                f"No matches passed the Lowe ratio test "
                f"(threshold={ratio_threshold})."
            ),
            failure_stage="ratio_test",
            timings=timings,
        )

    # --- 6. Geometric estimation (MAGSAC++) ---
    t0 = time.perf_counter()
    try:
        geo_result = estimate_transform(
            match_result.matches,
            model=model,
            reproj_threshold=reproj_threshold,
            confidence=confidence,
        )
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=f"Geometric estimation failed: {exc}",
            failure_stage="geometry",
            timings=timings,
        )
    timings["geometry"] = time.perf_counter() - t0

    if not geo_result.success:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=f"Geometric estimation failed: {geo_result.failure_reason}",
            failure_stage="geometry",
            timings=timings,
        )

    # --- 7. Image warping ---
    t0 = time.perf_counter()
    try:
        reg_result = warp_image(
            ref_raw.data,
            geo_result.transform_matrix,
            geo_result.transform_model,
            target_width=tgt_raw.width,
            target_height=tgt_raw.height,
        )
    except Exception as exc:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=f"Warp failed: {exc}",
            failure_stage="warp",
            timings=timings,
        )
    timings["warp"] = time.perf_counter() - t0

    if not reg_result.success:
        return RegistrationPipelineResult(
            result_id=result_id,
            success=False,
            pipeline_mode=PIPELINE_MODE,
            failure_reason=f"Warp failed: {reg_result.failure_reason}",
            failure_stage="warp",
            timings=timings,
        )

    # --- 8. Evaluation ---
    t0 = time.perf_counter()

    # Spatial distribution
    spatial = compute_spatial_distribution(
        match_result.matches,
        geo_result.inlier_mask,
        tgt_raw.width,
        tgt_raw.height,
    )

    # Quality summary
    summary = build_quality_summary(
        geo_result,
        spatial_entropy=spatial.normalized_entropy,
    )

    timings["evaluation"] = time.perf_counter() - t0

    # --- 9. Visualizations ---
    t0 = time.perf_counter()

    # Prepare display images (uint8)
    ref_display = _to_display(ref_raw.data)
    tgt_display = _to_display(tgt_raw.data)

    match_vis = draw_filtered_matches(ref_display, tgt_display, match_result.matches)
    inlier_vis = draw_inlier_outlier_matches(
        ref_display, tgt_display, match_result.matches, geo_result.inlier_mask,
    )
    overlay_vis = draw_registration_overlay(
        tgt_display, _to_display(reg_result.registered_image),
    )

    timings["visualization"] = time.perf_counter() - t0

    # --- Total ---
    timings["total"] = time.perf_counter() - total_start

    logger.info(
        "Registration complete: id=%s, inliers=%d/%d, RMSE=%.3f px, total=%.3f s",
        result_id,
        summary.inlier_count,
        summary.total_correspondences,
        summary.inlier_rmse,
        timings["total"],
    )

    return RegistrationPipelineResult(
        result_id=result_id,
        success=True,
        pipeline_mode=PIPELINE_MODE,
        quality_summary=summary,
        spatial=spatial,
        registered_image=reg_result.registered_image,
        match_visualization=match_vis,
        inlier_visualization=inlier_vis,
        overlay_visualization=overlay_vis,
        timings=timings,
        pair_type=classification.pair_type.value,
    )


def _to_display(img: np.ndarray) -> np.ndarray:
    """Convert image to uint8 for visualization."""
    if img is None:
        return np.zeros((1, 1), dtype=np.uint8)
    if img.dtype in (np.float32, np.float64):
        return np.clip(img, 0, 255).astype(np.uint8)
    if img.dtype == np.uint16:
        return (img / 256).astype(np.uint8)
    return img.astype(np.uint8) if img.dtype != np.uint8 else img
