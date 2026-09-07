"""
SIH26166 — Extreme-Scale Composed Registration (OHRC ↔ IIRS via TMC-2 Bridge).

Implements OHRC↔IIRS registration by composing the existing OHRC↔TMC-2
(cross-scale, ~20× ratio) and TMC-2↔IIRS (cross-modal, ~19.4× ratio) pipelines.
The ~394× scale gap between OHRC (~0.25 m/px) and IIRS (~82.70 m/px) makes direct
feature matching unreliable; this module avoids that entirely by chaining two
well-characterized intermediate transforms through a TMC-2 bridge image.

Architecture:
    Step A:  OHRC → TMC-2   via register_cross_scale()   → M_A (affine 2×3)
    Step B:  TMC-2 → IIRS   via register_cross_modal()   → M_B (affine 2×3)
    Step C:  Compose M_A, M_B → M_composed = M_B @ M_A   (algebraic, no re-matching)

The composed transform M_composed maps full-resolution OHRC pixel coordinates
directly to IIRS pixel coordinates in a single affine warp.

SCOPE & LIMITATIONS:
- Requires a real TMC-2 bridge image that overlaps both the OHRC and IIRS scenes.
- Only affine composition is supported (consistent with both sub-pipelines).
- Validated on SYNTHETIC composed transforms with known ground-truth matrices ONLY.
- NOT validated on real overlapping OHRC/TMC-2/IIRS imagery (no such triple fixture
  currently exists in the project repository).
- Does NOT implement: SuperPoint/LightGlue, sub-pixel refinement, crater anchoring,
  MIND/RIFT, or any modification to the existing cross_scale or cross_modal modules.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from backend.core.cross_modal_registration import (
    CrossModalRegistrationResult,
    register_cross_modal,
)
from backend.core.cross_scale_registration import (
    CrossScaleRegistrationResult,
    register_cross_scale,
)

logger = logging.getLogger("sih26166.core.extreme_scale_registration")


# ===================================================================
# Data Models
# ===================================================================

@dataclass
class StageResult:
    """Summary of one stage's registration outcome (for the intermediate block)."""

    success: bool
    transform_matrix: np.ndarray | None
    scale_ratio: float
    inlier_count: int
    total_correspondences: int
    inlier_rmse: float
    failure_reason: str = ""


@dataclass
class IntermediateInfo:
    """Metadata about the two-stage composed registration.

    Provides transparency into each stage's transform and which TMC-2
    bridge image was used, enabling downstream audit and debugging.
    """

    stage_a_transform: np.ndarray | None  # OHRC → TMC-2 full-resolution affine (2×3)
    stage_b_transform: np.ndarray | None  # TMC-2 → IIRS full-resolution affine (2×3)
    composed_transform: np.ndarray | None  # OHRC → IIRS composed affine (2×3)
    tmc2_bridge_image_path: str  # Path to real TMC-2 used, or "synthetic"
    stage_a_result: StageResult
    stage_b_result: StageResult


@dataclass
class ExtremeScaleRegistrationResult:
    """Result of extreme-scale composed registration (OHRC ↔ IIRS via TMC-2).

    Follows the same result shape as other registration paths:
    correspondence, accuracy, spatial, transform, output, timing —
    plus an ``intermediate`` block showing both stage transforms and
    which TMC-2 image was used as the bridge.

    Attributes
    ----------
    success : bool
        Whether the composed registration succeeded end-to-end.
    composed_transform_matrix : np.ndarray | None
        The final OHRC → IIRS affine transformation matrix (2×3).
        Maps full-resolution OHRC pixel coordinates directly to IIRS pixel coordinates.
    total_scale_ratio : float
        Overall scale ratio (OHRC resolution / IIRS resolution).
    inlier_count_stage_a : int
        Inlier count from Stage A (OHRC → TMC-2).
    inlier_count_stage_b : int
        Inlier count from Stage B (TMC-2 → IIRS).
    inlier_rmse_stage_a : float
        Inlier RMSE from Stage A in TMC-2 pixel units.
    inlier_rmse_stage_b : float
        Inlier RMSE from Stage B in IIRS pixel units.
    transform_model : str
        Always "affine" (composition math is affine-only).
    failure_reason : str
        Description of failure mode if success is False.
    failure_stage : str
        Which stage failed: "stage_a", "stage_b", or "" on success.
    intermediate : IntermediateInfo | None
        Detailed intermediate results from both stages.
    timings : dict[str, float]
        Per-stage and total timing in seconds.
    """

    success: bool
    composed_transform_matrix: np.ndarray | None = None
    total_scale_ratio: float = 0.0
    inlier_count_stage_a: int = 0
    inlier_count_stage_b: int = 0
    inlier_rmse_stage_a: float = 0.0
    inlier_rmse_stage_b: float = 0.0
    transform_model: str = "affine"
    failure_reason: str = ""
    failure_stage: str = ""
    intermediate: IntermediateInfo | None = None
    timings: dict[str, float] = field(default_factory=dict)


# ===================================================================
# Transform Composition
# ===================================================================

def compose_affine_transforms(
    m_a: np.ndarray,
    m_b: np.ndarray,
) -> np.ndarray:
    """Compose two affine transforms: M_composed = M_B @ M_A.

    Given two 2×3 affine matrices:
        M_A maps coordinates from space A → space B  (e.g. OHRC → TMC-2)
        M_B maps coordinates from space B → space C  (e.g. TMC-2 → IIRS)

    The composed transform M_composed maps A → C directly:
        p_C = M_B(M_A(p_A))

    In homogeneous coordinates:
        [x_C]     [m_B00 m_B01 t_Bx] [m_A00 m_A01 t_Ax] [x_A]
        [y_C]  =  [m_B10 m_B11 t_By]·[m_A10 m_A11 t_Ay]·[y_A]
        [ 1 ]     [  0     0    1  ] [  0     0    1  ] [ 1 ]

    Parameters
    ----------
    m_a : np.ndarray
        First affine transform (2×3 or 3×3). Maps space A → space B.
    m_b : np.ndarray
        Second affine transform (2×3 or 3×3). Maps space B → space C.

    Returns
    -------
    np.ndarray
        Composed affine transform (2×3). Maps space A → space C.

    Raises
    ------
    ValueError
        If inputs are not 2×3 or 3×3 matrices.
    """
    # Lift to 3×3 homogeneous form
    a_3x3 = _to_3x3(m_a, "m_a")
    b_3x3 = _to_3x3(m_b, "m_b")

    # Compose: B applied after A
    composed_3x3 = b_3x3 @ a_3x3

    # Return 2×3 affine form (drop the [0, 0, 1] row)
    return composed_3x3[:2, :].astype(np.float64)


def _to_3x3(matrix: np.ndarray, name: str) -> np.ndarray:
    """Lift a 2×3 affine matrix to 3×3 homogeneous form."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape == (2, 3):
        return np.vstack([m, [0.0, 0.0, 1.0]])
    elif m.shape == (3, 3):
        return m.copy()
    else:
        raise ValueError(
            f"Expected 2×3 or 3×3 matrix for {name}, got shape {m.shape}"
        )


# ===================================================================
# Main Registration Orchestrator
# ===================================================================

def register_extreme_scale(
    ohrc_image: np.ndarray,
    ohrc_resolution_m_per_px: float,
    iirs_image: np.ndarray,
    iirs_resolution_m_per_px: float,
    tmc2_image: np.ndarray,
    tmc2_resolution_m_per_px: float,
    tmc2_bridge_path: str = "synthetic",
    *,
    transform_model: str = "affine",
    ratio_threshold: float = 0.75,
    reproj_threshold: float = 3.0,
    confidence: float = 0.999,
    nscale: int = 4,
    norient: int = 6,
) -> ExtremeScaleRegistrationResult:
    """Register OHRC against IIRS via composed TMC-2 bridge transforms.

    Parameters
    ----------
    ohrc_image : np.ndarray
        OHRC image array (highest resolution, ~0.25 m/px).
    ohrc_resolution_m_per_px : float
        OHRC spatial resolution in meters/pixel.
    iirs_image : np.ndarray
        IIRS image array (lowest resolution, ~82.70 m/px).
    iirs_resolution_m_per_px : float
        IIRS spatial resolution in meters/pixel.
    tmc2_image : np.ndarray
        TMC-2 bridge image array (intermediate resolution, ~5.0 m/px).
    tmc2_resolution_m_per_px : float
        TMC-2 spatial resolution in meters/pixel.
    tmc2_bridge_path : str
        Path to the TMC-2 bridge image file (for provenance), or "synthetic".
    transform_model : str
        Must be "affine" (only supported model for composition).
    ratio_threshold : float
        Lowe ratio test threshold for both stages.
    reproj_threshold : float
        MAGSAC++ reprojection threshold for both stages.
    confidence : float
        Estimation confidence for both stages.
    nscale : int
        Phase congruency wavelet scales (Stage B only).
    norient : int
        Phase congruency filter orientations (Stage B only).

    Returns
    -------
    ExtremeScaleRegistrationResult
        Complete result with composed transform and intermediate stage details.
    """
    timings: dict[str, float] = {}
    total_start = time.perf_counter()

    # 0. Validate: only affine composition is supported
    if transform_model != "affine":
        return ExtremeScaleRegistrationResult(
            success=False,
            failure_reason=(
                f"Unsupported transform model '{transform_model}' for extreme-scale "
                "composed registration. Only 'affine' is supported (composition math "
                "is affine-only, consistent with both sub-pipelines)."
            ),
            failure_stage="configuration",
            timings=timings,
        )

    total_scale_ratio = ohrc_resolution_m_per_px / iirs_resolution_m_per_px

    # ===================================================================
    # Stage A: OHRC → TMC-2 (cross-scale registration)
    # ===================================================================
    logger.info(
        "Stage A: OHRC → TMC-2 cross-scale registration "
        "(OHRC %.3f m/px → TMC-2 %.3f m/px, ~%.1f× ratio)",
        ohrc_resolution_m_per_px,
        tmc2_resolution_m_per_px,
        ohrc_resolution_m_per_px / tmc2_resolution_m_per_px,
    )

    t0 = time.perf_counter()
    try:
        stage_a_result = register_cross_scale(
            fine_image=ohrc_image,
            fine_resolution_m_per_px=ohrc_resolution_m_per_px,
            coarse_image=tmc2_image,
            coarse_resolution_m_per_px=tmc2_resolution_m_per_px,
            transform_model="affine",
            ratio_threshold=ratio_threshold,
            reproj_threshold=reproj_threshold,
            confidence=confidence,
        )
    except Exception as exc:
        timings["stage_a"] = time.perf_counter() - t0
        timings["total"] = time.perf_counter() - total_start
        return ExtremeScaleRegistrationResult(
            success=False,
            total_scale_ratio=total_scale_ratio,
            failure_reason=f"Stage A (OHRC→TMC-2) raised exception: {exc}",
            failure_stage="stage_a",
            timings=timings,
        )
    timings["stage_a"] = time.perf_counter() - t0

    if not stage_a_result.success:
        timings["total"] = time.perf_counter() - total_start
        stage_a_summary = StageResult(
            success=False,
            transform_matrix=None,
            scale_ratio=stage_a_result.scale_ratio,
            inlier_count=stage_a_result.inlier_count,
            total_correspondences=stage_a_result.total_correspondences,
            inlier_rmse=stage_a_result.inlier_rmse,
            failure_reason=stage_a_result.failure_reason,
        )
        return ExtremeScaleRegistrationResult(
            success=False,
            total_scale_ratio=total_scale_ratio,
            failure_reason=f"Stage A (OHRC→TMC-2) failed: {stage_a_result.failure_reason}",
            failure_stage="stage_a",
            intermediate=IntermediateInfo(
                stage_a_transform=None,
                stage_b_transform=None,
                composed_transform=None,
                tmc2_bridge_image_path=tmc2_bridge_path,
                stage_a_result=stage_a_summary,
                stage_b_result=StageResult(
                    success=False,
                    transform_matrix=None,
                    scale_ratio=0.0,
                    inlier_count=0,
                    total_correspondences=0,
                    inlier_rmse=0.0,
                    failure_reason="Not attempted (Stage A failed).",
                ),
            ),
            timings=timings,
        )

    m_a = stage_a_result.full_resolution_transform_matrix
    logger.info(
        "Stage A succeeded: %d/%d inliers, RMSE=%.3f TMC-2 px",
        stage_a_result.inlier_count,
        stage_a_result.total_correspondences,
        stage_a_result.inlier_rmse,
    )

    # ===================================================================
    # Stage B: TMC-2 → IIRS (cross-modal registration)
    # ===================================================================
    logger.info(
        "Stage B: TMC-2 → IIRS cross-modal registration "
        "(TMC-2 %.3f m/px → IIRS %.3f m/px, ~%.1f× ratio)",
        tmc2_resolution_m_per_px,
        iirs_resolution_m_per_px,
        tmc2_resolution_m_per_px / iirs_resolution_m_per_px,
    )

    t0 = time.perf_counter()
    try:
        stage_b_result = register_cross_modal(
            fine_image=tmc2_image,
            fine_resolution_m_per_px=tmc2_resolution_m_per_px,
            coarse_image=iirs_image,
            coarse_resolution_m_per_px=iirs_resolution_m_per_px,
            transform_model="affine",
            nscale=nscale,
            norient=norient,
            ratio_threshold=ratio_threshold,
            reproj_threshold=reproj_threshold,
            confidence=confidence,
        )
    except Exception as exc:
        timings["stage_b"] = time.perf_counter() - t0
        timings["total"] = time.perf_counter() - total_start
        stage_a_summary = StageResult(
            success=True,
            transform_matrix=m_a,
            scale_ratio=stage_a_result.scale_ratio,
            inlier_count=stage_a_result.inlier_count,
            total_correspondences=stage_a_result.total_correspondences,
            inlier_rmse=stage_a_result.inlier_rmse,
        )
        return ExtremeScaleRegistrationResult(
            success=False,
            total_scale_ratio=total_scale_ratio,
            failure_reason=f"Stage B (TMC-2→IIRS) raised exception: {exc}",
            failure_stage="stage_b",
            intermediate=IntermediateInfo(
                stage_a_transform=m_a,
                stage_b_transform=None,
                composed_transform=None,
                tmc2_bridge_image_path=tmc2_bridge_path,
                stage_a_result=stage_a_summary,
                stage_b_result=StageResult(
                    success=False,
                    transform_matrix=None,
                    scale_ratio=0.0,
                    inlier_count=0,
                    total_correspondences=0,
                    inlier_rmse=0.0,
                    failure_reason=f"Exception: {exc}",
                ),
            ),
            timings=timings,
        )
    timings["stage_b"] = time.perf_counter() - t0

    if not stage_b_result.success:
        timings["total"] = time.perf_counter() - total_start
        stage_a_summary = StageResult(
            success=True,
            transform_matrix=m_a,
            scale_ratio=stage_a_result.scale_ratio,
            inlier_count=stage_a_result.inlier_count,
            total_correspondences=stage_a_result.total_correspondences,
            inlier_rmse=stage_a_result.inlier_rmse,
        )
        stage_b_summary = StageResult(
            success=False,
            transform_matrix=None,
            scale_ratio=stage_b_result.scale_ratio,
            inlier_count=stage_b_result.inlier_count,
            total_correspondences=stage_b_result.total_correspondences,
            inlier_rmse=stage_b_result.inlier_rmse,
            failure_reason=stage_b_result.failure_reason,
        )
        return ExtremeScaleRegistrationResult(
            success=False,
            total_scale_ratio=total_scale_ratio,
            inlier_count_stage_a=stage_a_result.inlier_count,
            inlier_rmse_stage_a=stage_a_result.inlier_rmse,
            failure_reason=f"Stage B (TMC-2→IIRS) failed: {stage_b_result.failure_reason}",
            failure_stage="stage_b",
            intermediate=IntermediateInfo(
                stage_a_transform=m_a,
                stage_b_transform=None,
                composed_transform=None,
                tmc2_bridge_image_path=tmc2_bridge_path,
                stage_a_result=stage_a_summary,
                stage_b_result=stage_b_summary,
            ),
            timings=timings,
        )

    m_b = stage_b_result.full_resolution_transform_matrix
    logger.info(
        "Stage B succeeded: %d/%d inliers, RMSE=%.3f IIRS px",
        stage_b_result.inlier_count,
        stage_b_result.total_correspondences,
        stage_b_result.inlier_rmse,
    )

    # ===================================================================
    # Stage C: Compose M_A and M_B → M_composed (algebraic, no re-matching)
    # ===================================================================
    t0 = time.perf_counter()
    try:
        m_composed = compose_affine_transforms(m_a, m_b)
    except Exception as exc:
        timings["composition"] = time.perf_counter() - t0
        timings["total"] = time.perf_counter() - total_start
        return ExtremeScaleRegistrationResult(
            success=False,
            total_scale_ratio=total_scale_ratio,
            failure_reason=f"Transform composition failed: {exc}",
            failure_stage="composition",
            timings=timings,
        )
    timings["composition"] = time.perf_counter() - t0

    logger.info(
        "Stage C: composed OHRC→IIRS transform (algebraic composition, no re-matching)"
    )

    # ===================================================================
    # Build complete result
    # ===================================================================
    timings["total"] = time.perf_counter() - total_start

    stage_a_summary = StageResult(
        success=True,
        transform_matrix=m_a,
        scale_ratio=stage_a_result.scale_ratio,
        inlier_count=stage_a_result.inlier_count,
        total_correspondences=stage_a_result.total_correspondences,
        inlier_rmse=stage_a_result.inlier_rmse,
    )
    stage_b_summary = StageResult(
        success=True,
        transform_matrix=m_b,
        scale_ratio=stage_b_result.scale_ratio,
        inlier_count=stage_b_result.inlier_count,
        total_correspondences=stage_b_result.total_correspondences,
        inlier_rmse=stage_b_result.inlier_rmse,
    )

    intermediate = IntermediateInfo(
        stage_a_transform=m_a,
        stage_b_transform=m_b,
        composed_transform=m_composed,
        tmc2_bridge_image_path=tmc2_bridge_path,
        stage_a_result=stage_a_summary,
        stage_b_result=stage_b_summary,
    )

    logger.info(
        "Extreme-scale composed registration complete: "
        "OHRC(%.3f m/px) → TMC-2(%.3f m/px) → IIRS(%.3f m/px), "
        "total scale ratio=%.1f×, total=%.3f s",
        ohrc_resolution_m_per_px,
        tmc2_resolution_m_per_px,
        iirs_resolution_m_per_px,
        1.0 / total_scale_ratio if total_scale_ratio > 0 else 0.0,
        timings["total"],
    )

    return ExtremeScaleRegistrationResult(
        success=True,
        composed_transform_matrix=m_composed,
        total_scale_ratio=total_scale_ratio,
        inlier_count_stage_a=stage_a_result.inlier_count,
        inlier_count_stage_b=stage_b_result.inlier_count,
        inlier_rmse_stage_a=stage_a_result.inlier_rmse,
        inlier_rmse_stage_b=stage_b_result.inlier_rmse,
        transform_model="affine",
        failure_reason="",
        failure_stage="",
        intermediate=intermediate,
        timings=timings,
    )
