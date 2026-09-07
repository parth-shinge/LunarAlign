"""
SIH26166 — Tests for Standalone Cross-Modal Registration (TMC-2 <-> IIRS).

Two independent tiers:
- TIER A: Synthetic rich lunar scene with KNOWN ground-truth affine transformation
  AND a NONLINEAR intensity transform (contrast/intensity inversion) to simulate
  a genuine cross-modal appearance difference (visible vs infrared). Demonstrates
  that phase congruency recovers ground-truth coordinates where raw SIFT fails.
- TIER B: Real Chandrayaan-2 PDS4 fixture rasters (TMC-2 at 4.27 m/px and IIRS at
  82.70 m/px). Sanity-only test demonstrating end-to-end pipeline completion
  on real pixel data at confirmed resolutions without crashing or NaN.
- Edge case tests: parameter validation, resolution ordering, blank image handling.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.core.cross_modal_registration import (
    CrossModalRegistrationResult,
    _pc_to_uint8,
    register_cross_modal,
)
from backend.core.cross_scale_registration import _to_feature_img
from backend.features.sift import extract_sift
from backend.geometry.estimation import estimate_transform
from backend.matching.flann import flann_knn_match
from backend.matching.ratio_test import apply_ratio_test
from backend.preprocessing.io import load_image
from backend.preprocessing.scale_handler import compute_scale_ratio, downsample_to_match

PDS4_FIXTURES = Path("tests/fixtures/pds4")
TMC2_FIXTURE_XML = PDS4_FIXTURES / "tmc2" / "ch2_tmc_sample_500x500.xml"
IIRS_FIXTURE_XML = PDS4_FIXTURES / "iirs" / "ch2_iirs_sample_200x200x16.xml"


# ---------------------------------------------------------------------------
# Synthetic Scene Generator (Deterministic multi-scale crater scene)
# ---------------------------------------------------------------------------

def _make_rich_scene(width: int, height: int, scale: float) -> np.ndarray:
    """Generate a deterministic synthetic scene with multi-scale crater-like features.

    All feature coordinates and radii are multiplied by `scale` so that
    downsampling an image generated at scale=20.0 produces the same visual
    content as an image generated directly at scale=1.0.
    """
    img = np.zeros((height, width), dtype=np.uint8)

    # Smooth background texture gradient
    for y in range(height):
        img[y, :] = int(30 + 100 * y / max(height - 1, 1))

    # Grid of crater structures (rings, rims, central peaks)
    for i in range(50):
        cx = int((30 + (i % 8) * 45) * scale)
        cy = int((30 + (i // 8) * 45) * scale)
        r = int((8 + (i % 5) * 4) * scale)
        v1 = int(180 + (i % 4) * 20)
        v2 = int(40 + (i % 3) * 30)

        # Outer rim
        cv2.circle(img, (cx, cy), r, v1, max(1, int(2 * scale)))
        # Crater floor
        cv2.circle(img, (cx, cy), max(1, r - int(3 * scale)), v2, -1)
        # Central peak
        if r > 10 * scale:
            cv2.circle(img, (cx + int(scale), cy), int(2 * scale), 230, -1)

    # Cross markers for sharp edge/corner detection
    for i in range(20):
        x = int((20 + (i * 17) % 260) * scale)
        y = int((20 + (i * 23) % 260) * scale)
        cv2.drawMarker(
            img,
            (x, y),
            255,
            markerType=cv2.MARKER_CROSS,
            markerSize=int(10 * scale),
            thickness=max(1, int(scale)),
        )

    return img


# =========================================================================
# TIER A: Synthetic Known Ground Truth & Cross-Modal Invariance Proof
# =========================================================================

class TestTierASyntheticGroundTruth:
    """Rigorous verification of cross-modal registration against known ground truth."""

    def test_cross_modal_20x_synthetic_intensity_inversion(self):
        """Verify ~20x cross-scale registration under nonlinear intensity inversion.

        Setup:
        - Coarse resolution: 5.0 m/px (target base 400x400)
        - Fine resolution: 0.25 m/px (source base 8000x8000, 20.0x scale ratio)
        - Known affine transformation applied to coarse target:
          rotation angle = 3.0 deg around center (200, 200),
          scale factor = 1.0,
          translation tx = +10.0 px, ty = +6.0 px.
        - Ground truth matrix M_full_true = M_coarse_true @ S, mapping
          full-resolution fine points (x_full, y_full) -> coarse points (x_coarse, y_coarse).
        - Modality simulation: NONLINEAR intensity inversion (255 - target) applied
          to coarse target to simulate visible-to-thermal IR contrast inversion.

        Assertions:
        - res.success is True
        - inlier_count >= 50 (demonstrates robust correspondence recovery)
        - M_full maps known full-resolution fine points to coarse points within
          an explicitly stated tolerance of 1.0 coarse pixels across the entire
          8000x8000 domain.
        """
        coarse_base = _make_rich_scene(400, 400, scale=1.0)
        fine_image = _make_rich_scene(8000, 8000, scale=20.0)

        # 1. Define known ground-truth transformation in coarse pixel space
        center = (200.0, 200.0)
        angle_deg = 3.0
        scale = 1.0
        tx = 10.0
        ty = 6.0

        M_coarse_true = cv2.getRotationMatrix2D(center, angle_deg, scale)
        M_coarse_true[0, 2] += tx
        M_coarse_true[1, 2] += ty

        # Generate target coarse image by applying known transform
        coarse_target = cv2.warpAffine(
            coarse_base, M_coarse_true, (400, 400), borderMode=cv2.BORDER_REFLECT
        )

        # Apply genuine nonlinear cross-modal transform: contrast inversion
        coarse_target_inverted = (255 - coarse_target).astype(np.uint8)

        # 2. Execute cross-modal registration
        res = register_cross_modal(
            fine_image=fine_image,
            fine_resolution_m_per_px=0.25,
            coarse_image=coarse_target_inverted,
            coarse_resolution_m_per_px=5.0,
            transform_model="affine",
        )

        # 3. Assertions on pipeline completion and inlier count
        assert res.success is True, f"Registration failed: {res.failure_reason}"
        assert res.inlier_count >= 50, (
            f"Expected >= 50 inliers for structured synthetic scene under inversion, "
            f"got {res.inlier_count} / {res.total_correspondences}"
        )
        assert res.downsampled_fine_shape == (400, 400)
        assert res.scale_ratio == 0.05
        assert res.inlier_rmse < 1.0, f"Inlier RMSE too high: {res.inlier_rmse:.3f} px"
        assert res.inlier_median_error < 0.6, f"Inlier median error too high: {res.inlier_median_error:.3f} px"
        assert res.inlier_max_error < 3.0, f"Inlier max error too high: {res.inlier_max_error:.3f} px"

        # 4. Rigorous ground truth verification: evaluate M_full
        s = 0.05
        S = np.array([
            [s, 0.0, 0.0],
            [0.0, s, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        M_coarse_3x3 = np.vstack([M_coarse_true, [0.0, 0.0, 1.0]])
        M_full_true = (M_coarse_3x3 @ S)[:2, :]

        # Test points across full-resolution fine image: corners + center
        test_pts_fine = np.array([
            [0.0, 0.0],
            [8000.0, 0.0],
            [0.0, 8000.0],
            [8000.0, 8000.0],
            [4000.0, 4000.0],
        ], dtype=np.float64)

        pts_homo = np.hstack([test_pts_fine, np.ones((len(test_pts_fine), 1))])
        expected_coarse = (M_full_true @ pts_homo.T).T
        actual_coarse = (res.full_resolution_transform_matrix @ pts_homo.T).T
        pt_errors = np.sqrt(np.sum((actual_coarse - expected_coarse) ** 2, axis=1))

        # Print detailed numerical recovery report
        print("\n" + "=" * 70)
        print("TIER A CROSS-MODAL REGISTRATION (TMC-2 <-> IIRS SYNTHETIC INVERSION)")
        print("=" * 70)
        print(f"{'Parameter':<25} | {'True':<12} | {'Recovered':<12} | {'Error':<12}")
        print("-" * 70)

        # Decompose affine matrix to extract rotation, scale, translations
        M_rec = res.coarse_transform_matrix
        a00, a01, tx_rec = M_rec[0, 0], M_rec[0, 1], M_rec[0, 2]
        a10, a11, ty_rec = M_rec[1, 0], M_rec[1, 1], M_rec[1, 2]

        angle_rec = -float(np.degrees(np.arctan2(a10, a00)))
        scale_x_rec = float(np.sqrt(a00**2 + a10**2))
        scale_y_rec = float(np.sqrt(a01**2 + a11**2))

        tx_true = float(M_coarse_true[0, 2])
        ty_true = float(M_coarse_true[1, 2])

        print(f"{'Rotation Angle (deg)':<25} | {angle_deg:<12.4f} | {angle_rec:<12.4f} | {abs(angle_rec - angle_deg):<12.4f}")
        print(f"{'Scale X':<25} | {scale:<12.4f} | {scale_x_rec:<12.4f} | {abs(scale_x_rec - scale):<12.4f}")
        print(f"{'Scale Y':<25} | {scale:<12.4f} | {scale_y_rec:<12.4f} | {abs(scale_y_rec - scale):<12.4f}")
        print(f"{'Matrix Trans X (px)':<25} | {tx_true:<12.4f} | {tx_rec:<12.4f} | {abs(tx_rec - tx_true):<12.4f}")
        print(f"{'Matrix Trans Y (px)':<25} | {ty_true:<12.4f} | {ty_rec:<12.4f} | {abs(ty_rec - ty_true):<12.4f}")
        print("-" * 70)
        print(f"Inliers: {res.inlier_count} / {res.total_correspondences} (RMSE: {res.inlier_rmse:.3f} px)")
        print(f"Max point reprojection error across 8000x8000 image: {pt_errors.max():.4f} coarse px")
        print("=" * 70)

        # Stated tolerance assertion: all corners and center within 1.0 coarse pixel
        for i, (name, err) in enumerate(zip(["(0,0)", "(8000,0)", "(0,8000)", "(8000,8000)", "(4000,4000)"], pt_errors)):
            assert err < 1.0, (
                f"Point {name} error {err:.4f} px exceeds strict tolerance of 1.0 coarse pixels"
            )

    def test_raw_sift_fails_under_intensity_inversion(self):
        """Demonstrate that raw SIFT fails on the same cross-modal inverted pair.

        This directly validates the necessity of phase congruency for cross-modal
        matching: under identical ~20x downsampling and intensity inversion, raw
        SIFT gradients invert/decorrelate, causing Lowe ratio test and geometric
        estimation to fail with insufficient inliers.
        """
        coarse_base = _make_rich_scene(400, 400, scale=1.0)
        fine_image = _make_rich_scene(8000, 8000, scale=20.0)

        center = (200.0, 200.0)
        angle_deg = 3.0
        M_coarse_true = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        M_coarse_true[0, 2] += 10.0
        M_coarse_true[1, 2] += 6.0

        coarse_target = cv2.warpAffine(coarse_base, M_coarse_true, (400, 400), borderMode=cv2.BORDER_REFLECT)
        coarse_target_inverted = (255 - coarse_target).astype(np.uint8)

        # Downsample fine image as usual
        s = compute_scale_ratio(0.25, 5.0)
        downsampled_fine = downsample_to_match(fine_image, s)

        # Run RAW SIFT on inverted images without phase congruency
        ref_feat = _to_feature_img(downsampled_fine)
        tgt_feat = _to_feature_img(coarse_target_inverted)

        ref_sift = extract_sift(ref_feat)
        tgt_sift = extract_sift(tgt_feat)
        raw_matches = flann_knn_match(ref_sift, tgt_sift)
        match_result = apply_ratio_test(raw_matches, ref_sift, tgt_sift)
        geo_result = estimate_transform(match_result.matches)

        print("\n" + "-" * 70)
        print("RAW SIFT VS PHASE CONGRUENCY COMPARISON (UNDER INTENSITY INVERSION)")
        print("-" * 70)
        print(f"Raw SIFT accepted correspondences: {len(match_result.matches)}")
        print(f"Raw SIFT inliers: {geo_result.inlier_count} (success={geo_result.success})")
        print("-" * 70)

        # Raw SIFT cannot cope with contrast inversion and yields insufficient matches / fails
        assert geo_result.success is False or geo_result.inlier_count < 10, (
            f"Expected raw SIFT to fail or produce < 10 inliers on inverted scene, got {geo_result.inlier_count}"
        )


# =========================================================================
# TIER B: Real Fixture Data Pipeline Completion (Sanity Only)
# =========================================================================

class TestTierBRealFixtureSanity:
    """Sanity-only verification using real Chandrayaan-2 PDS4 rasters.

    IMPORTANT NOTE:
    TMC-2 and IIRS fixture samples in this repository cover different lunar sites
    (non-overlapping), so there is no real ground truth or spatial correspondence
    between them. This test strictly verifies that:
    1. The real PDS4 XML rasters load and downsample at confirmed real resolutions
       (TMC-2 = 4.27 m/px, IIRS = 82.70 m/px).
    2. Phase congruency executes cleanly on real lunar pixel distributions without
       raising exceptions, producing NaNs, or overflowing uint8.
    3. The pipeline runs to completion and produces structured finite outputs.
    """

    def test_real_tmc2_iirs_fixtures_pipeline_completion(self):
        """Verify pipeline execution on real TMC-2 (4.27m/px) and IIRS (82.70m/px) fixtures."""
        tmc_raw = load_image(TMC2_FIXTURE_XML)
        iirs_raw = load_image(IIRS_FIXTURE_XML)

        assert tmc_raw.data.ndim == 2
        assert iirs_raw.data.ndim == 2

        res_tmc = 4.27
        res_iirs = 82.70

        result = register_cross_modal(
            fine_image=tmc_raw.data,
            fine_resolution_m_per_px=res_tmc,
            coarse_image=iirs_raw.data,
            coarse_resolution_m_per_px=res_iirs,
            transform_model="affine",
        )

        assert isinstance(result, CrossModalRegistrationResult)
        assert result.scale_ratio == pytest.approx(4.27 / 82.70, rel=1e-3)
        assert result.downsampled_fine_shape == (26, 26)
        assert result.pc_map_fine is not None
        assert result.pc_map_coarse is not None
        assert result.pc_map_fine.shape == (26, 26)
        assert result.pc_map_coarse.shape == (200, 200)
        assert np.all(np.isfinite(result.pc_map_fine))
        assert np.all(np.isfinite(result.pc_map_coarse))

        # Because the two fixtures do not overlap, 0 inliers and success=False is expected
        assert result.success is False
        assert result.inlier_count == 0
        assert "failure_reason" in result.__dict__
        assert len(result.failure_reason) > 0


# =========================================================================
# Unit Tests: Edge Cases & Adapters
# =========================================================================

class TestCrossModalEdgeCases:
    """Validation of parameter checking, resolution ordering, and adapters."""

    def test_pc_to_uint8_conversion(self):
        """_pc_to_uint8 converts [0, 1] float32 array to valid [0, 255] uint8."""
        arr = np.linspace(0.0, 1.0, 100, dtype=np.float32).reshape(10, 10)
        u8 = _pc_to_uint8(arr)
        assert u8.dtype == np.uint8
        assert u8.shape == (10, 10)
        assert u8.min() == 0
        assert u8.max() == 255

    def test_pc_to_uint8_handles_nans_and_constants(self):
        """_pc_to_uint8 safely handles constant or NaN-infused arrays."""
        const = np.full((10, 10), 0.5, dtype=np.float32)
        const[0, 0] = np.nan
        u8 = _pc_to_uint8(const)
        assert u8.dtype == np.uint8
        assert np.all(np.isfinite(u8))

    def test_reversed_resolutions_raises_value_error(self):
        """Passing fine_resolution >= coarse_resolution raises ValueError."""
        img = np.zeros((50, 50), dtype=np.uint8)
        with pytest.raises(ValueError, match="strictly higher resolution"):
            register_cross_modal(
                fine_image=img,
                fine_resolution_m_per_px=82.70,
                coarse_image=img,
                coarse_resolution_m_per_px=4.27,
            )

    def test_equal_resolutions_raises_value_error(self):
        """Passing equal resolutions raises ValueError."""
        img = np.zeros((50, 50), dtype=np.uint8)
        with pytest.raises(ValueError, match="strictly higher resolution"):
            register_cross_modal(
                fine_image=img,
                fine_resolution_m_per_px=5.0,
                coarse_image=img,
                coarse_resolution_m_per_px=5.0,
            )

    def test_non_positive_resolution_raises_value_error(self):
        """Passing resolution <= 0 raises ValueError."""
        img = np.zeros((50, 50), dtype=np.uint8)
        with pytest.raises(ValueError, match="positive floats"):
            register_cross_modal(
                fine_image=img,
                fine_resolution_m_per_px=-1.0,
                coarse_image=img,
                coarse_resolution_m_per_px=5.0,
            )

    def test_unsupported_model_raises_value_error(self):
        """Passing unsupported transform model raises ValueError."""
        img = np.zeros((50, 50), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported transform model"):
            register_cross_modal(
                fine_image=img,
                fine_resolution_m_per_px=1.0,
                coarse_image=img,
                coarse_resolution_m_per_px=5.0,
                transform_model="invalid_model",
            )

    def test_blank_image_returns_structured_failure(self):
        """Blank images yield zero SIFT keypoints and return clean structured failure."""
        blank_fine = np.zeros((200, 200), dtype=np.uint8)
        blank_coarse = np.zeros((50, 50), dtype=np.uint8)

        res = register_cross_modal(
            fine_image=blank_fine,
            fine_resolution_m_per_px=1.0,
            coarse_image=blank_coarse,
            coarse_resolution_m_per_px=4.0,
        )

        assert res.success is False
        assert "Insufficient SIFT features" in res.failure_reason
        assert res.inlier_count == 0
