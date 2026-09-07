"""
SIH26166 — Tests for standalone cross-scale registration module.

Test Structure:
- TIER A: Synthetic data with KNOWN ground truth (rigorous correctness proof of
  resolution normalization and matrix composition math).
- TIER B: Real Chandrayaan-2 PDS4 fixture execution (sanity check confirming real
  pixel formats and metadata flow through the pipeline without crash).
- Input Validation: Tests parameter guards (reversed resolutions, invalid models).
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.core.cross_scale_registration import (
    CrossScaleRegistrationResult,
    compose_scale_transform,
    register_cross_scale,
)
from backend.geometry.transform import transform_points
from backend.preprocessing.pds4 import extract_pds4_metadata, load_pds4_raster
from backend.preprocessing.scale_handler import compute_scale_ratio, get_resolution

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

OHRC_XML = FIXTURES_DIR / "pds4" / "ohrc" / "ch2_ohrc_sample_500x500.xml"
TMC2_XML = FIXTURES_DIR / "pds4" / "tmc2" / "ch2_tmc_sample_500x500.xml"


# =========================================================================
# Synthetic Scene Generator (Multi-scale geometric structure)
# =========================================================================

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
# TIER A: Synthetic Known Ground Truth Tests
# =========================================================================

class TestTierASyntheticGroundTruth:
    """Rigorous verification of cross-scale registration against known ground truth."""

    def test_cross_scale_20x_synthetic_ground_truth(self):
        """Verify ~20x cross-scale registration and transform composition accuracy.

        Setup:
        - Coarse resolution: 5.0 m/px (target base 400x400)
        - Fine resolution: 0.25 m/px (source base 8000x8000, 20.0x scale ratio)
        - Known affine transformation applied to coarse target:
          rotation angle = 3.0 deg around center (200, 200),
          scale factor = 1.0,
          translation tx = +10.0 px, ty = +6.0 px.
        - Ground truth matrix M_full_true = M_coarse_true @ S, mapping
          full-resolution fine points (x_full, y_full) -> coarse points (x_coarse, y_coarse).
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

        # 2. Execute cross-scale registration
        res = register_cross_scale(
            fine_image=fine_image,
            fine_resolution_m_per_px=0.25,
            coarse_image=coarse_target,
            coarse_resolution_m_per_px=5.0,
            transform_model="affine",
        )

        # 3. Assertions on pipeline completion and inlier count
        assert res.success is True, f"Registration failed: {res.failure_reason}"
        assert res.inlier_count >= 50, (
            f"Expected >= 50 inliers for structured synthetic scene, got {res.inlier_count} / {res.total_correspondences}"
        )
        assert res.downsampled_fine_shape == (400, 400)
        assert res.scale_ratio == 0.05
        assert res.coarse_transform_matrix is not None
        assert res.full_resolution_transform_matrix is not None

        # 4. Verify point mapping accuracy using composed M_full
        M_full_true = compose_scale_transform(M_coarse_true, 0.05, "affine")
        M_full_recovered = res.full_resolution_transform_matrix

        # Test points distributed across the 8000x8000 fine canvas
        test_points_fine = np.array([
            [1000.0, 1000.0],
            [2000.0, 3000.0],
            [5000.0, 2000.0],
            [6000.0, 6000.0],
            [7000.0, 1500.0],
        ], dtype=np.float64)

        pts_coarse_expected = transform_points(test_points_fine, M_full_true)
        pts_coarse_actual = transform_points(test_points_fine, M_full_recovered)

        point_errors = np.linalg.norm(pts_coarse_actual - pts_coarse_expected, axis=1)
        max_point_error = float(np.max(point_errors))

        # Assert sub-pixel point mapping accuracy at coarse scale (< 0.5 coarse pixels)
        assert max_point_error < 0.5, (
            f"Composed transformation mapping error too large: max={max_point_error:.3f} px (tolerance 0.5 px). "
            f"Per-point errors: {point_errors}"
        )

    def test_compose_scale_transform_affine_and_homography(self):
        """Verify compose_scale_transform math directly for affine and homography."""
        scale_ratio = 0.05  # 20x downsampling

        # Affine 2x3
        m_coarse_affine = np.array([
            [0.9961947, -0.0871557, 15.0],
            [0.0871557,  0.9961947, -8.0],
        ], dtype=np.float64)

        m_full_affine = compose_scale_transform(m_coarse_affine, scale_ratio, "affine")
        assert m_full_affine.shape == (2, 3)
        assert math.isclose(m_full_affine[0, 0], 0.9961947 * 0.05)
        assert math.isclose(m_full_affine[0, 1], -0.0871557 * 0.05)
        assert math.isclose(m_full_affine[0, 2], 15.0)
        assert math.isclose(m_full_affine[1, 0], 0.0871557 * 0.05)
        assert math.isclose(m_full_affine[1, 1], 0.9961947 * 0.05)
        assert math.isclose(m_full_affine[1, 2], -8.0)

        # Homography 3x3
        h_coarse = np.array([
            [0.99, -0.05, 12.0],
            [0.05,  0.99, -4.0],
            [1e-4, -2e-4,  1.0],
        ], dtype=np.float64)

        h_full = compose_scale_transform(h_coarse, scale_ratio, "homography")
        assert h_full.shape == (3, 3)
        assert math.isclose(h_full[0, 0], 0.99 * 0.05)
        assert math.isclose(h_full[0, 1], -0.05 * 0.05)
        assert math.isclose(h_full[0, 2], 12.0)
        assert math.isclose(h_full[1, 0], 0.05 * 0.05)
        assert math.isclose(h_full[1, 1], 0.99 * 0.05)
        assert math.isclose(h_full[1, 2], -4.0)
        assert math.isclose(h_full[2, 0], 1e-4 * 0.05)
        assert math.isclose(h_full[2, 1], -2e-4 * 0.05)
        assert math.isclose(h_full[2, 2], 1.0)


# =========================================================================
# TIER B: Real PDS4 Fixture Sanity Tests
# =========================================================================

class TestTierBRealFixturesSanity:
    """Sanity execution test on real Chandrayaan-2 OHRC and TMC-2 PDS4 data."""

    def test_real_pds4_ohrc_tmc2_pipeline_execution(self):
        """Execute cross-scale pipeline on genuine OHRC (0.21 m/px) and TMC-2 (4.27 m/px) fixtures.

        Note on Evaluation:
        TMC-2 and OHRC sample fixtures in this test suite do not spatially overlap
        (they cover different geographic regions of the lunar surface). Therefore,
        no registration accuracy or inlier ground truth is asserted here.
        This test only proves that the pipeline runs end-to-end without crashing on
        real pixel rasters with confirmed PDS4 metadata resolutions.
        """
        assert OHRC_XML.exists(), f"Missing OHRC fixture: {OHRC_XML}"
        assert TMC2_XML.exists(), f"Missing TMC2 fixture: {TMC2_XML}"

        # 1. Load real raster data
        ohrc_raw = load_pds4_raster(OHRC_XML)
        tmc2_raw = load_pds4_raster(TMC2_XML)

        # 2. Extract resolutions from PDS4 metadata dicts
        ohrc_meta = extract_pds4_metadata(OHRC_XML)
        tmc2_meta = extract_pds4_metadata(TMC2_XML)

        res_ohrc = get_resolution({"isda_product_params": ohrc_meta.isda_product_params}, "OHRC")
        res_tmc2 = get_resolution({"isda_product_params": tmc2_meta.isda_product_params}, "TMC2")

        assert math.isclose(res_ohrc, 0.21, rel_tol=1e-2)
        assert math.isclose(res_tmc2, 4.27, rel_tol=1e-2)

        # 3. Execute cross-scale registration
        res = register_cross_scale(
            fine_image=ohrc_raw.data,
            fine_resolution_m_per_px=res_ohrc,
            coarse_image=tmc2_raw.data,
            coarse_resolution_m_per_px=res_tmc2,
            transform_model="affine",
        )

        # 4. Verify result object structure and finite outputs
        assert isinstance(res, CrossScaleRegistrationResult)
        assert res.scale_ratio == compute_scale_ratio(res_ohrc, res_tmc2)
        assert res.downsampled_fine_shape == (25, 25)  # 500 * (0.21 / 4.27) = ~24.59 -> 25
        assert res.inlier_count >= 0
        assert res.total_correspondences >= 0
        assert math.isfinite(res.inlier_rmse)


# =========================================================================
# Input Validation Tests
# =========================================================================

class TestCrossScaleInputValidation:
    """Validate argument guards and exception handling."""

    def test_reversed_resolutions_raise_value_error(self):
        """Passing fine resolution >= coarse resolution must raise ValueError."""
        img1 = np.zeros((100, 100), dtype=np.uint8)
        img2 = np.zeros((100, 100), dtype=np.uint8)

        with pytest.raises(ValueError, match="must be strictly smaller"):
            register_cross_scale(img1, 5.0, img2, 0.25)

    def test_equal_resolutions_raise_value_error(self):
        """Passing identical resolutions must raise ValueError."""
        img1 = np.zeros((100, 100), dtype=np.uint8)
        img2 = np.zeros((100, 100), dtype=np.uint8)

        with pytest.raises(ValueError, match="must be strictly smaller"):
            register_cross_scale(img1, 5.0, img2, 5.0)

    def test_non_positive_resolutions_raise_value_error(self):
        """Negative or zero resolutions must raise ValueError."""
        img = np.zeros((100, 100), dtype=np.uint8)

        with pytest.raises(ValueError, match="positive floats"):
            register_cross_scale(img, -0.25, img, 5.0)

        with pytest.raises(ValueError, match="positive floats"):
            register_cross_scale(img, 0.25, img, 0.0)

    def test_unsupported_transform_model_raises_value_error(self):
        """Unsupported transform models (e.g. 'rigid', 'projective') must raise ValueError."""
        img = np.zeros((100, 100), dtype=np.uint8)

        with pytest.raises(ValueError, match="Unsupported transform model"):
            register_cross_scale(img, 0.25, img, 5.0, transform_model="rigid")
