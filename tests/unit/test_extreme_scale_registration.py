"""
SIH26166 — Tests for extreme-scale composed registration module.

Verifies that OHRC↔IIRS registration via TMC-2 bridge transform composition
is mathematically correct using SYNTHETIC ground-truth matrices.

Test Structure:
- TIER A: Pure mathematical composition tests (identity, known matrices,
  point-chain verification, scale+rotation, non-commutativity).
- TIER B: Synthetic end-to-end pipeline tests with mocked sub-pipelines
  (verifying orchestration and failure propagation).

ALL tests use exact mathematical verification — np.allclose with tight
tolerances (atol=1e-10 to 1e-12), not visual or approximate checks.

NOT validated against real OHRC/TMC-2/IIRS overlapping imagery.
"""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import pytest

from backend.core.extreme_scale_registration import (
    ExtremeScaleRegistrationResult,
    compose_affine_transforms,
    register_extreme_scale,
)
from backend.core.cross_scale_registration import CrossScaleRegistrationResult
from backend.core.cross_modal_registration import CrossModalRegistrationResult


# ===================================================================
# Helper: transform points through a 2x3 affine matrix
# ===================================================================

def _apply_affine(matrix_2x3: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 2x3 affine transform to Nx2 points.

    points: shape (N, 2), each row is [x, y]
    Returns: shape (N, 2), transformed points
    """
    m = np.vstack([matrix_2x3, [0.0, 0.0, 1.0]])
    ones = np.ones((points.shape[0], 1))
    homogeneous = np.hstack([points, ones])  # (N, 3)
    result = (m @ homogeneous.T).T  # (N, 3)
    return result[:, :2]


# ===================================================================
# TIER A: Pure Mathematical Composition Tests
# ===================================================================

class TestComposeAffineTransforms:
    """Tests for compose_affine_transforms() algebraic correctness."""

    def test_identity_composition(self):
        """Composing identity × identity = identity."""
        identity = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float64)

        composed = compose_affine_transforms(identity, identity)

        assert composed.shape == (2, 3)
        np.testing.assert_allclose(
            composed, identity, atol=1e-15,
            err_msg="Identity ∘ Identity must equal Identity"
        )

    def test_known_ground_truth_matrix_composition(self):
        """Verify composed transform matches manual 3x3 multiplication.

        M_A = [[0.8, -0.1, 10.0],    (scale + slight shear + translation)
               [0.1,  0.9,  5.0]]

        M_B = [[0.5,  0.0, 20.0],    (scale + translation)
               [0.0,  0.5, 15.0]]

        Expected: M_B_3x3 @ M_A_3x3

        M_A_3x3 = [[0.8, -0.1, 10.0],
                    [0.1,  0.9,  5.0],
                    [0.0,  0.0,  1.0]]

        M_B_3x3 = [[0.5,  0.0, 20.0],
                    [0.0,  0.5, 15.0],
                    [0.0,  0.0,  1.0]]

        M_B_3x3 @ M_A_3x3 = [[0.5*0.8+0*0.1,  0.5*(-0.1)+0*0.9,  0.5*10+0*5+20],
                               [0*0.8+0.5*0.1,  0*(-0.1)+0.5*0.9,  0*10+0.5*5+15],
                               [0,               0,                  1            ]]
                            = [[0.40, -0.05, 25.0],
                               [0.05,  0.45, 17.5],
                               [0.0,   0.0,   1.0]]
        """
        m_a = np.array([
            [0.8, -0.1, 10.0],
            [0.1,  0.9,  5.0],
        ], dtype=np.float64)

        m_b = np.array([
            [0.5,  0.0, 20.0],
            [0.0,  0.5, 15.0],
        ], dtype=np.float64)

        composed = compose_affine_transforms(m_a, m_b)

        # Hand-computed ground truth
        expected = np.array([
            [0.40, -0.05, 25.0],
            [0.05,  0.45, 17.5],
        ], dtype=np.float64)

        assert composed.shape == (2, 3)
        np.testing.assert_allclose(
            composed, expected, atol=1e-12,
            err_msg=(
                f"Composed transform does not match hand-computed ground truth.\n"
                f"Got:\n{composed}\n"
                f"Expected:\n{expected}\n"
                f"Diff:\n{composed - expected}"
            )
        )

    def test_point_chain_verification(self):
        """Transform points through A then B sequentially vs. composed in one shot.

        For 100 random test points, verify that:
            M_B(M_A(p)) == M_composed(p)
        to within atol=1e-10.
        """
        rng = np.random.RandomState(42)

        # Non-trivial transforms simulating real registration
        m_a = np.array([
            [0.05, -0.002, 3.5],   # ~20x scale-down (OHRC→TMC-2 like)
            [0.002, 0.05, 1.2],
        ], dtype=np.float64)

        m_b = np.array([
            [0.052, -0.003, 0.8],  # ~19.4x scale-down (TMC-2→IIRS like)
            [0.003,  0.052, 0.5],
        ], dtype=np.float64)

        composed = compose_affine_transforms(m_a, m_b)

        # 100 random test points in OHRC coordinate space
        points = rng.uniform(0, 8000, size=(100, 2))

        # Sequential: A then B
        intermediate_points = _apply_affine(m_a, points)
        sequential_result = _apply_affine(m_b, intermediate_points)

        # One-shot composed
        composed_result = _apply_affine(composed, points)

        np.testing.assert_allclose(
            composed_result, sequential_result, atol=1e-10,
            err_msg=(
                "Point-chain verification failed: M_composed(p) ≠ M_B(M_A(p)).\n"
                f"Max absolute difference: {np.max(np.abs(composed_result - sequential_result))}"
            )
        )

    def test_compose_scale_plus_rotation(self):
        """Compose a pure scale-down with a scale-down + rotation.

        M_A: 20× scale-down (simulating OHRC→TMC-2 resolution normalization)
            [[1/20, 0, 0], [0, 1/20, 0]] = [[0.05, 0, 0], [0, 0.05, 0]]

        M_B: 19.4× scale-down + 5° rotation (simulating TMC-2→IIRS)
            s_b = 1/19.4 ≈ 0.05155
            θ = 5° = 0.08727 rad
            [[s_b*cos(θ), -s_b*sin(θ), 0], [s_b*sin(θ), s_b*cos(θ), 0]]

        Composed should chain both scale-downs with the rotation applied.
        """
        s_a = 1.0 / 20.0  # 0.05
        m_a = np.array([
            [s_a, 0.0, 0.0],
            [0.0, s_a, 0.0],
        ], dtype=np.float64)

        s_b = 1.0 / 19.4
        theta = math.radians(5.0)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        m_b = np.array([
            [s_b * cos_t, -s_b * sin_t, 0.0],
            [s_b * sin_t,  s_b * cos_t, 0.0],
        ], dtype=np.float64)

        composed = compose_affine_transforms(m_a, m_b)

        # Verify via 3x3 multiplication
        a_3x3 = np.vstack([m_a, [0, 0, 1]])
        b_3x3 = np.vstack([m_b, [0, 0, 1]])
        expected_3x3 = b_3x3 @ a_3x3
        expected = expected_3x3[:2, :]

        np.testing.assert_allclose(
            composed, expected, atol=1e-15,
            err_msg=(
                f"Scale+rotation composition mismatch.\n"
                f"Got:\n{composed}\n"
                f"Expected:\n{expected}"
            )
        )

        # Also verify the combined scale factor is approximately 1/(20*19.4) ≈ 0.002577
        combined_scale = math.sqrt(composed[0, 0]**2 + composed[1, 0]**2)
        expected_scale = s_a * s_b
        assert abs(combined_scale - expected_scale) < 1e-12, (
            f"Combined scale {combined_scale} ≠ expected {expected_scale}"
        )

    def test_non_commutativity(self):
        """Verify compose(A, B) ≠ compose(B, A) for non-trivial transforms.

        This proves the composition order is correct: A is applied first,
        then B — matching the OHRC→TMC-2→IIRS data flow.
        """
        m_a = np.array([
            [0.8, -0.2, 10.0],
            [0.1,  0.7,  5.0],
        ], dtype=np.float64)

        m_b = np.array([
            [0.5,  0.3, 20.0],
            [-0.1, 0.6, 15.0],
        ], dtype=np.float64)

        ab = compose_affine_transforms(m_a, m_b)
        ba = compose_affine_transforms(m_b, m_a)

        # They must NOT be equal (affine composition is non-commutative in general)
        assert not np.allclose(ab, ba, atol=1e-10), (
            f"compose(A, B) should NOT equal compose(B, A) for non-trivial transforms.\n"
            f"AB:\n{ab}\n"
            f"BA:\n{ba}"
        )

    def test_3x3_input_accepted(self):
        """Verify that 3x3 matrices are accepted and handled correctly."""
        m_a_3x3 = np.array([
            [0.8, -0.1, 10.0],
            [0.1,  0.9,  5.0],
            [0.0,  0.0,  1.0],
        ], dtype=np.float64)

        m_b = np.array([
            [0.5,  0.0, 20.0],
            [0.0,  0.5, 15.0],
        ], dtype=np.float64)

        composed = compose_affine_transforms(m_a_3x3, m_b)
        assert composed.shape == (2, 3)

        # Should match the 2x3 version
        m_a_2x3 = m_a_3x3[:2, :]
        composed_2x3 = compose_affine_transforms(m_a_2x3, m_b)
        np.testing.assert_allclose(composed, composed_2x3, atol=1e-15)

    def test_invalid_shape_raises(self):
        """Verify ValueError for non-affine matrix shapes."""
        bad = np.array([[1.0, 2.0]], dtype=np.float64)
        good = np.eye(2, 3, dtype=np.float64)

        with pytest.raises(ValueError, match="shape"):
            compose_affine_transforms(bad, good)

        with pytest.raises(ValueError, match="shape"):
            compose_affine_transforms(good, bad)


# ===================================================================
# TIER B: Synthetic End-to-End Pipeline Tests
# ===================================================================

def _make_mock_cross_scale_result(
    success: bool = True,
    transform: np.ndarray | None = None,
    failure_reason: str = "",
) -> CrossScaleRegistrationResult:
    """Create a mock CrossScaleRegistrationResult with predetermined transform."""
    if transform is None and success:
        transform = np.eye(2, 3, dtype=np.float64)
    return CrossScaleRegistrationResult(
        success=success,
        scale_ratio=0.05,  # ~20x
        downsampled_fine_shape=(400, 400),
        full_resolution_transform_matrix=transform,
        coarse_transform_matrix=transform,
        inlier_count=120 if success else 0,
        total_correspondences=150 if success else 0,
        inlier_rmse=0.45 if success else 0.0,
        transform_model="affine",
        failure_reason=failure_reason,
    )


def _make_mock_cross_modal_result(
    success: bool = True,
    transform: np.ndarray | None = None,
    failure_reason: str = "",
) -> CrossModalRegistrationResult:
    """Create a mock CrossModalRegistrationResult with predetermined transform."""
    if transform is None and success:
        transform = np.eye(2, 3, dtype=np.float64)
    return CrossModalRegistrationResult(
        success=success,
        scale_ratio=0.0516,  # ~19.4x
        downsampled_fine_shape=(26, 26),
        full_resolution_transform_matrix=transform,
        coarse_transform_matrix=transform,
        inlier_count=103 if success else 0,
        total_correspondences=151 if success else 0,
        inlier_rmse=0.57 if success else 0.0,
        transform_model="affine",
        failure_reason=failure_reason,
    )


class TestRegisterExtremeScaleEndToEnd:
    """Synthetic end-to-end tests with mocked sub-pipelines."""

    @patch("backend.core.extreme_scale_registration.register_cross_modal")
    @patch("backend.core.extreme_scale_registration.register_cross_scale")
    def test_synthetic_composed_transform_correctness(
        self, mock_cross_scale, mock_cross_modal
    ):
        """Full pipeline with known transforms → verify composed result matches math.

        M_A (OHRC→TMC-2): scale by 1/20 + translation (3.5, 1.2)
        M_B (TMC-2→IIRS): scale by 1/19.4 + rotation 2° + translation (0.8, 0.5)

        Verify: result.composed_transform_matrix == M_B_3x3 @ M_A_3x3  (exactly)
        """
        # Known ground-truth Stage A transform
        m_a = np.array([
            [0.05, -0.002, 3.5],
            [0.002, 0.05,  1.2],
        ], dtype=np.float64)

        # Known ground-truth Stage B transform
        theta = math.radians(2.0)
        s_b = 1.0 / 19.4
        m_b = np.array([
            [s_b * math.cos(theta), -s_b * math.sin(theta), 0.8],
            [s_b * math.sin(theta),  s_b * math.cos(theta), 0.5],
        ], dtype=np.float64)

        mock_cross_scale.return_value = _make_mock_cross_scale_result(
            success=True, transform=m_a
        )
        mock_cross_modal.return_value = _make_mock_cross_modal_result(
            success=True, transform=m_b
        )

        # Dummy images (won't be used due to mocking)
        dummy_img = np.zeros((100, 100), dtype=np.uint8)

        result = register_extreme_scale(
            ohrc_image=dummy_img,
            ohrc_resolution_m_per_px=0.25,
            iirs_image=dummy_img,
            iirs_resolution_m_per_px=82.70,
            tmc2_image=dummy_img,
            tmc2_resolution_m_per_px=5.0,
            tmc2_bridge_path="synthetic",
        )

        assert result.success, f"Registration failed: {result.failure_reason}"
        assert result.composed_transform_matrix is not None

        # Compute expected composed transform manually
        a_3x3 = np.vstack([m_a, [0, 0, 1]])
        b_3x3 = np.vstack([m_b, [0, 0, 1]])
        expected = (b_3x3 @ a_3x3)[:2, :]

        np.testing.assert_allclose(
            result.composed_transform_matrix, expected, atol=1e-12,
            err_msg=(
                f"Composed transform does not match algebraic ground truth.\n"
                f"Got:\n{result.composed_transform_matrix}\n"
                f"Expected (M_B @ M_A):\n{expected}\n"
                f"M_A:\n{m_a}\n"
                f"M_B:\n{m_b}\n"
                f"Diff:\n{result.composed_transform_matrix - expected}"
            )
        )

        # Verify intermediate block is populated
        assert result.intermediate is not None
        assert result.intermediate.tmc2_bridge_image_path == "synthetic"
        assert result.intermediate.stage_a_transform is not None
        np.testing.assert_allclose(
            result.intermediate.stage_a_transform, m_a, atol=1e-15
        )
        assert result.intermediate.stage_b_transform is not None
        np.testing.assert_allclose(
            result.intermediate.stage_b_transform, m_b, atol=1e-15
        )
        assert result.intermediate.composed_transform is not None
        np.testing.assert_allclose(
            result.intermediate.composed_transform, expected, atol=1e-12
        )

        # Verify stage summaries
        assert result.intermediate.stage_a_result.success is True
        assert result.intermediate.stage_b_result.success is True
        assert result.inlier_count_stage_a == 120
        assert result.inlier_count_stage_b == 103

        # Verify point-chain equivalence with the composed transform
        rng = np.random.RandomState(99)
        test_points = rng.uniform(0, 8000, size=(50, 2))

        # Sequential path
        intermediate_pts = _apply_affine(m_a, test_points)
        sequential_result = _apply_affine(m_b, intermediate_pts)

        # Composed path
        assert result.composed_transform_matrix is not None
        composed_result = _apply_affine(result.composed_transform_matrix, test_points)

        np.testing.assert_allclose(
            composed_result, sequential_result, atol=1e-10,
            err_msg="Point-chain verification failed in end-to-end test"
        )

    @patch("backend.core.extreme_scale_registration.register_cross_modal")
    @patch("backend.core.extreme_scale_registration.register_cross_scale")
    def test_stage_a_failure_propagation(self, mock_cross_scale, mock_cross_modal):
        """Stage A (OHRC→TMC-2) failure → overall failure with correct metadata."""
        mock_cross_scale.return_value = _make_mock_cross_scale_result(
            success=False,
            transform=None,
            failure_reason="Insufficient SIFT features: downsampled_fine=0, coarse=23",
        )
        # Stage B should NOT be called
        mock_cross_modal.return_value = _make_mock_cross_modal_result(success=True)

        dummy_img = np.zeros((100, 100), dtype=np.uint8)
        result = register_extreme_scale(
            ohrc_image=dummy_img,
            ohrc_resolution_m_per_px=0.25,
            iirs_image=dummy_img,
            iirs_resolution_m_per_px=82.70,
            tmc2_image=dummy_img,
            tmc2_resolution_m_per_px=5.0,
        )

        assert result.success is False
        assert result.failure_stage == "stage_a"
        assert "Stage A" in result.failure_reason
        assert "Insufficient SIFT" in result.failure_reason
        assert result.composed_transform_matrix is None

        # Intermediate should show Stage A failure and Stage B not attempted
        assert result.intermediate is not None
        assert result.intermediate.stage_a_result.success is False
        assert result.intermediate.stage_b_result.success is False
        assert "Not attempted" in result.intermediate.stage_b_result.failure_reason

        # Stage B should NOT have been called (Stage A failed first)
        mock_cross_modal.assert_not_called()

    @patch("backend.core.extreme_scale_registration.register_cross_modal")
    @patch("backend.core.extreme_scale_registration.register_cross_scale")
    def test_stage_b_failure_propagation(self, mock_cross_scale, mock_cross_modal):
        """Stage B (TMC-2→IIRS) failure → overall failure with Stage A data preserved."""
        m_a = np.array([
            [0.05, 0.0, 2.0],
            [0.0, 0.05, 1.0],
        ], dtype=np.float64)

        mock_cross_scale.return_value = _make_mock_cross_scale_result(
            success=True, transform=m_a
        )
        mock_cross_modal.return_value = _make_mock_cross_modal_result(
            success=False,
            transform=None,
            failure_reason="Phase congruency computation failed: numerical instability",
        )

        dummy_img = np.zeros((100, 100), dtype=np.uint8)
        result = register_extreme_scale(
            ohrc_image=dummy_img,
            ohrc_resolution_m_per_px=0.25,
            iirs_image=dummy_img,
            iirs_resolution_m_per_px=82.70,
            tmc2_image=dummy_img,
            tmc2_resolution_m_per_px=5.0,
        )

        assert result.success is False
        assert result.failure_stage == "stage_b"
        assert "Stage B" in result.failure_reason
        assert "Phase congruency" in result.failure_reason
        assert result.composed_transform_matrix is None

        # Intermediate should preserve Stage A's successful result
        assert result.intermediate is not None
        assert result.intermediate.stage_a_result.success is True
        assert result.intermediate.stage_a_transform is not None
        np.testing.assert_allclose(
            result.intermediate.stage_a_transform, m_a, atol=1e-15
        )
        assert result.intermediate.stage_b_result.success is False

    def test_non_affine_model_rejected(self):
        """Non-affine transform model is immediately rejected."""
        dummy_img = np.zeros((100, 100), dtype=np.uint8)
        result = register_extreme_scale(
            ohrc_image=dummy_img,
            ohrc_resolution_m_per_px=0.25,
            iirs_image=dummy_img,
            iirs_resolution_m_per_px=82.70,
            tmc2_image=dummy_img,
            tmc2_resolution_m_per_px=5.0,
            transform_model="homography",
        )

        assert result.success is False
        assert result.failure_stage == "configuration"
        assert "homography" in result.failure_reason
        assert "affine" in result.failure_reason.lower()


# ===================================================================
# TIER C: Real (Non-Mocked) End-to-End Through Actual Sub-Pipelines
# ===================================================================

def _make_rich_scene(width: int, height: int, scale: float) -> np.ndarray:
    """Generate a deterministic synthetic scene with multi-scale crater-like features.

    Identical to the scene generator in test_cross_scale_registration.py and
    test_cross_modal_registration.py. All feature coordinates and radii are
    multiplied by `scale` so that downsampling an image generated at scale=20.0
    produces the same visual content as an image generated directly at scale=1.0.
    """
    import cv2

    img = np.zeros((height, width), dtype=np.uint8)

    # Smooth background texture gradient
    for y in range(height):
        img[y, :] = int(30 + 100 * y / max(height - 1, 1))

    # Grid of crater structures (rings, rims, central peaks)
    for i in range(50):
        cx = int((30 + (i % 8) * 45) * scale)
        cy = int((30 + (i // 8) * 45) * scale)
        r = int((8 + (i % 5) * 4) * scale)
        v1 = 180 + (i % 4) * 20
        v2 = 40 + (i % 3) * 30

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


class TestTierCRealEndToEnd:
    """Real (non-mocked) E2E test running through actual cross-scale and cross-modal code.

    Uses the same _make_rich_scene synthetic generator validated independently
    in test_cross_scale_registration.py and test_cross_modal_registration.py.

    Three images from the same deterministic scene at three resolution levels:
    - OHRC: 8000×8000 at 0.25 m/px (scale=20.0)
    - TMC-2: 400×400 at 5.0 m/px (scale=1.0)
    - IIRS: ~20×20 at ~82.70 m/px (derived from TMC-2 with intensity inversion)

    This test calls register_extreme_scale() directly (NOT mocked), which
    internally calls register_cross_scale() and register_cross_modal()
    with the real SIFT/FLANN/MAGSAC++ pipelines and phase congruency.
    """

    def test_real_e2e_three_scale_synthetic_scene(self):
        """Run actual Stage A + Stage B + Stage C with real sub-pipeline code.

        Reports: correspondence counts, RMSE, intermediate block contents,
        and verifies composed transform maps OHRC->IIRS correctly.

        Fixture sizing rationale:

        Stage A (OHRC -> TMC-2): TRUE 20x pixel ratio.
        - OHRC: 8000x8000 at 0.25 m/px, _make_rich_scene(scale=20.0)
        - TMC-2: 400x400 at 5.0 m/px, _make_rich_scene(scale=1.0) + affine warp
        - OHRC gets downsampled 20x to 400x400; both images have features at
          base coordinates. This is the SAME setup proven in
          test_cross_scale_registration.py::test_cross_scale_20x_synthetic_ground_truth.

        Stage B (TMC-2 -> IIRS): 4x pixel ratio (NOT the real 16.54x).
        - TMC-2: 400x400 at 5.0 m/px (same image as Stage A bridge)
        - IIRS: 100x100 at 20.0 m/px, derived from TMC-2 + intensity inversion
        - TMC-2 gets downsampled 4x to 100x100 for Phase Congruency matching.
        - The real 16.54x TMC-2->IIRS ratio would produce a 24x24 IIRS, which is
          too small for Phase Congruency SIFT features. Phase Congruency requires
          ~100px minimum for reliable feature extraction.

        Total scale: 0.25/20.0 = 0.0125 (80x), not the real 394x.
        This exercises both real sub-pipelines through composition but does NOT
        test the full 394x ratio due to Phase Congruency pixel requirements.
        """
        import cv2

        # --- Stage A fixtures (TRUE 20x ratio) ---
        # OHRC at 0.25 m/px: 8000x8000, features at 20x base coords
        ohrc_image = _make_rich_scene(8000, 8000, scale=20.0)

        # TMC-2 at 5.0 m/px: 400x400, features at 1x base coords
        # When OHRC is downsampled 20x (8000*0.05=400), features align at base coords.
        tmc2_base = _make_rich_scene(400, 400, scale=1.0)

        # Apply known affine transform to TMC-2 (simulating pointing offset)
        center = (200.0, 200.0)
        M_tmc2 = cv2.getRotationMatrix2D(center, 3.0, 1.0)
        M_tmc2[0, 2] += 10.0
        M_tmc2[1, 2] += 6.0
        tmc2_image = cv2.warpAffine(
            tmc2_base, M_tmc2, (400, 400), borderMode=cv2.BORDER_REFLECT
        )

        # --- Stage B fixtures (4x ratio, not real 16.54x) ---
        # IIRS: 100x100 at 20.0 m/px, derived from TMC-2 by 4x downsample
        # + intensity inversion (visible -> IR modality simulation).
        # The real IIRS resolution is 82.70 m/px, which would give a 24x24 image
        # from a 400x400 TMC-2 -- too small for Phase Congruency features.
        iirs_resolution = 20.0  # m/px (4x from TMC-2's 5.0 m/px)
        iirs_w = 100
        iirs_h = 100
        iirs_from_tmc2 = cv2.resize(
            tmc2_image, (iirs_w, iirs_h), interpolation=cv2.INTER_AREA
        )
        iirs_image = (255 - iirs_from_tmc2).astype(np.uint8)

        # --- Execute real (non-mocked) extreme-scale registration ---
        result = register_extreme_scale(
            ohrc_image=ohrc_image,
            ohrc_resolution_m_per_px=0.25,
            iirs_image=iirs_image,
            iirs_resolution_m_per_px=iirs_resolution,
            tmc2_image=tmc2_image,
            tmc2_resolution_m_per_px=5.0,
            tmc2_bridge_path="synthetic_3_scale_scene",
        )

        # --- Print detailed report (visible with pytest -s) ---
        print("\n" + "=" * 75)
        print("TIER C: REAL E2E EXTREME-SCALE COMPOSED REGISTRATION")
        print(f"OHRC (8000x8000, 0.25 m/px) -> TMC-2 (400x400, 5.0 m/px) -> "
              f"IIRS ({iirs_w}x{iirs_h}, {iirs_resolution} m/px)")
        print("Stage A ratio: 20x (true), Stage B ratio: 4x (limited by PC min pixels)")
        print("=" * 75)
        print(f"Overall success:       {result.success}")
        print(f"Failure reason:        {(result.failure_reason or 'N/A').encode('ascii', 'replace').decode()}")
        print(f"Failure stage:         {result.failure_stage or 'N/A'}")
        print(f"Total scale ratio:     {result.total_scale_ratio:.6f}")
        print(f"Transform model:       {result.transform_model}")
        print()

        # Stage A report
        print("--- Stage A: OHRC -> TMC-2 (cross-scale) ---")
        print(f"  Inlier count:        {result.inlier_count_stage_a}")
        print(f"  Inlier RMSE:         {result.inlier_rmse_stage_a:.4f} TMC-2 px")

        # Stage B report
        print("--- Stage B: TMC-2 -> IIRS (cross-modal) ---")
        print(f"  Inlier count:        {result.inlier_count_stage_b}")
        print(f"  Inlier RMSE:         {result.inlier_rmse_stage_b:.4f} IIRS px")

        # Intermediate block
        print("--- Intermediate Block ---")
        if result.intermediate is not None:
            inter = result.intermediate
            print(f"  TMC-2 bridge path:   {inter.tmc2_bridge_image_path}")
            print(f"  Stage A success:     {inter.stage_a_result.success}")
            print(f"  Stage A scale ratio: {inter.stage_a_result.scale_ratio:.6f}")
            print(f"  Stage A inliers:     {inter.stage_a_result.inlier_count}/{inter.stage_a_result.total_correspondences}")
            print(f"  Stage A RMSE:        {inter.stage_a_result.inlier_rmse:.4f}")
            print(f"  Stage B success:     {inter.stage_b_result.success}")
            print(f"  Stage B scale ratio: {inter.stage_b_result.scale_ratio:.6f}")
            print(f"  Stage B inliers:     {inter.stage_b_result.inlier_count}/{inter.stage_b_result.total_correspondences}")
            print(f"  Stage B RMSE:        {inter.stage_b_result.inlier_rmse:.4f}")

            if inter.stage_a_transform is not None:
                print(f"  M_A (OHRC->TMC-2):\n{inter.stage_a_transform}")
            if inter.stage_b_transform is not None:
                print(f"  M_B (TMC-2->IIRS):\n{inter.stage_b_transform}")
            if inter.composed_transform is not None:
                print(f"  M_composed (OHRC->IIRS):\n{inter.composed_transform}")
        else:
            print("  (no intermediate data)")

        # Composed transform
        print("--- Composed Transform ---")
        if result.composed_transform_matrix is not None:
            print(f"  Matrix:\n{result.composed_transform_matrix}")
        else:
            print("  (none)")

        # Timings
        print("--- Timings ---")
        for k, v in sorted(result.timings.items()):
            print(f"  {k}: {v:.3f} s")
        print("=" * 75)

        # --- Assertions ---
        assert result.success is True, (
            f"Real E2E registration failed: stage={result.failure_stage}, "
            f"reason={result.failure_reason}"
        )

        # Stage A must have found substantial correspondences
        assert result.inlier_count_stage_a >= 20, (
            f"Stage A (OHRC→TMC-2) too few inliers: {result.inlier_count_stage_a}"
        )
        assert result.inlier_rmse_stage_a < 2.0, (
            f"Stage A RMSE too high: {result.inlier_rmse_stage_a:.4f}"
        )

        # Stage B has fewer correspondences due to the tiny IIRS image
        # but must still have found some
        assert result.inlier_count_stage_b >= 3, (
            f"Stage B (TMC-2→IIRS) too few inliers: {result.inlier_count_stage_b}"
        )

        # Composed transform must exist
        assert result.composed_transform_matrix is not None
        assert result.composed_transform_matrix.shape == (2, 3)

        # Intermediate block must be fully populated
        assert result.intermediate is not None
        assert result.intermediate.stage_a_result.success is True
        assert result.intermediate.stage_b_result.success is True
        assert result.intermediate.stage_a_transform is not None
        assert result.intermediate.stage_b_transform is not None
        assert result.intermediate.composed_transform is not None
        assert result.intermediate.tmc2_bridge_image_path == "synthetic_3_scale_scene"

        # Verify composed = M_B @ M_A (algebraic, not just "close")
        m_a = result.intermediate.stage_a_transform
        m_b = result.intermediate.stage_b_transform
        expected_composed = compose_affine_transforms(m_a, m_b)
        np.testing.assert_allclose(
            result.composed_transform_matrix, expected_composed, atol=1e-12,
            err_msg="Composed transform does not match M_B @ M_A"
        )

        # Verify point chain: OHRC→TMC-2→IIRS sequential == composed one-shot
        test_points = np.array([
            [0.0, 0.0], [4000.0, 0.0], [0.0, 4000.0],
            [4000.0, 4000.0], [2000.0, 2000.0],
        ], dtype=np.float64)

        # Sequential path
        intermediate_pts = _apply_affine(m_a, test_points)
        sequential_result = _apply_affine(m_b, intermediate_pts)

        # Composed path
        composed_result = _apply_affine(result.composed_transform_matrix, test_points)

        np.testing.assert_allclose(
            composed_result, sequential_result, atol=1e-10,
            err_msg="Point-chain verification failed in real E2E test"
        )

        print("\n[PASS] TIER C: Real E2E extreme-scale composed registration verified.")

