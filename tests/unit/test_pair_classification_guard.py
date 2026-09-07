"""SIH26166 — Unit tests for the pair-classification pre-flight guard.

Validates that run_classical_registration() enforces early rejection of
scientifically unsupported cross-instrument pairs (OHRC vs TMC-2, TMC-2 vs IIRS,
OHRC vs IIRS) while preserving same-instrument PDS4 registration and non-PDS4
(UNCLASSIFIED) plain image fallback registration with equivalent numeric outcomes.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.core.registration_service import run_classical_registration

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

TMC2_DIR = FIXTURES_DIR / "pds4" / "tmc2"
TMC2_XML = TMC2_DIR / "ch2_tmc_sample_500x500.xml"
TMC2_IMG = TMC2_DIR / "ch2_tmc_sample_500x500.img"

OHRC_DIR = FIXTURES_DIR / "pds4" / "ohrc"
OHRC_XML = OHRC_DIR / "ch2_ohrc_sample_500x500.xml"
OHRC_IMG = OHRC_DIR / "ch2_ohrc_sample_500x500.img"

IIRS_DIR = FIXTURES_DIR / "pds4" / "iirs"
IIRS_XML = FIXTURES_DIR / "pds4" / "iirs" / "ch2_iirs_sample_200x200x16.xml"
IIRS_QUB = FIXTURES_DIR / "pds4" / "iirs" / "ch2_iirs_sample_200x200x16.qub"

REF_PNG = FIXTURES_DIR / "ref_lunar.png"
TGT_PNG = FIXTURES_DIR / "tgt_lunar.png"


# =========================================================================
# 1. Cross-Instrument Rejection Guard
# =========================================================================

class TestCrossInstrumentRejectionGuard:
    """Test early pre-flight rejection of unsupported cross-instrument pairs."""

    def test_cross_scale_ohrc_tmc2_routed_to_cross_scale(self):
        """OHRC vs TMC-2 is now implemented and routed to cross_scale_registration (not rejected at guard)."""
        result = run_classical_registration(OHRC_XML, TMC2_XML)

        # Because the sample fixtures do not spatially overlap, registration fails at cross_scale stage,
        # proving it passed the pair_classification guard and entered the cross-scale pipeline.
        assert result.pair_type == "CROSS_SCALE_OHRC_TMC2"
        assert result.pipeline_mode == "cross_scale_affine_v1"
        assert result.failure_stage == "cross_scale_registration"
        assert "cross-scale registration" in result.failure_reason.lower()
        assert "load" in result.timings
        assert "cross_scale_registration" in result.timings

    def test_cross_modal_tmc2_iirs_routed(self):
        """TMC-2 (visible) vs IIRS (infrared) now routes to dedicated cross-modal branch."""
        result = run_classical_registration(TMC2_XML, IIRS_XML)

        assert result.success is False
        assert result.failure_stage == "cross_modal_registration"
        assert result.pair_type == "CROSS_MODAL_TMC2_IIRS"
        assert result.pipeline_mode == "cross_modal_phase_congruency_v1"
        assert "cross-modal registration" in result.failure_reason.lower()
        # Proves it reached load and cross_modal_registration stages
        assert "load" in result.timings
        assert "cross_modal_registration" in result.timings

    def test_cross_modal_extreme_scale_ohrc_iirs_requires_bridge(self):
        """OHRC (0.25m visible) vs IIRS (82.7m IR, ~394x ratio) requires TMC-2 bridge.

        The pair type is now is_implemented=True with extreme_scale_composed_v1.
        It passes the pair_classification guard and enters the dedicated extreme-scale
        branch, which immediately fails at 'configuration' because no TMC-2 bridge
        image was provided (tmc2_bridge_path=None).
        """
        result = run_classical_registration(OHRC_XML, IIRS_XML)

        assert result.success is False
        assert result.failure_stage == "configuration"
        assert result.pair_type == "CROSS_MODAL_EXTREME_SCALE_OHRC_IIRS"
        assert result.pipeline_mode == "extreme_scale_composed_v1"
        assert "TMC-2 bridge" in result.failure_reason
        assert "tmc2_bridge_path" in result.failure_reason
        assert result.registered_image is None
        assert result.quality_summary is None

    def test_rejection_is_order_independent(self):
        """Routing and guard rejection must trigger regardless of reference/target ordering."""
        # Reverse ordering tests: TMC2 vs OHRC routes to cross-scale
        res_tmc2_ohrc = run_classical_registration(TMC2_XML, OHRC_XML)
        assert res_tmc2_ohrc.pair_type == "CROSS_SCALE_OHRC_TMC2"
        assert res_tmc2_ohrc.pipeline_mode == "cross_scale_affine_v1"
        assert res_tmc2_ohrc.failure_stage == "cross_scale_registration"

        # Reverse ordering for TMC2 vs IIRS routes to cross-modal
        res_iirs_tmc2 = run_classical_registration(IIRS_XML, TMC2_XML)
        assert res_iirs_tmc2.success is False
        assert res_iirs_tmc2.failure_stage == "cross_modal_registration"
        assert res_iirs_tmc2.pair_type == "CROSS_MODAL_TMC2_IIRS"
        assert res_iirs_tmc2.pipeline_mode == "cross_modal_phase_congruency_v1"
        assert "cross_modal_registration" in res_iirs_tmc2.timings

        res_iirs_ohrc = run_classical_registration(IIRS_XML, OHRC_XML)
        assert res_iirs_ohrc.success is False
        assert res_iirs_ohrc.failure_stage == "configuration"
        assert res_iirs_ohrc.pair_type == "CROSS_MODAL_EXTREME_SCALE_OHRC_IIRS"
        assert "TMC-2 bridge" in res_iirs_ohrc.failure_reason


# =========================================================================
# 2. Same-Instrument PDS4 Registration Preservation
# =========================================================================

class TestSameInstrumentPDS4Preservation:
    """Test that same-instrument PDS4 pairs proceed and produce high-quality metrics."""

    def test_same_instrument_tmc2_numeric_preservation(self, tmp_path):
        """TMC-2 vs transformed TMC-2 must succeed with sub-pixel RMSE and ~250 inliers."""
        ref_data = np.fromfile(TMC2_IMG, dtype="<u2").reshape((500, 500))
        M_synth = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0]], dtype=np.float32)
        tgt_data = cv2.warpAffine(
            ref_data, M_synth, (500, 500), borderMode=cv2.BORDER_REFLECT
        )

        tgt_img_path = tmp_path / "tmc2_tgt.img"
        tgt_img_path.write_bytes(tgt_data.astype("<u2").tobytes())
        xml_text = TMC2_XML.read_text(encoding="utf-8").replace(
            "ch2_tmc_sample_500x500.img", "tmc2_tgt.img"
        )
        tgt_xml_path = tmp_path / "tmc2_tgt.xml"
        tgt_xml_path.write_text(xml_text, encoding="utf-8")

        result = run_classical_registration(TMC2_XML, tgt_xml_path)

        assert result.success is True
        assert result.pair_type == "SAME_INSTRUMENT_TMC2"
        assert result.quality_summary is not None
        assert result.quality_summary.inlier_count >= 240
        assert result.quality_summary.total_correspondences >= 250
        assert result.quality_summary.inlier_ratio > 0.95
        assert result.quality_summary.inlier_rmse < 0.25
        assert result.registered_image is not None
        assert "sift" in result.timings
        assert "load" in result.timings

    def test_same_instrument_ohrc_rotation_scale_numeric_preservation(self, tmp_path):
        """OHRC vs rotated/scaled OHRC must succeed with sub-pixel RMSE and ~440 inliers."""
        ref_data = np.fromfile(OHRC_IMG, dtype=np.uint8).reshape((500, 500))
        center = (250.0, 250.0)
        M_synth = cv2.getRotationMatrix2D(center, 8.0, 0.95)
        M_synth[0, 2] += 20.0
        M_synth[1, 2] += -12.0
        tgt_data = cv2.warpAffine(
            ref_data, M_synth, (500, 500), borderMode=cv2.BORDER_REFLECT
        )

        tgt_img_path = tmp_path / "ohrc_tgt.img"
        tgt_img_path.write_bytes(tgt_data.tobytes())
        xml_text = OHRC_XML.read_text(encoding="utf-8").replace(
            "ch2_ohrc_sample_500x500.img", "ohrc_tgt.img"
        )
        tgt_xml_path = tmp_path / "ohrc_tgt.xml"
        tgt_xml_path.write_text(xml_text, encoding="utf-8")

        result = run_classical_registration(OHRC_XML, tgt_xml_path)

        assert result.success is True
        assert result.pair_type == "SAME_INSTRUMENT_OHRC"
        assert result.quality_summary is not None
        assert result.quality_summary.inlier_count >= 400
        assert result.quality_summary.total_correspondences >= 420
        assert result.quality_summary.inlier_ratio > 0.95
        assert result.quality_summary.inlier_rmse < 0.25
        assert result.registered_image is not None
        assert "sift" in result.timings

    def test_same_instrument_iirs_numeric_preservation(self, tmp_path):
        """IIRS vs transformed IIRS must succeed with sub-pixel RMSE and >100 inliers."""
        ref_data = np.fromfile(IIRS_QUB, dtype="<f4").reshape((200, 200, 16))
        M_synth = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0]], dtype=np.float32)
        tgt_data = np.zeros_like(ref_data)
        for b in range(16):
            tgt_data[:, :, b] = cv2.warpAffine(
                ref_data[:, :, b], M_synth, (200, 200), borderMode=cv2.BORDER_REFLECT
            )

        tgt_qub_path = tmp_path / "iirs_tgt.qub"
        tgt_qub_path.write_bytes(tgt_data.astype("<f4").tobytes())
        xml_text = IIRS_XML.read_text(encoding="utf-8").replace(
            "ch2_iirs_sample_200x200x16.qub", "iirs_tgt.qub"
        )
        tgt_xml_path = tmp_path / "iirs_tgt.xml"
        tgt_xml_path.write_text(xml_text, encoding="utf-8")

        result = run_classical_registration(IIRS_XML, tgt_xml_path)

        assert result.success is True
        assert result.pair_type == "SAME_INSTRUMENT_IIRS"
        assert result.quality_summary is not None
        assert result.quality_summary.inlier_count >= 100
        assert result.quality_summary.total_correspondences >= 150
        assert result.quality_summary.inlier_ratio > 0.60
        assert result.quality_summary.inlier_rmse < 0.15
        assert result.registered_image is not None
        assert "sift" in result.timings


# =========================================================================
# 3. Plain PNG / Non-PDS4 UNCLASSIFIED Fallback Preservation
# =========================================================================

class TestNonPDS4UnclassifiedFallback:
    """Test that plain PNG inputs fall back to UNCLASSIFIED and run unaffected."""

    def test_plain_png_unclassified_numeric_preservation(self):
        """Plain PNG fixture pair must succeed with exact expected metrics."""
        assert REF_PNG.exists(), f"Missing fixture {REF_PNG}"
        assert TGT_PNG.exists(), f"Missing fixture {TGT_PNG}"

        result = run_classical_registration(REF_PNG, TGT_PNG)

        assert result.success is True
        assert result.pair_type == "UNCLASSIFIED"
        assert result.quality_summary is not None
        assert result.quality_summary.inlier_count == 102
        assert result.quality_summary.total_correspondences == 103
        assert math.isclose(result.quality_summary.inlier_rmse, 0.22398736, rel_tol=1e-3)
        assert result.registered_image is not None
        assert "sift" in result.timings
        assert "load" in result.timings
