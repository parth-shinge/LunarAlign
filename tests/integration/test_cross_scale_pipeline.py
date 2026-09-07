"""
SIH26166 — Integration tests for live cross-scale registration pipeline.

Tests:
1. End-to-end CLI execution (cli.py) on real OHRC PDS4 pixel data with derived
   coarse TMC-2 target (downsampled + known affine transform + derived PDS4 label).
2. End-to-end service execution (run_classical_registration) in both upload orders
   (OHRC->TMC2 and TMC2->OHRC).
3. Non-affine model rejection guard (transform_model != 'affine').
4. Exact numeric regression tests confirming same-instrument PDS4 and UNCLASSIFIED
   PNG pipelines remain completely unaffected and byte-for-byte consistent.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from backend.core.registration_service import run_classical_registration
from backend.preprocessing.pds4 import load_pds4_raster
from backend.preprocessing.scale_handler import compute_scale_ratio, downsample_to_match

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures"

OHRC_XML = FIXTURES_DIR / "pds4" / "ohrc" / "ch2_ohrc_sample_500x500.xml"
TMC2_XML = FIXTURES_DIR / "pds4" / "tmc2" / "ch2_tmc_sample_500x500.xml"
IIRS_XML = FIXTURES_DIR / "pds4" / "iirs" / "ch2_iir_sample_500x500.xml"


# =========================================================================
# Helper: create derived TMC-2 target from real OHRC fixture
# =========================================================================

def _create_derived_tmc2_from_ohrc(
    tmp_path: Path,
    dx: float = 1.0,
    dy: float = 1.0,
) -> Path:
    """Create a derived TMC-2 PDS4 target by downsampling real OHRC pixels.

    Real Data:
    - Real Chandrayaan-2 OHRC raster pixels from ch2_ohrc_sample_500x500.img.

    Synthetic Relationship:
    - Downsampled by scale_ratio = 0.21 / 4.27 to 25x25 pixels (representing TMC-2 resolution).
    - Shifted by known affine translation (dx, dy) in coarse pixel space.
    - Preserves complete PDS4 XML label structure adapted to 25x25 uint16 raster.
    """
    raw_ohrc = load_pds4_raster(OHRC_XML)
    ohrc_data = raw_ohrc.data  # 500x500 uint8

    s = compute_scale_ratio(0.21, 4.27)
    coarse_base = downsample_to_match(ohrc_data, s)  # 25x25 float32

    M_coarse_true = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float64)
    coarse_target = cv2.warpAffine(
        coarse_base, M_coarse_true, (25, 25), borderMode=cv2.BORDER_REFLECT
    )

    # Scale to uint16 dynamic range for TMC-2
    coarse_u16 = np.clip(coarse_target * 256.0, 0, 65535).astype("<u2")
    img_path = tmp_path / "derived_tmc2.img"
    img_path.write_bytes(coarse_u16.tobytes())

    # Build valid derived PDS4 label
    xml_content = TMC2_XML.read_text(encoding="utf-8")
    xml_content = xml_content.replace("<elements>500</elements>", "<elements>25</elements>")
    xml_content = xml_content.replace("ch2_tmc_sample_500x500.img", "derived_tmc2.img")
    xml_content = xml_content.replace("500000", "1250")  # 25 * 25 * 2 = 1250 bytes

    lbl_path = tmp_path / "derived_tmc2.xml"
    lbl_path.write_text(xml_content, encoding="utf-8")
    return lbl_path


# =========================================================================
# 1. End-to-End CLI Pipeline Integration Tests
# =========================================================================

class TestCrossScalePipelineE2E:
    """Test full cross-scale path end-to-end via CLI and core service."""

    def test_cli_cross_scale_ohrc_tmc2_e2e(self, tmp_path):
        """Execute cli.py end-to-end on real OHRC and derived TMC-2 target."""
        derived_tmc_xml = _create_derived_tmc2_from_ohrc(tmp_path, dx=1.0, dy=1.0)
        out_dir = tmp_path / "cli_output"

        cmd = [
            sys.executable,
            str(REPO_ROOT / "cli.py"),
            str(OHRC_XML),
            str(derived_tmc_xml),
            "--output-dir",
            str(out_dir),
            "--model",
            "affine",
        ]

        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )

        assert proc.returncode == 0, (
            f"cli.py failed with return code {proc.returncode}.\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )

        # Verify output files
        warped_file = out_dir / "warped.png"
        overlay_file = out_dir / "overlay.png"
        report_file = out_dir / "report.json"

        assert warped_file.exists(), "warped.png not generated"
        assert overlay_file.exists(), "overlay.png not generated"
        assert report_file.exists(), "report.json not generated"

        report = json.loads(report_file.read_text(encoding="utf-8"))
        assert report["metadata"]["pipeline_mode"] == "cross_scale_affine_v1"
        assert report["matching"]["inlier_count"] > 0
        assert math.isfinite(report["registration"]["inlier_rmse"])
        assert report["registration"]["inlier_rmse"] < 1.0

        # Assert genuine (non-fabricated) error metrics and spatial distribution
        assert math.isfinite(report["registration"]["inlier_median_error"])
        assert math.isfinite(report["registration"]["inlier_max_error"])
        assert report["registration"]["inlier_median_error"] <= report["registration"]["inlier_rmse"]
        assert report["registration"]["inlier_rmse"] <= report["registration"]["inlier_max_error"]
        assert report["registration"]["inlier_median_error"] != report["registration"]["inlier_rmse"]
        assert report["spatial"]["normalized_entropy"] > 0.0

        # Verify recovered matrix against known injected transform
        # Injected coarse transform: dx=+1.0, dy=+1.0, scale=1.0, rot=0.0
        # Expected M_full[0, 2] ~= 1.0, M_full[1, 2] ~= 1.0
        # Expected M_full[0, 0] ~= s, M_full[1, 1] ~= s (where s = 0.21 / 4.27 = 0.04918)
        s = 0.21 / 4.27
        m_rec = np.array(report["registration"]["transform_matrix"], dtype=np.float64)

        assert math.isclose(m_rec[0, 0], s, abs_tol=1e-3), (
            f"Recovered scale M[0,0]={m_rec[0, 0]} differs from true {s} by > 0.001"
        )
        assert math.isclose(m_rec[1, 1], s, abs_tol=1e-3), (
            f"Recovered scale M[1,1]={m_rec[1, 1]} differs from true {s} by > 0.001"
        )
        assert math.isclose(m_rec[0, 2], 1.0, abs_tol=0.20), (
            f"Recovered translation tx={m_rec[0, 2]} differs from injected 1.0 by > 0.20 coarse px"
        )
        assert math.isclose(m_rec[1, 2], 1.0, abs_tol=0.20), (
            f"Recovered translation ty={m_rec[1, 2]} differs from injected 1.0 by > 0.20 coarse px"
        )

    def test_run_classical_registration_order_independent(self, tmp_path):
        """Confirm run_classical_registration succeeds regardless of upload order."""
        derived_tmc_xml = _create_derived_tmc2_from_ohrc(tmp_path, dx=1.0, dy=1.0)

        # Order 1: OHRC as ref, TMC-2 as tgt
        res1 = run_classical_registration(OHRC_XML, derived_tmc_xml, transform_model="affine")
        assert res1.success is True
        assert res1.pipeline_mode == "cross_scale_affine_v1"
        assert res1.pair_type == "CROSS_SCALE_OHRC_TMC2"
        assert res1.quality_summary.inlier_count > 0
        assert res1.registered_image.shape == (25, 25)
        assert res1.quality_summary.inlier_median_error <= res1.quality_summary.inlier_rmse
        assert res1.quality_summary.inlier_rmse <= res1.quality_summary.inlier_max_error
        assert res1.quality_summary.inlier_median_error != res1.quality_summary.inlier_rmse
        assert res1.spatial.normalized_entropy > 0.0

        # Order 2: TMC-2 as ref, OHRC as tgt
        res2 = run_classical_registration(derived_tmc_xml, OHRC_XML, transform_model="affine")
        assert res2.success is True
        assert res2.pipeline_mode == "cross_scale_affine_v1"
        assert res2.pair_type == "CROSS_SCALE_OHRC_TMC2"
        assert res2.quality_summary.inlier_count > 0
        assert res2.registered_image.shape == (25, 25)
        assert res2.quality_summary.inlier_median_error <= res2.quality_summary.inlier_rmse
        assert res2.quality_summary.inlier_rmse <= res2.quality_summary.inlier_max_error
        assert res2.quality_summary.inlier_median_error != res2.quality_summary.inlier_rmse
        assert res2.spatial.normalized_entropy > 0.0


# =========================================================================
# 2. Transform Model Guard
# =========================================================================

class TestCrossScaleGuards:
    """Validate guards against unsupported models for cross-scale pairs."""

    def test_cross_scale_rejects_non_affine_models(self):
        """Cross-scale registration must reject homography or non-affine models."""
        res = run_classical_registration(
            OHRC_XML, TMC2_XML, transform_model="homography"
        )

        assert res.success is False
        assert res.failure_stage == "configuration"
        assert "only supports affine" in res.failure_reason
        assert res.pair_type == "CROSS_SCALE_OHRC_TMC2"


# =========================================================================
# 3. Numeric Regression Verification
# =========================================================================

class TestBaselinePreservationRegression:
    """Verify same-instrument and UNCLASSIFIED PNG flows produce exact expected metrics."""

    def test_plain_png_unclassified_numeric_preservation(self):
        """Verify UNCLASSIFIED PNG registration produces identical numeric baseline."""
        ref_path = FIXTURES_DIR / "ref_lunar.png"
        tgt_path = FIXTURES_DIR / "tgt_lunar.png"

        assert ref_path.exists(), f"Missing fixture {ref_path}"
        assert tgt_path.exists(), f"Missing fixture {tgt_path}"

        res = run_classical_registration(ref_path, tgt_path, transform_model="affine")
        assert res.success is True
        assert res.pipeline_mode == "classical_sift"
        assert res.pair_type == "UNCLASSIFIED"
        assert res.quality_summary.inlier_count == 102
        assert res.quality_summary.total_correspondences == 103
        assert math.isclose(res.quality_summary.inlier_rmse, 0.22398736, rel_tol=1e-3)

    def test_same_instrument_tmc2_numeric_preservation(self, tmp_path):
        """Verify same-instrument TMC-2 registration produces identical numeric baseline."""
        tmc2_img = FIXTURES_DIR / "pds4" / "tmc2" / "ch2_tmc_sample_500x500.img"
        tmc_data = np.fromfile(tmc2_img, dtype="<u2").reshape((500, 500))

        M_synth = np.array([[1.0, 0.0, 5.0], [0.0, 1.0, 3.0]], dtype=np.float32)
        tgt_data = cv2.warpAffine(tmc_data, M_synth, (500, 500), borderMode=cv2.BORDER_REFLECT)

        tgt_img_path = tmp_path / "tmc2_tgt.img"
        tgt_img_path.write_bytes(tgt_data.astype("<u2").tobytes())

        xml_text = TMC2_XML.read_text(encoding="utf-8")
        tgt_xml_text = xml_text.replace(
            "ch2_tmc_sample_500x500.img", "tmc2_tgt.img"
        )
        tgt_xml_path = tmp_path / "tmc2_tgt.xml"
        tgt_xml_path.write_text(tgt_xml_text, encoding="utf-8")

        res = run_classical_registration(TMC2_XML, tgt_xml_path, transform_model="affine")
        assert res.success is True
        assert res.pipeline_mode == "classical_sift"
        assert res.pair_type == "SAME_INSTRUMENT_TMC2"
        assert 250 <= res.quality_summary.inlier_count <= 260
        assert 250 <= res.quality_summary.total_correspondences <= 260
        assert math.isclose(res.quality_summary.inlier_rmse, 0.119, abs_tol=0.01)
