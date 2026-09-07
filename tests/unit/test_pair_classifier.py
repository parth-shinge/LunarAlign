"""SIH26166 — Unit tests for the pair-type classifier and routing scaffold."""

from __future__ import annotations

from pathlib import Path
import pytest

from backend.routing.pair_classifier import PairClassification, PairType, classify_pair

FIXTURES_PDS4 = Path("tests/fixtures/pds4")
TMC2_XML = FIXTURES_PDS4 / "tmc2" / "ch2_tmc_sample_500x500.xml"
OHRC_XML = FIXTURES_PDS4 / "ohrc" / "ch2_ohrc_sample_500x500.xml"
IIRS_XML = FIXTURES_PDS4 / "iirs" / "ch2_iirs_sample_200x200x16.xml"


class TestSameInstrumentClassification:
    """Validate all three same-instrument combinations against real PDS4 fixtures."""

    def test_same_instrument_tmc2(self):
        result = classify_pair(TMC2_XML, TMC2_XML)
        assert isinstance(result, PairClassification)
        assert result.pair_type == PairType.SAME_INSTRUMENT_TMC2
        assert result.instrument_a == "TMC2"
        assert result.instrument_b == "TMC2"
        assert result.is_same_instrument is True
        assert result.is_implemented is True
        assert result.recommended_pipeline_stage == "classical_sift_baseline"
        assert "synthetic same-instrument transforms" in result.reason

    def test_same_instrument_ohrc(self):
        result = classify_pair(OHRC_XML, OHRC_XML)
        assert isinstance(result, PairClassification)
        assert result.pair_type == PairType.SAME_INSTRUMENT_OHRC
        assert result.instrument_a == "OHRC"
        assert result.instrument_b == "OHRC"
        assert result.is_same_instrument is True
        assert result.is_implemented is True
        assert result.recommended_pipeline_stage == "classical_sift_baseline"
        assert "synthetic same-instrument transforms" in result.reason

    def test_same_instrument_iirs(self):
        result = classify_pair(IIRS_XML, IIRS_XML)
        assert isinstance(result, PairClassification)
        assert result.pair_type == PairType.SAME_INSTRUMENT_IIRS
        assert result.instrument_a == "IIRS"
        assert result.instrument_b == "IIRS"
        assert result.is_same_instrument is True
        assert result.is_implemented is True
        assert result.recommended_pipeline_stage == "classical_sift_baseline"
        assert "synthetic same-instrument transforms" in result.reason


class TestCrossInstrumentClassification:
    """Validate all three cross-instrument combinations and blocking capability reasons."""

    def test_cross_scale_ohrc_tmc2(self):
        result = classify_pair(OHRC_XML, TMC2_XML)
        assert result.pair_type == PairType.CROSS_SCALE_OHRC_TMC2
        assert result.instrument_a == "OHRC"
        assert result.instrument_b == "TMC2"
        assert result.is_same_instrument is False
        assert result.is_implemented is True
        assert result.recommended_pipeline_stage == "cross_scale_affine_v1"
        assert len(result.reason) > 0
        assert "scale" in result.reason.lower()

    def test_cross_modal_tmc2_iirs(self):
        result = classify_pair(TMC2_XML, IIRS_XML)
        assert result.pair_type == PairType.CROSS_MODAL_TMC2_IIRS
        assert result.instrument_a == "TMC2"
        assert result.instrument_b == "IIRS"
        assert result.is_same_instrument is False
        assert result.is_implemented is True
        assert result.recommended_pipeline_stage == "cross_modal_phase_congruency_v1"
        assert len(result.reason) > 0
        assert "phase congruency" in result.reason.lower()

    def test_cross_modal_extreme_scale_ohrc_iirs(self):
        result = classify_pair(OHRC_XML, IIRS_XML)
        assert result.pair_type == PairType.CROSS_MODAL_EXTREME_SCALE_OHRC_IIRS
        assert result.instrument_a == "OHRC"
        assert result.instrument_b == "IIRS"
        assert result.is_same_instrument is False
        assert result.is_implemented is False
        assert result.recommended_pipeline_stage == "not_implemented"
        assert len(result.reason) > 0
        assert "scale" in result.reason.lower()
        assert "82.70" in result.reason
        assert any(
            cap in result.reason for cap in ["Phase Congruency", "MIND", "RIFT", "cross-modal"]
        )


class TestOrderIndependence:
    """Validate that input argument ordering does not change the classification outcome."""

    @pytest.mark.parametrize(
        "path_a, path_b, expected_type",
        [
            (OHRC_XML, TMC2_XML, PairType.CROSS_SCALE_OHRC_TMC2),
            (TMC2_XML, IIRS_XML, PairType.CROSS_MODAL_TMC2_IIRS),
            (OHRC_XML, IIRS_XML, PairType.CROSS_MODAL_EXTREME_SCALE_OHRC_IIRS),
        ],
    )
    def test_order_invariance(self, path_a, path_b, expected_type):
        res_ab = classify_pair(path_a, path_b)
        res_ba = classify_pair(path_b, path_a)

        assert res_ab.pair_type == expected_type
        assert res_ba.pair_type == expected_type
        assert res_ab.pair_type == res_ba.pair_type
        assert res_ab.is_implemented == res_ba.is_implemented
        assert res_ab.is_same_instrument == res_ba.is_same_instrument
        assert res_ab.recommended_pipeline_stage == res_ba.recommended_pipeline_stage
        assert res_ab.reason == res_ba.reason

        # Positional instruments should reflect the arguments passed
        assert res_ab.instrument_a == res_ba.instrument_b
        assert res_ab.instrument_b == res_ba.instrument_a


class TestUnclassifiedFallback:
    """Validate UNCLASSIFIED category for non-PDS4 or unidentifiable inputs."""

    def test_plain_png_both_sides(self, tmp_path):
        png_a = tmp_path / "img_a.png"
        png_b = tmp_path / "img_b.png"
        png_a.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
        png_b.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

        res = classify_pair(png_a, png_b)
        assert res.pair_type == PairType.UNCLASSIFIED
        assert res.instrument_a is None
        assert res.instrument_b is None
        assert res.is_same_instrument is False
        assert res.is_implemented is True
        assert res.recommended_pipeline_stage == "classical_sift_baseline"
        assert "falls back" in res.reason.lower()

    def test_mixed_pds4_and_plain_image(self, tmp_path):
        png = tmp_path / "demo.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

        res_a = classify_pair(TMC2_XML, png)
        assert res_a.pair_type == PairType.UNCLASSIFIED
        assert res_a.instrument_a == "TMC2"
        assert res_a.instrument_b is None
        assert res_a.is_implemented is True
        assert res_a.recommended_pipeline_stage == "classical_sift_baseline"

        res_b = classify_pair(png, TMC2_XML)
        assert res_b.pair_type == PairType.UNCLASSIFIED
        assert res_b.instrument_a is None
        assert res_b.instrument_b == "TMC2"
        assert res_b.is_implemented is True
        assert res_b.recommended_pipeline_stage == "classical_sift_baseline"

    def test_nonexistent_files_do_not_raise(self):
        res = classify_pair("nonexistent_ref.png", "nonexistent_tgt.png")
        assert res.pair_type == PairType.UNCLASSIFIED
        assert res.instrument_a is None
        assert res.instrument_b is None
        assert res.is_implemented is True
        assert res.recommended_pipeline_stage == "classical_sift_baseline"

    def test_malformed_xml_does_not_raise(self, tmp_path):
        bad_xml = tmp_path / "corrupt.xml"
        bad_xml.write_text("<unclosed_tag", encoding="utf-8")

        res = classify_pair(bad_xml, TMC2_XML)
        assert res.pair_type == PairType.UNCLASSIFIED
        assert res.instrument_a is None
        assert res.instrument_b == "TMC2"
        assert res.is_implemented is True
