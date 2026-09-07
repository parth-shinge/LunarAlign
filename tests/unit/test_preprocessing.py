"""
SIH26166 — Tests for the preprocessing layer.

All fixtures are tiny synthetic images created in-memory or as temp files.
No real Chandrayaan imagery is used.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# Helpers — create temp image files
# ---------------------------------------------------------------------------


def _save_png(arr: np.ndarray, path: Path, mode: str | None = None) -> Path:
    """Save a NumPy array as a PNG via Pillow."""
    if mode is None:
        if arr.ndim == 2:
            mode = "L"
        elif arr.ndim == 3 and arr.shape[2] == 3:
            mode = "RGB"
        elif arr.ndim == 3 and arr.shape[2] == 4:
            mode = "RGBA"
        else:
            mode = "L"
    img = PILImage.fromarray(arr, mode=mode)
    img.save(path)
    return path


def _save_jpeg(arr: np.ndarray, path: Path) -> Path:
    """Save a NumPy array as JPEG (must be RGB uint8)."""
    img = PILImage.fromarray(arr, mode="RGB")
    img.save(path, format="JPEG")
    return path


def _save_tiff_pillow(arr: np.ndarray, path: Path, mode: str = "L") -> Path:
    """Save a NumPy array as TIFF via Pillow."""
    img = PILImage.fromarray(arr, mode=mode)
    img.save(path, format="TIFF")
    return path


def _save_multiband_tiff(arr: np.ndarray, path: Path) -> Path:
    """Save a multi-band array as GeoTIFF via rasterio.

    arr shape: (H, W, C) — bands-last.
    """
    import rasterio
    from rasterio.transform import from_bounds

    h, w, bands = arr.shape
    transform = from_bounds(0, 0, w, h, w, h)
    with rasterio.open(
        path, "w",
        driver="GTiff",
        height=h, width=w,
        count=bands,
        dtype=arr.dtype,
        transform=transform,
    ) as dst:
        for b in range(bands):
            dst.write(arr[:, :, b], b + 1)
    return path


# ---------------------------------------------------------------------------
# IMAGE I/O TESTS
# ---------------------------------------------------------------------------

class TestImageIO:
    """Tests for backend.preprocessing.io.load_image."""

    def test_load_png_rgb(self, tmp_path):
        """Load a small RGB PNG."""
        from backend.preprocessing.io import load_image

        arr = np.random.randint(0, 255, (8, 12, 3), dtype=np.uint8)
        path = _save_png(arr, tmp_path / "test.png", mode="RGB")
        raw = load_image(path)

        assert raw.width == 12
        assert raw.height == 8
        assert raw.num_bands == 3
        assert raw.dtype == np.uint8
        assert raw.source_format == "png"
        assert raw.data.shape == (8, 12, 3)
        np.testing.assert_array_equal(raw.data, arr)

    def test_load_png_grayscale(self, tmp_path):
        """Load a grayscale PNG."""
        from backend.preprocessing.io import load_image

        arr = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
        path = _save_png(arr, tmp_path / "gray.png", mode="L")
        raw = load_image(path)

        assert raw.width == 10
        assert raw.height == 10
        assert raw.num_bands == 1
        assert raw.data.ndim == 2

    def test_load_jpeg(self, tmp_path):
        """Load a small JPEG image."""
        from backend.preprocessing.io import load_image

        arr = np.random.randint(0, 255, (16, 20, 3), dtype=np.uint8)
        path = _save_jpeg(arr, tmp_path / "test.jpg")
        raw = load_image(path)

        assert raw.width == 20
        assert raw.height == 16
        assert raw.num_bands == 3
        assert raw.source_format == "jpeg"
        # JPEG is lossy so we cannot assert pixel equality

    def test_load_tiff_pillow(self, tmp_path):
        """Load a simple single-band TIFF via Pillow fallback."""
        from backend.preprocessing.io import load_image

        arr = np.random.randint(0, 255, (6, 8), dtype=np.uint8)
        path = _save_tiff_pillow(arr, tmp_path / "test.tif")
        raw = load_image(path)

        assert raw.width == 8
        assert raw.height == 6
        assert raw.num_bands == 1
        assert raw.data.ndim == 2

    def test_load_tiff_rgb(self, tmp_path):
        """Load a 3-band RGB TIFF."""
        from backend.preprocessing.io import load_image

        arr = np.random.randint(0, 255, (10, 12, 3), dtype=np.uint8)
        path = _save_tiff_pillow(arr, tmp_path / "rgb.tif", mode="RGB")
        raw = load_image(path)

        assert raw.width == 12
        assert raw.height == 10
        assert raw.num_bands == 3

    def test_load_multiband_tiff_rasterio(self, tmp_path):
        """Load a 5-band GeoTIFF via rasterio."""
        from backend.preprocessing.io import load_image

        arr = np.random.randint(0, 1000, (8, 10, 5), dtype=np.uint16)
        path = _save_multiband_tiff(arr, tmp_path / "multi.tif")
        raw = load_image(path)

        assert raw.width == 10
        assert raw.height == 8
        assert raw.num_bands == 5
        assert raw.dtype == np.uint16
        assert raw.source_format == "rasterio"
        assert raw.data.shape == (8, 10, 5)
        np.testing.assert_array_equal(raw.data, arr)

    def test_load_tiff_preserves_uint16(self, tmp_path):
        """dtype should be preserved, not silently cast to uint8."""
        from backend.preprocessing.io import load_image

        arr = np.random.randint(0, 65535, (4, 4, 3), dtype=np.uint16)
        path = _save_multiband_tiff(arr, tmp_path / "u16.tif")
        raw = load_image(path)

        assert raw.dtype == np.uint16

    def test_load_nonexistent_file(self):
        """Loading a nonexistent file should raise FileNotFoundError."""
        from backend.preprocessing.io import load_image

        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/image.png")

    def test_load_unsupported_format(self, tmp_path):
        """Loading an unsupported format should raise ValueError."""
        from backend.preprocessing.io import load_image

        path = tmp_path / "test.bmp"
        path.write_bytes(b"\x00" * 100)
        with pytest.raises(ValueError, match="Unsupported"):
            load_image(path)

    def test_load_corrupted_png(self, tmp_path):
        """Loading a corrupted file should raise ValueError."""
        from backend.preprocessing.io import load_image

        path = tmp_path / "corrupt.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50)
        with pytest.raises(ValueError):
            load_image(path)

    def test_dimensions_correct(self, tmp_path):
        """Width corresponds to columns, height to rows."""
        from backend.preprocessing.io import load_image

        # Deliberately non-square: 5 rows x 13 cols
        arr = np.zeros((5, 13, 3), dtype=np.uint8)
        path = _save_png(arr, tmp_path / "rect.png", mode="RGB")
        raw = load_image(path)

        assert raw.height == 5
        assert raw.width == 13
        assert raw.data.shape == (5, 13, 3)


# ---------------------------------------------------------------------------
# GRAYSCALE CONVERSION TESTS
# ---------------------------------------------------------------------------

class TestGrayscaleConversion:
    """Tests for backend.preprocessing.grayscale.to_feature_image."""

    def test_rgb_to_gray(self):
        """RGB image should become a 2-D float32 array."""
        from backend.preprocessing.datamodel import RawImage
        from backend.preprocessing.grayscale import to_feature_image

        data = np.random.randint(0, 255, (10, 12, 3), dtype=np.uint8)
        raw = RawImage(data=data, width=12, height=10, num_bands=3,
                       dtype=np.dtype("uint8"), source_format="png")

        feat = to_feature_image(raw)

        assert feat.data.ndim == 2
        assert feat.data.shape == (10, 12)
        assert feat.data.dtype == np.float32
        assert feat.width == 12
        assert feat.height == 10
        assert feat.conversion_method == "luminance_bt709"

    def test_grayscale_remains_2d(self):
        """Single-band image should pass through as 2-D float32."""
        from backend.preprocessing.datamodel import RawImage
        from backend.preprocessing.grayscale import to_feature_image

        data = np.random.randint(0, 255, (8, 8), dtype=np.uint8)
        raw = RawImage(data=data, width=8, height=8, num_bands=1,
                       dtype=np.dtype("uint8"), source_format="tiff")

        feat = to_feature_image(raw)

        assert feat.data.ndim == 2
        assert feat.data.shape == (8, 8)
        assert feat.conversion_method == "passthrough"

    def test_rgba_drops_alpha(self):
        """RGBA should drop alpha and convert RGB to grayscale."""
        from backend.preprocessing.datamodel import RawImage
        from backend.preprocessing.grayscale import to_feature_image

        data = np.random.randint(0, 255, (6, 6, 4), dtype=np.uint8)
        raw = RawImage(data=data, width=6, height=6, num_bands=4,
                       dtype=np.dtype("uint8"), source_format="png")

        feat = to_feature_image(raw)

        assert feat.data.ndim == 2
        assert feat.data.shape == (6, 6)
        assert "alpha_dropped" in feat.conversion_method

    def test_multiband_to_2d(self):
        """Multi-band (>4) should use band mean → 2-D."""
        from backend.preprocessing.datamodel import RawImage
        from backend.preprocessing.grayscale import to_feature_image

        data = np.random.randint(0, 1000, (4, 6, 7), dtype=np.uint16)
        raw = RawImage(data=data, width=6, height=4, num_bands=7,
                       dtype=np.dtype("uint16"), source_format="rasterio")

        feat = to_feature_image(raw)

        assert feat.data.ndim == 2
        assert feat.data.shape == (4, 6)
        assert feat.data.dtype == np.float32
        assert "band_mean" in feat.conversion_method

    def test_bt709_luminance_correctness(self):
        """BT.709 weights should be applied correctly."""
        from backend.preprocessing.datamodel import RawImage
        from backend.preprocessing.grayscale import to_feature_image

        # Pure red pixel
        data = np.array([[[255, 0, 0]]], dtype=np.uint8)
        raw = RawImage(data=data, width=1, height=1, num_bands=3,
                       dtype=np.dtype("uint8"), source_format="png")
        feat = to_feature_image(raw)
        expected = 0.2126 * 255
        np.testing.assert_allclose(feat.data[0, 0], expected, atol=0.5)

    def test_original_data_not_modified(self):
        """Conversion should not mutate the raw image data."""
        from backend.preprocessing.datamodel import RawImage
        from backend.preprocessing.grayscale import to_feature_image

        data = np.random.randint(0, 255, (4, 4, 3), dtype=np.uint8)
        original = data.copy()
        raw = RawImage(data=data, width=4, height=4, num_bands=3,
                       dtype=np.dtype("uint8"), source_format="png")

        to_feature_image(raw)

        np.testing.assert_array_equal(raw.data, original)


# ---------------------------------------------------------------------------
# NORMALIZATION TESTS
# ---------------------------------------------------------------------------

class TestNormalization:

    def test_minmax_normal(self):
        """Min-max should map to [0, 1]."""
        from backend.preprocessing.normalize import normalize_minmax

        arr = np.array([10, 50, 100, 200], dtype=np.float32)
        out = normalize_minmax(arr)

        assert out.dtype == np.float32
        np.testing.assert_allclose(out.min(), 0.0)
        np.testing.assert_allclose(out.max(), 1.0)

    def test_minmax_constant_image(self):
        """Constant image should return array of out_range[0]."""
        from backend.preprocessing.normalize import normalize_minmax

        arr = np.full((4, 4), 42.0, dtype=np.float32)
        out = normalize_minmax(arr)

        np.testing.assert_array_equal(out, np.zeros((4, 4), dtype=np.float32))

    def test_minmax_custom_range(self):
        """Custom output range should be respected."""
        from backend.preprocessing.normalize import normalize_minmax

        arr = np.array([0, 255], dtype=np.float32)
        out = normalize_minmax(arr, out_range=(0.0, 255.0))

        np.testing.assert_allclose(out[0], 0.0)
        np.testing.assert_allclose(out[1], 255.0)

    def test_minmax_preserves_original(self):
        """Original array should not be modified."""
        from backend.preprocessing.normalize import normalize_minmax

        arr = np.array([10, 50, 200], dtype=np.float32)
        original = arr.copy()
        normalize_minmax(arr)
        np.testing.assert_array_equal(arr, original)

    def test_percentile_normal(self):
        """Percentile normalization should clip outliers."""
        from backend.preprocessing.normalize import normalize_percentile

        arr = np.array([0, 1, 2, 3, 4, 5, 100], dtype=np.float32)
        out = normalize_percentile(arr, low_pct=0, high_pct=80)

        # Value 100 should be clipped
        np.testing.assert_allclose(out.max(), 1.0, atol=0.05)

    def test_percentile_constant(self):
        """Constant image with percentile norm should return fill value."""
        from backend.preprocessing.normalize import normalize_percentile

        arr = np.full((3, 3), 50.0, dtype=np.float32)
        out = normalize_percentile(arr)

        np.testing.assert_array_equal(out, np.zeros((3, 3), dtype=np.float32))

    def test_percentile_preserves_original(self):
        """Original array should not be modified."""
        from backend.preprocessing.normalize import normalize_percentile

        arr = np.array([0, 50, 100, 200, 255], dtype=np.float32)
        original = arr.copy()
        normalize_percentile(arr)
        np.testing.assert_array_equal(arr, original)

    def test_minmax_2d_image(self):
        """Works on 2-D arrays (typical feature image shape)."""
        from backend.preprocessing.normalize import normalize_minmax

        arr = np.random.rand(10, 10).astype(np.float32) * 1000
        out = normalize_minmax(arr)

        assert out.shape == (10, 10)
        np.testing.assert_allclose(out.min(), 0.0, atol=1e-6)
        np.testing.assert_allclose(out.max(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# CLAHE TESTS
# ---------------------------------------------------------------------------

class TestCLAHE:

    def test_output_dimensions_unchanged(self):
        """CLAHE should not change image dimensions."""
        from backend.preprocessing.datamodel import FeatureImage
        from backend.preprocessing.clahe import apply_clahe

        data = np.random.rand(32, 48).astype(np.float32) * 255
        feat = FeatureImage(data=data, width=48, height=32,
                            source_dtype=np.dtype("uint8"),
                            conversion_method="passthrough")

        result = apply_clahe(feat)

        assert result.data.shape == (32, 48)
        assert result.width == 48
        assert result.height == 32

    def test_output_is_valid_numeric(self):
        """CLAHE output should be finite float32."""
        from backend.preprocessing.datamodel import FeatureImage
        from backend.preprocessing.clahe import apply_clahe

        data = np.random.rand(16, 16).astype(np.float32) * 200
        feat = FeatureImage(data=data, width=16, height=16,
                            source_dtype=np.dtype("uint8"),
                            conversion_method="passthrough")

        result = apply_clahe(feat)

        assert result.data.dtype == np.float32
        assert np.all(np.isfinite(result.data))

    def test_does_not_modify_original(self):
        """CLAHE should not modify the input FeatureImage data."""
        from backend.preprocessing.datamodel import FeatureImage
        from backend.preprocessing.clahe import apply_clahe

        data = np.random.rand(16, 16).astype(np.float32) * 200
        original = data.copy()
        feat = FeatureImage(data=data, width=16, height=16,
                            source_dtype=np.dtype("uint8"),
                            conversion_method="passthrough")

        apply_clahe(feat)

        np.testing.assert_array_equal(feat.data, original)

    def test_constant_image_is_noop(self):
        """CLAHE on a constant image should not crash."""
        from backend.preprocessing.datamodel import FeatureImage
        from backend.preprocessing.clahe import apply_clahe

        data = np.full((16, 16), 100.0, dtype=np.float32)
        feat = FeatureImage(data=data, width=16, height=16,
                            source_dtype=np.dtype("uint8"),
                            conversion_method="passthrough")

        result = apply_clahe(feat)

        assert result.data.shape == (16, 16)

    def test_conversion_method_tracked(self):
        """Conversion method should include '+clahe' suffix."""
        from backend.preprocessing.datamodel import FeatureImage
        from backend.preprocessing.clahe import apply_clahe

        data = np.random.rand(16, 16).astype(np.float32) * 200
        feat = FeatureImage(data=data, width=16, height=16,
                            source_dtype=np.dtype("uint8"),
                            conversion_method="luminance_bt709")

        result = apply_clahe(feat)

        assert result.conversion_method == "luminance_bt709+clahe"


# ---------------------------------------------------------------------------
# METADATA EXTRACTION TESTS
# ---------------------------------------------------------------------------

class TestMetadataExtraction:

    def test_png_metadata(self, tmp_path):
        """Extract metadata from a PNG file."""
        from backend.preprocessing.metadata import extract_image_metadata

        arr = np.zeros((20, 30, 3), dtype=np.uint8)
        path = _save_png(arr, tmp_path / "test.png", mode="RGB")
        meta = extract_image_metadata(path)

        assert meta["width"] == 30
        assert meta["height"] == 20
        assert meta["num_bands"] == 3
        assert meta["image_dtype"] == "uint8"

    def test_grayscale_png_metadata(self, tmp_path):
        """Grayscale PNG should report 1 band."""
        from backend.preprocessing.metadata import extract_image_metadata

        arr = np.zeros((10, 10), dtype=np.uint8)
        path = _save_png(arr, tmp_path / "gray.png", mode="L")
        meta = extract_image_metadata(path)

        assert meta["num_bands"] == 1
        assert meta["image_dtype"] == "uint8"

    def test_multiband_tiff_metadata(self, tmp_path):
        """Multi-band TIFF metadata via rasterio."""
        from backend.preprocessing.metadata import extract_image_metadata

        arr = np.zeros((4, 6, 5), dtype=np.uint16)
        path = _save_multiband_tiff(arr, tmp_path / "multi.tif")
        meta = extract_image_metadata(path)

        assert meta["width"] == 6
        assert meta["height"] == 4
        assert meta["num_bands"] == 5
        assert meta["image_dtype"] == "uint16"


# ---------------------------------------------------------------------------
# END-TO-END: load → convert → normalize → clahe
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_full_pipeline(self, tmp_path):
        """Load an image, convert to feature image, normalize, apply CLAHE."""
        from backend.preprocessing.io import load_image
        from backend.preprocessing.grayscale import to_feature_image
        from backend.preprocessing.normalize import normalize_minmax
        from backend.preprocessing.clahe import apply_clahe
        from backend.preprocessing.datamodel import FeatureImage

        # Create a non-trivial RGB image
        arr = np.random.randint(10, 240, (32, 48, 3), dtype=np.uint8)
        path = _save_png(arr, tmp_path / "e2e.png", mode="RGB")

        raw = load_image(path)
        assert raw.data.shape == (32, 48, 3)

        feat = to_feature_image(raw)
        assert feat.data.shape == (32, 48)
        assert feat.data.dtype == np.float32

        normed = normalize_minmax(feat.data)
        assert normed.min() >= 0.0
        assert normed.max() <= 1.0

        # CLAHE needs values in a reasonable range
        feat_for_clahe = FeatureImage(
            data=feat.data,
            width=feat.width, height=feat.height,
            source_dtype=feat.source_dtype,
            conversion_method=feat.conversion_method,
        )
        clahe_result = apply_clahe(feat_for_clahe)
        assert clahe_result.data.shape == (32, 48)
        assert np.all(np.isfinite(clahe_result.data))


# ---------------------------------------------------------------------------
# STAGE 2B PREPROCESSING PIPELINE TESTS (USING REAL FIXTURES)
# ---------------------------------------------------------------------------

PDS4_FIXTURES = Path("tests/fixtures/pds4")
TMC2_FIXTURE_XML = PDS4_FIXTURES / "tmc2" / "ch2_tmc_sample_500x500.xml"
OHRC_FIXTURE_XML = PDS4_FIXTURES / "ohrc" / "ch2_ohrc_sample_500x500.xml"
IIRS_FIXTURE_XML = PDS4_FIXTURES / "iirs" / "ch2_iirs_sample_200x200x16.xml"


class TestPDS4LoadImage:
    """Tests for PDS4 XML loading through backend.preprocessing.io.load_image."""

    def test_load_pds4_iirs_valid_2d(self):
        """Loading real IIRS PDS4 XML through load_image produces a valid 2D single-channel image."""
        from backend.preprocessing.io import load_image
        from backend.preprocessing.datamodel import RawImage

        raw = load_image(IIRS_FIXTURE_XML)
        assert isinstance(raw, RawImage)
        assert raw.data.ndim == 2, f"Expected 2D image array, got {raw.data.ndim}D shape {raw.data.shape}"
        assert raw.data.shape == (200, 200), f"Expected shape (200, 200), got {raw.data.shape}"
        assert raw.num_bands == 1, f"Expected num_bands == 1, got {raw.num_bands}"
        assert raw.width == 200
        assert raw.height == 200
        assert raw.dtype == np.float32, f"Expected float32 dtype, got {raw.dtype}"
        assert raw.data.dtype == np.float32
        assert raw.source_format == "PDS4"
        assert raw.metadata.get("instrument") == "IIRS"
        assert raw.metadata.get("band_reduction_method") == "mean"
        assert "solar_reflective_bands" in raw.metadata

        min_val = float(raw.data.min())
        max_val = float(raw.data.max())
        assert min_val >= 0.0, f"Expected min >= 0.0, got {min_val}"
        assert max_val <= 255.0, f"Expected max <= 255.0, got {max_val}"
        assert max_val > min_val, f"Expected non-trivial dynamic range, got min={min_val}, max={max_val}"

    def test_load_pds4_unsupported_instrument_raises(self, tmp_path):
        """PDS4 label with unsupported instrument raises ValueError from load_image."""
        from backend.preprocessing.io import load_image
        import re

        txt = TMC2_FIXTURE_XML.read_text(encoding="utf-8")
        bad_txt = re.sub(r"<name>terrain mapping camera</name>", "<name>unknown sensor</name>", txt, flags=re.IGNORECASE)
        bad_xml = tmp_path / "unknown_inst.xml"
        bad_xml.write_text(bad_txt, encoding="utf-8")

        with pytest.raises(ValueError, match="Unsupported PDS4 instrument: 'UNKNOWN'"):
            load_image(bad_xml)

    def test_load_pds4_iirs_missing_wavelength_metadata_raises(self, tmp_path):
        """IIRS product with no Band_Bin metadata must raise ValueError rather than defaulting to all bands."""
        from backend.preprocessing.io import load_image
        import re
        import shutil

        # Copy binary qub file next to temporary XML so raster loading succeeds
        iirs_qub = PDS4_FIXTURES / "iirs" / "ch2_iirs_sample_200x200x16.qub"
        shutil.copyfile(iirs_qub, tmp_path / iirs_qub.name)

        txt = IIRS_FIXTURE_XML.read_text(encoding="utf-8")
        bad_txt = re.sub(r"<Band_Bin_Set>.*?</Band_Bin_Set>", "", txt, flags=re.DOTALL)
        bad_xml = tmp_path / "no_band_bins.xml"
        bad_xml.write_text(bad_txt, encoding="utf-8")

        with pytest.raises(
            ValueError,
            match="has no Band_Bin wavelength metadata; cannot determine solar-reflective bands",
        ):
            load_image(bad_xml)


class TestNormalize:
    """Tests for backend.preprocessing.normalize (Module 01)."""

    def test_clahe_uint16_tmc2(self):
        """CLAHE on real uint16 TMC-2 fixture -> float32 in [0, 1]."""
        from backend.preprocessing.pds4 import load_pds4_raster
        from backend.preprocessing.normalize import normalize_intensity

        raw = load_pds4_raster(TMC2_FIXTURE_XML)
        assert raw.dtype == np.uint16
        out = normalize_intensity(raw.data, method="clahe")

        assert out.shape == raw.data.shape
        assert out.dtype == np.float32
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0
        assert float(out.max()) > float(out.min())

    def test_clahe_uint8_ohrc(self):
        """CLAHE on real uint8 OHRC fixture -> float32 in [0, 1]."""
        from backend.preprocessing.pds4 import load_pds4_raster
        from backend.preprocessing.normalize import normalize_intensity

        raw = load_pds4_raster(OHRC_FIXTURE_XML)
        assert raw.dtype == np.uint8
        out = normalize_intensity(raw.data, method="clahe")

        assert out.shape == raw.data.shape
        assert out.dtype == np.float32
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_histogram_eq_method(self):
        """Global histogram equalization method works and returns float32 [0, 1]."""
        from backend.preprocessing.pds4 import load_pds4_raster
        from backend.preprocessing.normalize import normalize_intensity

        raw = load_pds4_raster(TMC2_FIXTURE_XML)
        out = normalize_intensity(raw.data, method="histogram_eq")

        assert out.shape == raw.data.shape
        assert out.dtype == np.float32
        assert float(out.min()) >= 0.0
        assert float(out.max()) <= 1.0

    def test_minmax_method(self):
        """Minmax method maps array strictly to [0, 1]."""
        from backend.preprocessing.pds4 import load_pds4_raster
        from backend.preprocessing.normalize import normalize_intensity

        raw = load_pds4_raster(OHRC_FIXTURE_XML)
        out = normalize_intensity(raw.data, method="minmax")

        assert out.shape == raw.data.shape
        assert out.dtype == np.float32
        np.testing.assert_allclose(float(out.min()), 0.0, atol=1e-5)
        np.testing.assert_allclose(float(out.max()), 1.0, atol=1e-5)

    def test_match_histograms_tmc_ohrc(self):
        """Histogram matching between TMC-2 and OHRC real fixtures."""
        from backend.preprocessing.pds4 import load_pds4_raster
        from backend.preprocessing.normalize import match_histograms, normalize_intensity

        tmc_raw = load_pds4_raster(TMC2_FIXTURE_XML)
        ohrc_raw = load_pds4_raster(OHRC_FIXTURE_XML)

        src = normalize_intensity(ohrc_raw.data, method="minmax")
        ref = normalize_intensity(tmc_raw.data, method="minmax")

        matched = match_histograms(src, ref)

        assert matched.shape == src.shape
        assert matched.dtype == np.float32
        assert np.all(np.isfinite(matched))

    def test_output_shape_preserved(self):
        """Normalization preserves arbitrary non-square 2D shape."""
        from backend.preprocessing.normalize import normalize_intensity

        arr = np.random.randint(0, 1000, (37, 89), dtype=np.uint16)
        out = normalize_intensity(arr, method="clahe")
        assert out.shape == (37, 89)

    def test_nan_inf_handling(self):
        """Arrays containing NaNs and Infs do not crash normalization."""
        from backend.preprocessing.normalize import normalize_intensity

        arr = np.array([[np.nan, 10.0], [np.inf, -np.inf]], dtype=np.float32)
        out = normalize_intensity(arr, method="minmax")
        assert out.dtype == np.float32
        assert np.all(np.isfinite(out))

    def test_unsupported_method_raises(self):
        """Invalid normalization method name raises ValueError."""
        from backend.preprocessing.normalize import normalize_intensity

        arr = np.zeros((10, 10), dtype=np.uint8)
        with pytest.raises(ValueError, match="Unsupported normalization method"):
            normalize_intensity(arr, method="invalid_method")


class TestBandReduction:
    """Tests for backend.preprocessing.band_reduction (Module 02)."""

    def test_pca_iirs_16band_fixture(self):
        """PCA on real IIRS 16-band fixture reduces to (n_components, H, W)."""
        from backend.preprocessing.pds4 import load_pds4_raster
        from backend.preprocessing.band_reduction import reduce_bands_pca

        raw = load_pds4_raster(IIRS_FIXTURE_XML)
        # RawImage.data is (200, 200, 16) bands-last or (16, 200, 200) BSQ
        data = raw.data
        if data.ndim == 3 and data.shape[2] == 16:
            cube = np.moveaxis(data, -1, 0)  # to (16, 200, 200)
        else:
            cube = data

        reduced = reduce_bands_pca(cube, n_components=3)

        assert reduced.shape == (3, 200, 200)
        assert reduced.dtype == np.float32
        assert np.all(np.isfinite(reduced))

    def test_pca_pc1_captures_structure(self):
        """PC1 has non-trivial variance across the spatial domain."""
        from backend.preprocessing.pds4 import load_pds4_raster
        from backend.preprocessing.band_reduction import reduce_bands_pca

        raw = load_pds4_raster(IIRS_FIXTURE_XML)
        data = raw.data
        cube = np.moveaxis(data, -1, 0) if data.ndim == 3 and data.shape[2] == 16 else data

        reduced = reduce_bands_pca(cube, n_components=1)
        pc1 = reduced[0]

        assert pc1.shape == (200, 200)
        assert np.var(pc1) > 0.0

    def test_mean_reduction(self):
        """Mean band reduction collapses 3D cube to 2D (H, W)."""
        from backend.preprocessing.pds4 import load_pds4_raster
        from backend.preprocessing.band_reduction import reduce_bands_mean

        raw = load_pds4_raster(IIRS_FIXTURE_XML)
        data = raw.data
        cube = np.moveaxis(data, -1, 0) if data.ndim == 3 and data.shape[2] == 16 else data

        mean_img = reduce_bands_mean(cube)

        assert mean_img.shape == (200, 200)
        assert mean_img.dtype == np.float32

    def test_select_bands_valid(self):
        """Select specific bands extracts requested subset."""
        from backend.preprocessing.pds4 import load_pds4_raster
        from backend.preprocessing.band_reduction import select_bands

        raw = load_pds4_raster(IIRS_FIXTURE_XML)
        data = raw.data
        cube = np.moveaxis(data, -1, 0) if data.ndim == 3 and data.shape[2] == 16 else data

        sub = select_bands(cube, [0, 3, 7])
        assert sub.shape == (3, 200, 200)
        assert sub.dtype == np.float32

    def test_select_bands_invalid_raises(self):
        """Selecting out-of-range band indices raises IndexError."""
        from backend.preprocessing.band_reduction import select_bands

        cube = np.zeros((5, 20, 20), dtype=np.float32)
        with pytest.raises(IndexError):
            select_bands(cube, [10])

    def test_single_band_input_passthrough(self):
        """2D input passes through PCA without crashing."""
        from backend.preprocessing.band_reduction import reduce_bands_pca, reduce_bands_mean

        arr = np.random.rand(50, 50).astype(np.float32)
        out_pca = reduce_bands_pca(arr, n_components=1)
        out_mean = reduce_bands_mean(arr)

        assert out_pca.shape == (1, 50, 50)
        assert out_mean.shape == (50, 50)


class TestScaleHandler:
    """Tests for backend.preprocessing.scale_handler (Module 06)."""

    def test_compute_scale_ratio_known_values(self):
        """OHRC (0.21 m/px) to TMC-2 (5.0 m/px) ratio is 0.042."""
        from backend.preprocessing.scale_handler import compute_scale_ratio

        ratio = compute_scale_ratio(0.21, 5.0)
        np.testing.assert_allclose(ratio, 0.042, atol=1e-4)

    def test_downsample_to_match_dimensions(self):
        """downsample_to_match correctly reduces image dimensions."""
        from backend.preprocessing.scale_handler import downsample_to_match

        img = np.ones((500, 500), dtype=np.float32)
        downsampled = downsample_to_match(img, 0.5)

        assert downsampled.shape == (250, 250)
        assert downsampled.dtype == np.float32

    def test_gaussian_pyramid_levels(self):
        """Pyramid generates requested number of halved levels."""
        from backend.preprocessing.scale_handler import build_gaussian_pyramid

        img = np.zeros((128, 128), dtype=np.float32)
        pyr = build_gaussian_pyramid(img, levels=4)

        assert len(pyr) == 4
        assert pyr[0].shape == (128, 128)
        assert pyr[1].shape == (64, 64)
        assert pyr[2].shape == (32, 32)
        assert pyr[3].shape == (16, 16)

    def test_scale_ratio_one_returns_unchanged(self):
        """Scale ratio >= 1.0 returns image with original dimensions."""
        from backend.preprocessing.scale_handler import downsample_to_match

        img = np.random.rand(64, 64).astype(np.float32)
        out = downsample_to_match(img, 1.0)

        assert out.shape == (64, 64)

    def test_get_resolution_known_defaults(self):
        """get_resolution returns canonical defaults for instruments."""
        from backend.preprocessing.scale_handler import get_resolution

        assert get_resolution(None, "OHRC") == 0.21
        assert get_resolution(None, "TMC-2") == 4.27
        assert get_resolution(None, "TMC2") == 4.27
        assert get_resolution(None, "TMC") == 4.27
        assert get_resolution(None, "IIRS") == 82.7

    def test_get_resolution_from_metadata(self):
        """get_resolution extracts pixel_resolution from PDS4 metadata dictionary."""
        from backend.preprocessing.scale_handler import get_resolution

        meta = {
            "pds4_isda_product_params": {
                "pixel_resolution": {"value": "4.27", "unit": "m/pixel"}
            }
        }
        res = get_resolution(meta, "TMC2")
        assert res == 4.27


class TestPhaseCongruency:
    """Tests for backend.preprocessing.illumination (Module 03)."""

    def test_output_shape_matches_input(self):
        """Phase congruency output matches input shape."""
        from backend.preprocessing.illumination import compute_phase_congruency

        img = np.random.rand(40, 50).astype(np.float32)
        pc_map, ori_map = compute_phase_congruency(img, nscale=3, norient=4)

        assert pc_map.shape == (40, 50)
        assert ori_map.shape == (40, 50)
        assert pc_map.dtype == np.float32
        assert ori_map.dtype == np.float32

    def test_output_in_zero_one_range(self):
        """Phase congruency map is strictly bounded in [0, 1]."""
        from backend.preprocessing.illumination import compute_phase_congruency

        img = np.random.rand(48, 48).astype(np.float32)
        pc_map, _ = compute_phase_congruency(img)

        assert float(pc_map.min()) >= 0.0
        assert float(pc_map.max()) <= 1.0

    def test_step_edge_high_congruency(self):
        """Synthetic step edge produces high phase congruency at the boundary."""
        from backend.preprocessing.illumination import compute_edge_map

        img = np.zeros((64, 64), dtype=np.float32)
        img[:, 32:] = 1.0  # sharp vertical step edge at column 32

        edge_map = compute_edge_map(img, method="phase_congruency")

        # Edge vicinity (col 30..34) should have higher response than flat region (col 10..15)
        edge_response = float(np.mean(edge_map[:, 31:34]))
        flat_response = float(np.mean(edge_map[:, 10:15]))
        assert edge_response > flat_response

    def test_canny_fallback_method(self):
        """Canny edge map option works as fallback."""
        from backend.preprocessing.illumination import compute_edge_map

        img = np.zeros((64, 64), dtype=np.float32)
        img[:, 32:] = 1.0
        canny_map = compute_edge_map(img, method="canny")

        assert canny_map.shape == (64, 64)
        assert canny_map.dtype == np.float32
        assert float(canny_map.max()) > 0.0


class TestPreprocessPipeline:
    """Tests for backend.preprocessing.preprocess (Pipeline Orchestrator)."""

    def test_preprocess_single_tmc2(self):
        """preprocess_single processes TMC-2 fixture."""
        from backend.preprocessing.pds4 import load_pds4_raster, extract_pds4_metadata
        from backend.preprocessing.preprocess import preprocess_single

        raw = load_pds4_raster(TMC2_FIXTURE_XML)
        meta = extract_pds4_metadata(TMC2_FIXTURE_XML)
        prep = preprocess_single(raw.data, instrument="TMC-2", metadata=meta.__dict__)

        assert prep.data.shape == (500, 500)
        assert prep.data.dtype == np.float32
        assert prep.instrument == "TMC-2"
        assert "normalize_clahe" in prep.preprocessing_steps

    def test_preprocess_single_ohrc(self):
        """preprocess_single processes OHRC fixture."""
        from backend.preprocessing.pds4 import load_pds4_raster
        from backend.preprocessing.preprocess import preprocess_single

        raw = load_pds4_raster(OHRC_FIXTURE_XML)
        prep = preprocess_single(raw.data, instrument="OHRC")

        assert prep.data.shape == (500, 500)
        assert prep.data.dtype == np.float32
        assert prep.instrument == "OHRC"

    def test_preprocess_single_iirs_reduces_bands(self):
        """preprocess_single on 3D IIRS cube performs band reduction."""
        from backend.preprocessing.pds4 import load_pds4_raster
        from backend.preprocessing.preprocess import preprocess_single

        raw = load_pds4_raster(IIRS_FIXTURE_XML)
        prep = preprocess_single(raw.data, instrument="IIRS")

        assert prep.data.ndim == 2
        assert prep.data.shape == (200, 200)
        assert prep.data.dtype == np.float32
        assert "pca_band_reduction" in prep.preprocessing_steps

    def test_preprocess_pair_ohrc_tmc_resolution_matched(self):
        """preprocess_pair downsamples OHRC to match TMC-2 coarser resolution."""
        from backend.preprocessing.pds4 import load_pds4_raster, extract_pds4_metadata
        from backend.preprocessing.preprocess import preprocess_pair

        ohrc_raw = load_pds4_raster(OHRC_FIXTURE_XML)
        tmc_raw = load_pds4_raster(TMC2_FIXTURE_XML)
        ohrc_meta = extract_pds4_metadata(OHRC_FIXTURE_XML).__dict__
        tmc_meta = extract_pds4_metadata(TMC2_FIXTURE_XML).__dict__

        src_prep, ref_prep = preprocess_pair(
            source_raster=ohrc_raw.data,
            source_instrument="OHRC",
            source_meta=ohrc_meta,
            ref_raster=tmc_raw.data,
            ref_instrument="TMC-2",
            ref_meta=tmc_meta,
            apply_phase_congruency=True,
        )

        assert src_prep.pixel_resolution == ref_prep.pixel_resolution
        assert any("downsampled" in step for step in src_prep.preprocessing_steps)
        assert "phase_congruency" in src_prep.preprocessing_steps
        assert "phase_congruency" in ref_prep.preprocessing_steps

    def test_preprocess_steps_audit_trail(self):
        """Audit trail accurately records applied transformation steps."""
        from backend.preprocessing.preprocess import preprocess_single

        arr = np.random.rand(100, 100).astype(np.float32)
        prep = preprocess_single(
            arr,
            instrument="OHRC",
            target_resolution=5.0,  # Trigger downsampling from 0.21 -> 5.0
            normalize_method="minmax",
        )

        assert "normalize_minmax" in prep.preprocessing_steps
        assert any("downsampled" in s for s in prep.preprocessing_steps)

    def test_preprocess_pair_missing_ref_raises(self):
        """preprocess_pair raises ValueError if reference raster is None."""
        from backend.preprocessing.preprocess import preprocess_pair

        arr = np.ones((10, 10), dtype=np.float32)
        with pytest.raises(ValueError, match="ref_raster must be provided"):
            preprocess_pair(arr, "OHRC", {}, None, "TMC-2", {})


# ---------------------------------------------------------------------------
# UINT16 DYNAMIC-RANGE NORMALIZE FIX (TMC-2 → SIFT keypoint test)
# ---------------------------------------------------------------------------

class TestUint16DynamicRangeNormalizeFix:
    """Regression test for the uint16 contrast-destruction bug.

    Before the fix, to_feature_image() performed a raw float32 passthrough
    with no dynamic range stretch.  Then prepare_for_sift() would clip all
    values to [0, 255] via np.clip — which for TMC-2 uint16 data with
    pixel values in [187, 383] meant everything above 255 was flattened,
    destroying contrast and producing near-zero SIFT keypoints.

    The fix adds _normalize_dynamic_range() inside to_feature_image() to
    stretch out-of-range data to [0, 255] before SIFT consumption.
    """

    def test_tmc2_uint16_sift_keypoint_count(self):
        """Real TMC-2 uint16 fixture → to_feature_image → prepare_for_sift → SIFT detects ≥10 keypoints."""
        from backend.preprocessing.pds4 import load_pds4_raster
        from backend.preprocessing.datamodel import RawImage
        from backend.preprocessing.grayscale import to_feature_image
        from backend.features.sift import prepare_for_sift, extract_sift

        meta = load_pds4_raster(TMC2_FIXTURE_XML)
        assert meta.data.dtype == np.uint16, f"Expected uint16, got {meta.data.dtype}"

        # Confirm pixel values exceed uint8 range (the bug precondition)
        assert meta.data.max() > 255, (
            f"TMC-2 fixture max={meta.data.max()} — expected >255 for this test"
        )

        raw = RawImage(
            data=meta.data,
            width=meta.data.shape[1],
            height=meta.data.shape[0],
            num_bands=1,
            dtype=meta.data.dtype,
            source_format="pds4",
        )

        # Run the canonical pipeline path: to_feature_image → prepare_for_sift
        feat = to_feature_image(raw)

        # Verify dynamic range stretch occurred
        assert feat.data.min() >= 0.0
        assert feat.data.max() <= 255.0
        assert feat.data.max() > 200.0, (
            f"Stretch should use full range; max={feat.data.max():.1f}"
        )

        # Verify SIFT input has contrast (not clipped to uniform value)
        sift_input = prepare_for_sift(feat)
        assert sift_input.dtype == np.uint8
        unique_values = len(np.unique(sift_input))
        assert unique_values > 10, (
            f"Expected >10 unique uint8 values, got {unique_values} "
            f"(indicates clipping/contrast destruction)"
        )

        # Extract SIFT — the actual regression check
        sift_result = extract_sift(feat)
        assert sift_result.num_keypoints >= 10, (
            f"Expected ≥10 SIFT keypoints on 500×500 TMC-2, got "
            f"{sift_result.num_keypoints}. Pre-fix, this was near-zero "
            f"due to uint16 values being clipped to 255."
        )

