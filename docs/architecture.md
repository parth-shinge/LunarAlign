# Architecture

## Preprocessing Layer

The preprocessing layer (`backend/preprocessing/`) provides the
foundational image I/O and processing bridge between the upload/input
pipeline and the downstream feature-extraction / registration algorithms.

### Module Structure

```text
backend/preprocessing/
├── __init__.py       — package docstring and symbol exports
├── datamodel.py      — RawImage, FeatureImage, PDS4Metadata data structures
├── io.py             — image loading (PNG, JPEG, TIFF, PDS4 XML)
├── pds4.py           — PDS4 label parser and raster loader (TMC-2, OHRC, IIRS)
├── grayscale.py      — conversion to 2-D feature image with dynamic range normalization
├── normalize.py      — intensity normalization utilities
├── metadata.py       — lightweight header-only metadata extraction
└── clahe.py          — optional CLAHE contrast enhancement
```

### Internal Image Representation

Two distinct data structures are maintained:

| Structure      | Purpose                        | Shape            | dtype      |
|---------------|-------------------------------|------------------|------------|
| `RawImage`     | Full-fidelity loaded data      | `(H, W)` or `(H, W, C)` | preserved  |
| `FeatureImage` | 2-D single-channel for feature extraction | `(H, W)` | `float32` |

**RawImage** preserves all original bands, dtype, and source metadata.
No silent normalisation or dtype conversion is performed during loading.

**FeatureImage** is derived from a RawImage via an explicit conversion
step.  It is always 2-D float32 and is the representation consumed by
classical CV algorithms (SIFT, Phase Congruency, etc.).

### Image I/O

| Format     | Loader    | Notes                                    |
|-----------|-----------|------------------------------------------|
| PNG       | Pillow    | Standard RGB/RGBA/grayscale (`io.py:L71`) |
| JPEG      | Pillow    | Lossy; RGB uint8 (`io.py:L71`)           |
| TIFF      | rasterio → Pillow fallback | Supports multi-band GeoTIFF (`io.py:L69, L141-161`) |
| PDS4 XML  | rasterio GDAL PDS4 driver | TMC-2, OHRC, and IIRS (`io.py:L73-75, L85-185`) |

For TIFF files, rasterio is attempted first (accurate multi-band and
GeoTIFF support via `_load_rasterio` at `backend/preprocessing/io.py:L164-210`).
If rasterio fails, Pillow is used as a fallback for simple TIFF files.

For PDS4 `.xml` label inputs, `load_image()` (`backend/preprocessing/io.py:L42-76`)
delegates to `_load_pds4()` (`backend/preprocessing/io.py:L85-185`). The loader
validates that the file is a genuine PDS4 `Product_Observational` XML label via
`is_pds4_label()` (`backend/preprocessing/pds4.py:L112-125`), verifies the instrument
identity using `identify_instrument()` (`backend/preprocessing/pds4.py:L148-175`),
and accepts **TMC-2**, **OHRC**, and **IIRS** products (`io.py:L126-129`),
loading the pixel raster via `load_pds4_raster()` (`backend/preprocessing/pds4.py:L340-416`).
For **IIRS** hyperspectral products, `_load_pds4()` (`backend/preprocessing/io.py:L137-185`)
extracts per-band wavelength metadata (`extract_iirs_band_wavelengths`), selects solar-reflective
bands (< 2500 nm via `select_solar_reflective_bands`), subsets the spectral cube (`select_bands`),
and reduces the multi-band cube to a 2D float32 image via per-pixel mean reduction (`reduce_bands_mean`),
followed by dynamic range normalization (`_normalize_dynamic_range`).
Any non-PDS4 XML or unsupported instrument product raises a descriptive `ValueError`.

Multi-band rasters retain **all** bands in the RawImage.  No bands are
silently discarded.

### Grayscale / Feature Image Conversion

Conversion from `RawImage` to `FeatureImage` is performed by `to_feature_image()`
(`backend/preprocessing/grayscale.py:L33-90`):

| Input                  | Strategy                          | Method label           |
|-----------------------|----------------------------------|----------------------|
| Single-band (grayscale)| Passthrough to float32           | `passthrough` (`grayscale.py:L56-62`) |
| RGB (3-band)          | BT.709 luminance                 | `luminance_bt709` (`grayscale.py:L63-66`) |
| RGBA (4-band)         | Drop alpha → BT.709              | `luminance_bt709_alpha_dropped` (`grayscale.py:L67-71`) |
| Multi-band (>4)       | Per-pixel mean across bands      | `band_mean_Nbands` (`grayscale.py:L72-76`) |

#### Dynamic Range Normalization

Following channel reduction, `to_feature_image()` executes `_normalize_dynamic_range()`
(`backend/preprocessing/grayscale.py:L92-117`).
- When input pixel values exceed standard uint8 bounds (`lo < 0.0 or hi > 255.0`),
  such as raw uint16 TMC-2 radiance counts in `[187, 383]` (or up to 40,000+),
  a linear min-max stretch to `[0, 255]` is automatically applied (`grayscale.py:L108-113`).
  This ensures contrast is preserved and prevents OpenCV's downstream `prepare_for_sift()`
  (`backend/features/sift.py:L50-66`) from clipping all high-range values to uniform 255
  (which would result in near-zero keypoints).
- If pixel values already reside within `[0, 255]`, the array is preserved as-is without
  modification (`grayscale.py:L115-116`).

The multi-band mean strategy is a documented generic approach.  It will
be replaced by PCA, specific band selection, or domain-aware spectral
reduction when the IIRS pipeline is implemented.

### Normalization

Two opt-in normalization functions:

- **`normalize_minmax`** — linear stretch to `[0, 1]` (or custom range).
- **`normalize_percentile`** — robust stretch with percentile clipping
  (default 2nd–98th percentile) to reduce influence of outlier pixels.

Both operate on copies; the original data is never modified.
Normalisation is never automatic — the caller explicitly chooses when
and how to normalise.

### CLAHE

CLAHE (Contrast Limited Adaptive Histogram Equalisation) is available as
an **optional** preprocessing step via `apply_clahe()`.

- Operates only on `FeatureImage` (2-D float), never on raw data.
- Returns a new `FeatureImage`; the input is not modified.
- Default parameters: `clip_limit=2.0`, `tile_grid_size=(8, 8)`.
- Useful for lunar imagery with strong sun-angle / illumination
  variations.
- The conversion method string is appended with `+clahe` for
  provenance tracking.

### Upload → Preprocessing Boundary

The `/upload` endpoint remains responsible for:
- file validation
- storage
- **lightweight** metadata extraction (dimensions, bands, dtype via
  header-only reading)

Full pixel-data preprocessing (loading into RawImage, conversion,
normalisation, CLAHE) is **not** performed during upload.  It will be
triggered by the processing/registration pipeline in later stages.

### Metadata in API Response

The upload response now includes:

| Field         | Type       | Description                            |
|--------------|-----------|----------------------------------------|
| `width`       | `int?`    | Image width in pixels                  |
| `height`      | `int?`    | Image height in pixels                 |
| `num_bands`   | `int?`    | Number of spectral bands / channels    |
| `image_dtype` | `str?`    | NumPy dtype string (e.g. `"uint8"`)    |

These are populated via header-only reading at upload time.  Full
raster metadata (CRS, transform, band descriptions) is available in
the `RawImage.metadata` dict when the image is loaded for processing.

---

## Feature Extraction Layer

The feature extraction layer (`backend/features/`) provides keypoint
detection and descriptor computation.

### Module Structure

```text
backend/features/
├── __init__.py         — package docstring
├── models.py           — Keypoint, SIFTFeatures, match data structures
├── sift.py             — SIFT detection + descriptor extraction
└── visualization.py    — debug match drawing utility (optional)
```

### Data Flow

```text
FeatureImage (float32)
    → prepare_for_sift() → uint8 grayscale
    → cv2.SIFT_create().detectAndCompute()
    → SIFTFeatures (keypoints + descriptors)
```

The `FeatureImage` is explicitly converted to uint8 for OpenCV.
The original float32 data is never modified.

---

## Matching Layer

The matching layer (`backend/matching/`) provides descriptor matching
and correspondence filtering.

### Module Structure

```text
backend/matching/
├── __init__.py       — package docstring
├── flann.py          — FLANN kNN matching (KD-tree)
└── ratio_test.py     — Lowe distance ratio test
```

### Classical Baseline Pipeline

```text
SIFTFeatures (ref)  ─┐
                      ├─→ FLANN kNN match ─→ list[RawMatch]
SIFTFeatures (tgt)  ─┘                          │
                                                 ▼
                                    Lowe ratio test (d1/d2 < 0.75)
                                                 │
                                                 ▼
                                           MatchResult
                                    (FilteredMatch with ref_pt, tgt_pt)
```

### Coordinate Convention

All point coordinates throughout the project use **(x, y)** with origin
at the **top-left** corner of the image:

- `x` = column index (increases rightward)
- `y` = row index (increases downward)

This matches OpenCV's `cv2.KeyPoint.pt` convention.

### API Boundary

The feature/matching/geometry/registration/evaluation modules are
**internal services** orchestrated by the registration service.
The `/register` endpoint exposes the classical pipeline.
The `/upload` endpoint remains unchanged.

---

## Geometry Layer

The geometry layer (`backend/geometry/`) provides geometric verification,
transformation estimation, and point transformation utilities.

### Module Structure

```text
backend/geometry/
├── __init__.py       — package docstring
├── models.py          — GeometricResult, ErrorMetrics, TransformModel, etc.
├── estimation.py      — robust estimation (MAGSAC++ / USAC / RANSAC)
└── transform.py       — point transformation utilities
```

### Classical Registration Baseline Pipeline

```text
SIFTFeatures (ref)  ─┐
                      ├─→ FLANN kNN match ─→ Lowe ratio test
SIFTFeatures (tgt)  ─┘                          │
                                                 ▼
                                          MatchResult
                                    (list[FilteredMatch])
                                                 │
                                                 ▼
                                     estimate_transform()
                                    MAGSAC++ / RANSAC fallback
                                                 │
                                                 ▼
                                        GeometricResult
                              (transform_matrix, inlier_mask,
                               error_metrics, estimator_method)
```

### Estimator Selection

The module detects OpenCV's USAC capabilities at runtime:
- If `cv2.USAC_MAGSAC` is available → uses MAGSAC++
- Otherwise → falls back to `cv2.RANSAC`

The actual method used is always recorded in `GeometricResult.estimator_method`.

### Data Separation

Geometry is kept separate from feature extraction and matching:
- Input: `list[FilteredMatch]` (from matching layer)
- Output: `GeometricResult` (self-contained result)
- No direct dependency on SIFT internals or FLANN

---

## Registration Layer

The registration layer (`backend/registration/`) produces the registered
(warped) output image from the estimated transformation.

### Module Structure

```text
backend/registration/
├── __init__.py       — package docstring
├── models.py          — WarpConfig, RegistrationResult data structures
└── warping.py         — affine and perspective image warping
```

### Data Flow

```text
GeometricResult
   (transform_matrix, transform_model)
         │
         ▼
    warp_image()
   cv2.warpAffine / cv2.warpPerspective
         │
         ▼
  RegistrationResult
   (registered_image, metadata)
```

### Transform Direction

The matrix from `estimate_transform()` maps **reference → target**.
It is passed directly to `cv2.warpAffine` / `cv2.warpPerspective`
with **no inversion**.

- Source image: reference image
- Output: reference image warped into target coordinate frame

### Separation of Concerns

- Registration/warping does not re-estimate the transformation
- No dependency on SIFT, FLANN, or matching internals
- Input: `np.ndarray` (image) + `np.ndarray` (matrix) + `TransformModel`
- Output: `RegistrationResult` (self-contained)

---

## Evaluation Layer

The evaluation layer (`backend/evaluation/`) provides metrics
extraction, spatial distribution analysis, and visualization utilities.

### Module Structure

```text
backend/evaluation/
├── __init__.py         — package docstring
├── metrics.py           — MatchQualitySummary from GeometricResult
├── spatial.py           — 8×8 grid bucketing, normalized entropy
└── visualization.py     — match drawing, inlier/outlier, overlay
```

### Data Flow

```text
GeometricResult ──────┐
FilteredMatches ──────┤
RegistrationResult ───┤
                      ▼
               evaluation layer
              ┌────────────────┐
              │  metrics.py    │ → MatchQualitySummary
              │  spatial.py    │ → SpatialDistribution
              │  visualization │ → BGR uint8 images
              └────────────────┘
```

### Separation of Concerns

- Metrics **reuse** existing `ErrorMetrics` from geometry — no duplicate RMSE
- Spatial analysis consumes `FilteredMatch` + `inlier_mask` only
- Visualization returns NumPy arrays, never saves files
- No dependency on FastAPI, upload API, or frontend

---

## Registration API

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/register` | Run classical SIFT registration pipeline |
| GET | `/register/{id}/registered` | Retrieve registered image (PNG) |
| GET | `/register/{id}/matches` | Retrieve match visualization (PNG) |
| GET | `/register/{id}/inliers` | Retrieve inlier/outlier visualization (PNG) |
| GET | `/register/{id}/overlay` | Retrieve registration overlay (PNG) |

### Service Layer

```text
POST /register
    │
    ▼
backend/api/register.py        (thin route handler)
    │
    ▼
backend/core/registration_service.py   (orchestrator)
    │
    ├─→ routing.pair_classifier.classify_pair() [Pre-flight Guard]
    │      └─→ (if not implemented: early exit with success=False)
    │
    ├─→ preprocessing.io.load_image()
    ├─→ preprocessing.grayscale.to_feature_image()
    ├─→ features.sift.extract_sift()
    ├─→ matching.flann.flann_knn_match()
    ├─→ matching.ratio_test.apply_ratio_test()
    ├─→ geometry.estimation.estimate_transform()
    ├─→ registration.warping.warp_image()
    ├─→ evaluation.spatial.compute_spatial_distribution()
    ├─→ evaluation.metrics.build_quality_summary()
    └─→ evaluation.visualization.draw_*()
    │
    ▼
backend/core/result_store.py   (in-memory, FIFO eviction)
```

### Pre-Flight Pair Classification Guard and Routing

To prevent silent and misleading execution on scientifically unsupported cross-instrument pairs, `run_classical_registration()` executes an early pre-flight compatibility check via `classify_pair()` (`backend/routing/pair_classifier.py:L106-170`) immediately after parameter validation and before any image loading or feature extraction:

- **What it checks**:
  - Automatically identifies instrument metadata from PDS4 XML labels via `identify_instrument()`.
  - Classifies the image pair into defined categories (`SAME_INSTRUMENT_TMC2`, `SAME_INSTRUMENT_OHRC`, `SAME_INSTRUMENT_IIRS`, `CROSS_SCALE_OHRC_TMC2`, `CROSS_MODAL_TMC2_IIRS`, `CROSS_MODAL_EXTREME_SCALE_OHRC_IIRS`, `UNCLASSIFIED`).
  - Evaluates whether the pair category has a mature, validated pipeline implementation (`is_implemented`).

### Cross-Scale Registration (`CROSS_SCALE_OHRC_TMC2`)

- **Cross-Scale Routing**:
  - `CROSS_SCALE_OHRC_TMC2` evaluates to `is_implemented=True` with recommended pipeline stage `cross_scale_affine_v1`.
  - In `run_classical_registration()` (`backend/core/registration_service.py:L626-638`), it routes to a dedicated cross-scale pipeline path (`_run_cross_scale_registration()`):
    - Rejects non-affine transform models (`configuration` failure stage).
    - Determines fine (OHRC) and coarse (TMC-2) inputs dynamically from classification metadata regardless of upload order.
    - Extracts confirmed spatial resolutions from PDS4 metadata via `scale_handler.get_resolution()`.
    - Normalizes resolution by anti-aliased Gaussian downsampling of the fine image via `register_cross_scale()` (`backend/core/cross_scale_registration.py`).
    - Estimates coarse-space affine transform using classical SIFT/FLANN/MAGSAC++ and analytically composes it with the downsampling factor ($M_{\text{full}} = M_{\text{coarse}} \cdot S$).
    - Warps the full-resolution fine image into the coarse target frame using `warping.warp_image()`.
    - Evaluates registration quality in coarse pixel coordinates via `evaluation.metrics` and `evaluation.spatial`.

#### Known Limitations (Cross-Scale Registration)

KNOWN LIMITATION: The current OHRC↔TMC-2 real-data end-to-end test
(test_cli_cross_scale_ohrc_tmc2_e2e) uses small (500x500 downsampled to 25x25)
fixtures, which is too small for SIFT to produce a robust number of well-distributed
correspondences (currently 5 correspondences, spatial entropy 0.228). This is
confirmed to be a fixture-size limitation, not an algorithmic flaw — the same
cross-scale transform composition logic produces 120/149 inliers with 0.816 spatial
entropy on a large synthetic 8000x8000 scene. However, the large-scale test is
synthetic; it has NOT been confirmed that real OHRC/TMC-2 imagery at a comparable
scale will produce similarly favorable results, since real lunar texture density
may differ from a synthetic test scene. Do not present the current small real-data
E2E test's correspondence count or entropy as representative of real-world
performance. Obtaining larger real OHRC/TMC-2 fixtures for proper validation is
listed as follow-up work.

### Cross-Modal Registration (`CROSS_MODAL_TMC2_IIRS`)

- **Cross-Modal Routing**:
  - `CROSS_MODAL_TMC2_IIRS` evaluates to `is_implemented=True` with recommended pipeline stage `cross_modal_phase_congruency_v1` (`backend/routing/pair_classifier.py:L73-78`).
  - In `run_classical_registration()` (`backend/core/registration_service.py:L641-653`), it routes to a dedicated cross-modal pipeline path (`_run_cross_modal_registration()`, `backend/core/registration_service.py:L473-605`):
    - Rejects non-affine transform models (`configuration` failure stage; composition math is affine-only).
    - Determines fine (TMC-2, ~4.27 m/px) and coarse (IIRS, 82.70 m/px) inputs dynamically from classification metadata regardless of upload order.
    - Extracts confirmed spatial resolutions from PDS4 metadata via `scale_handler.get_resolution()`.
    - Normalizes resolution by anti-aliased Gaussian downsampling of the fine image (~19.4x ratio) via `register_cross_modal()` (`backend/core/cross_modal_registration.py:L175-422`).
    - Computes frequency-domain phase congruency maps (`backend/preprocessing/illumination.py`) on both images to extract structural edge/feature energy invariant to contrast and illumination differences.
    - Converts PC maps to uint8 representations for classical SIFT/FLANN/MAGSAC++ estimation and analytically composes the coarse-space affine transform with the downsampling factor ($M_{\text{full}} = M_{\text{coarse}} \cdot S$).
    - Warps the original full-resolution fine (TMC-2) image into the coarse (IIRS) target frame using `warping.warp_image()`.
    - Evaluates registration quality via `evaluation.metrics` and `evaluation.spatial`.

#### Known Limitations (Cross-Modal Registration)

KNOWN LIMITATION: The current TMC-2↔IIRS test's 'derived IIRS' fixture
(_create_derived_iirs_from_tmc2) simulates IIRS's 256-band hyperspectral structure by
tiling the SAME 2D downsampled TMC-2 slice across 16 bands (np.repeat), not genuine
per-band spectral diversity. This means the passing result (103/151 inliers, RMSE
0.569px, under a synthetic intensity inversion) demonstrates that Phase Congruency
provides structural robustness under a simulated modality/contrast difference — it
does NOT demonstrate registration performance against real IIRS spectral data, which
has genuinely different per-band content (256 distinct wavelengths, not a repeated
single band). Do not present this result as validating cross-modal registration
against real IIRS hyperspectral characteristics. A real IIRS PDS4 sample with genuine
multi-band content should be used for a stronger validation pass as follow-up work.

### Routing Edge Cases and Guard Behavior

- **Behavior on Rejection**:
  - Unsupported cross-instrument pairs (`CROSS_MODAL_EXTREME_SCALE_OHRC_IIRS` combining ~394x extreme scale and visible-to-infrared cross-modality) have `is_implemented=False` (`backend/routing/pair_classifier.py:L79-84`).
  - The pipeline immediately halts and returns a `RegistrationPipelineResult` with `success=False`, `failure_stage="pair_classification"`, and a descriptive `failure_reason` citing the pair category and required algorithmic capabilities.
  - Downstream stages (`load_image`, `to_feature_image`, `extract_sift`, `flann_knn_match`, `estimate_transform`, `warp_image`) are completely bypassed (`timings` remains empty).

- **Unaffected Pairs**:
  - **Same-instrument PDS4 pairs** (`TMC2-TMC2`, `OHRC-OHRC`, `IIRS-IIRS`) evaluate to `is_implemented=True` and proceed through the classical SIFT baseline without modification.
  - **Non-PDS4 / plain images** (e.g., PNG, JPEG, TIFF) where instrument metadata cannot be extracted fall back to `UNCLASSIFIED` with `is_implemented=True`, ensuring standard single-modality demo and testing workflows continue to run unaffected with identical numeric outcomes.

### Result Storage

- In-memory `OrderedDict` with FIFO eviction (max 50 results)
- No database, no persistence, no background workers
- Results keyed by UUID-based `result_id`
- Thread-safe access

### Pipeline Modes

Supported:
- `classical_sift` (same-modality baseline: TMC2-TMC2, OHRC-OHRC, IIRS-IIRS, UNCLASSIFIED)
- `cross_scale_affine_v1` (cross-scale OHRC ↔ TMC-2 via resolution normalization and composed affine transform)
- `cross_modal_phase_congruency_v1` (cross-modal TMC-2 ↔ IIRS visible-to-infrared via phase congruency, resolution normalization, and composed affine transform)

Not yet supported:
- Extreme-scale cross-modal (OHRC ↔ IIRS: ~394x scale + visible-to-infrared)
- Deep structural representations (MIND, RIFT)
- Multi-level pyramid feature tracking / ROI refinement on full-resolution OHRC
- Crater morphological anchoring
- Sub-pixel refinement

