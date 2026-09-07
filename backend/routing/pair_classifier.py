"""SIH26166 — Standalone pair-type classifier and routing scaffold.

Classifies pairs of input images by instrument and reports the appropriate
processing stage and implementation status.

This is a SCAFFOLD module only. It does not invoke or implement cross-modal
or multi-scale algorithms (Phase Congruency, MIND, RIFT, pyramid downsampling,
crater anchoring). It is wired into the live registration pipeline as a pre-flight guard in backend.core.registration_service.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import xml.etree.ElementTree as ET

from backend.preprocessing.pds4 import identify_instrument, is_pds4_label


class PairType(str, Enum):
    """Enumeration of lunar image registration pair categories."""

    SAME_INSTRUMENT_TMC2 = "SAME_INSTRUMENT_TMC2"
    SAME_INSTRUMENT_OHRC = "SAME_INSTRUMENT_OHRC"
    SAME_INSTRUMENT_IIRS = "SAME_INSTRUMENT_IIRS"
    CROSS_SCALE_OHRC_TMC2 = "CROSS_SCALE_OHRC_TMC2"
    CROSS_MODAL_TMC2_IIRS = "CROSS_MODAL_TMC2_IIRS"
    CROSS_MODAL_EXTREME_SCALE_OHRC_IIRS = "CROSS_MODAL_EXTREME_SCALE_OHRC_IIRS"
    UNCLASSIFIED = "UNCLASSIFIED"


@dataclass(frozen=True)
class PairClassification:
    """Result of classifying a pair of input images for registration."""

    instrument_a: str | None
    instrument_b: str | None
    pair_type: PairType
    is_same_instrument: bool
    is_implemented: bool
    recommended_pipeline_stage: str
    reason: str


# Static lookup table for classified instrument pairs: tuple(sorted([inst_a, inst_b])) -> config
# Mapping: key -> (PairType, is_implemented, recommended_pipeline_stage, reason)
_PAIR_CONFIGS: dict[tuple[str, str], tuple[PairType, bool, str, str]] = {
    ("TMC2", "TMC2"): (
        PairType.SAME_INSTRUMENT_TMC2,
        True,
        "classical_sift_baseline",
        "Same-instrument TMC-2 pair; validated with synthetic same-instrument transforms (rotation and scale) per the test suite.",
    ),
    ("OHRC", "OHRC"): (
        PairType.SAME_INSTRUMENT_OHRC,
        True,
        "classical_sift_baseline",
        "Same-instrument OHRC pair; validated with synthetic same-instrument transforms (rotation and scale) per the test suite.",
    ),
    ("IIRS", "IIRS"): (
        PairType.SAME_INSTRUMENT_IIRS,
        True,
        "classical_sift_baseline",
        "Same-instrument IIRS pair; validated with synthetic same-instrument transforms (rotation and scale) per the test suite.",
    ),
    ("OHRC", "TMC2"): (
        PairType.CROSS_SCALE_OHRC_TMC2,
        True,
        "cross_scale_affine_v1",
        "Cross-scale pair (OHRC ~0.21–0.25m/px vs TMC-2 ~4.27–5.0m/px, ~20x ratio); supported via resolution-normalization downsampling and analytical affine transform composition (no multi-level pyramid, ROI refinement, or crater anchoring).",
    ),
    ("IIRS", "TMC2"): (
        PairType.CROSS_MODAL_TMC2_IIRS,
        True,
        "cross_modal_phase_congruency_v1",
        "Cross-modal pair (TMC-2 ~4.27m/px visible vs IIRS 82.70m/px infrared, ~19.4x ratio); supported via resolution-normalization downsampling, phase congruency structural feature representation, classical SIFT/FLANN/MAGSAC++ matching, and analytical affine transform composition (affine-only; NOT MIND, NOT RIFT; not validated on real overlapping imagery since no overlapping pair currently exists).",
    ),
    ("IIRS", "OHRC"): (
        PairType.CROSS_MODAL_EXTREME_SCALE_OHRC_IIRS,
        False,
        "not_implemented",
        "Cross-modal extreme scale pair (OHRC ~0.25m/px visible vs IIRS 82.70m/px infrared [confirmed from fixture isda:pixel_resolution], >300x ratio); requires cross-modal matching (Phase Congruency/MIND/RIFT) and coarse-to-fine scale handling.",
    ),
}


def _extract_instrument(path: str | Path) -> str | None:
    """Extract canonical instrument identifier from path if it is a PDS4 label.

    Returns None for non-PDS4 images (PNG, JPEG, TIFF), nonexistent files,
    malformed XML, or unrecognized instruments. Reuses existing
    is_pds4_label() and identify_instrument() functions.
    """
    p = Path(path)
    if not is_pds4_label(p):
        return None
    try:
        tree = ET.parse(str(p))
        inst = identify_instrument(tree)
        return inst if inst != "UNKNOWN" else None
    except Exception:
        return None


def classify_pair(path_a: str | Path, path_b: str | Path) -> PairClassification:
    """Classify an image pair by instrument type and report routing information.

    Parameters
    ----------
    path_a : str | Path
        Path to the first image file (e.g. reference image).
    path_b : str | Path
        Path to the second image file (e.g. target image).

    Returns
    -------
    PairClassification
        Classification result containing identified instruments, pair type,
        implementation status, recommended pipeline stage, and reasoning.
        Classification is order-independent: classify_pair(a, b) and
        classify_pair(b, a) produce identical pair_type and routing metadata.
    """
    inst_a = _extract_instrument(path_a)
    inst_b = _extract_instrument(path_b)

    # If either instrument cannot be identified (e.g. plain PNG/TIFF inputs),
    # fall back to UNCLASSIFIED to preserve existing same-instrument-agnostic demo behavior.
    if inst_a is None or inst_b is None:
        return PairClassification(
            instrument_a=inst_a,
            instrument_b=inst_b,
            pair_type=PairType.UNCLASSIFIED,
            is_same_instrument=False,
            is_implemented=True,
            recommended_pipeline_stage="classical_sift_baseline",
            reason=(
                "Pair could not be instrument-classified from one or both inputs; "
                "falls back to default same-instrument-agnostic classical SIFT baseline."
            ),
        )

    # Order-independent lookup key
    key = tuple(sorted([inst_a, inst_b]))
    config = _PAIR_CONFIGS.get(key)

    if config is not None:
        pair_type, is_implemented, stage, reason = config
        is_same_inst = inst_a == inst_b
        return PairClassification(
            instrument_a=inst_a,
            instrument_b=inst_b,
            pair_type=pair_type,
            is_same_instrument=is_same_inst,
            is_implemented=is_implemented,
            recommended_pipeline_stage=stage,
            reason=reason,
        )

    # Fallback for unrecognized combination
    return PairClassification(
        instrument_a=inst_a,
        instrument_b=inst_b,
        pair_type=PairType.UNCLASSIFIED,
        is_same_instrument=(inst_a == inst_b),
        is_implemented=True,
        recommended_pipeline_stage="classical_sift_baseline",
        reason="Unrecognized instrument pair combination; falls back to classical SIFT baseline.",
    )
