"""Audited loading and independent-pilot preprocessing for TCGA LUAD RPPA."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROTEINS = (
    "EGFR",
    "HER2",
    "HER3",
    "Src",
    "Akt",
    "PTEN",
    "Tuberin",
    "mTOR",
    "S6",
    "4E-BP1",
)
GROUP_LABELS = ("low_T", "high_T")
RPPA_COLUMNS = tuple(f"M_{name}" for name in PROTEINS)

# STRING v12.0 functional network, human, combined score >= 0.9. The exact
# query and returned scores are frozen in graph_manifest.csv.
_PATHWAY_EDGE_NAMES = (
    ("Tuberin", "S6"),
    ("Tuberin", "mTOR"),
    ("Tuberin", "Akt"),
    ("HER3", "Src"),
    ("HER3", "EGFR"),
    ("HER3", "HER2"),
    ("HER2", "EGFR"),
    ("HER2", "Src"),
    ("EGFR", "PTEN"),
    ("EGFR", "Src"),
    ("4E-BP1", "Akt"),
    ("4E-BP1", "S6"),
    ("4E-BP1", "mTOR"),
    ("mTOR", "PTEN"),
    ("mTOR", "S6"),
    ("mTOR", "Akt"),
    ("PTEN", "Akt"),
    ("Src", "Akt"),
)
PATHWAY_EDGES = tuple(
    tuple(sorted((PROTEINS.index(left), PROTEINS.index(right))))
    for left, right in _PATHWAY_EDGE_NAMES
)

@dataclass(frozen=True)
class PreparedRPPA:
    source: pd.DataFrame
    proteins: tuple[str, ...]
    pilot_indices: np.ndarray
    analysis_indices: np.ndarray
    pilot_x: np.ndarray
    analysis_x: np.ndarray
    pilot_groups: np.ndarray
    analysis_groups: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    base_location: np.ndarray
    base_scale: np.ndarray
    log_t_low_cut: float
    log_t_high_cut: float
    seed: int

    @property
    def pilot_frame(self) -> pd.DataFrame:
        return self.source.iloc[self.pilot_indices].copy()

    @property
    def analysis_frame(self) -> pd.DataFrame:
        return self.source.iloc[self.analysis_indices].copy()


def _robust_scale(values: np.ndarray, *, floor: float = 1e-6) -> np.ndarray:
    center = np.median(values, axis=0)
    scale = 1.4826 * np.median(np.abs(values - center), axis=0)
    fallback = np.std(values, axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > floor), scale, fallback)
    if np.any(~np.isfinite(scale)) or np.any(scale <= floor):
        raise ValueError("At least one pilot coordinate has zero robust scale")
    return scale


def load_luad_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run `python prepare_data.py` in tcga_rppa first."
        )
    frame = pd.read_csv(path)
    required = {
        "participant_id",
        "rppa_barcode",
        "rna_barcode",
        "t_cell_score",
        *RPPA_COLUMNS,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"LUAD CSV is missing columns: {missing}")
    if frame.shape[0] != 360:
        raise ValueError(f"Expected 360 LUAD rows, found {frame.shape[0]}")
    if frame["participant_id"].nunique() != frame.shape[0]:
        raise ValueError("LUAD rows are not unique at the participant level")
    sample_types = frame["rppa_barcode"].str.split("-").str[3].str[:2]
    if not np.all(sample_types == "01"):
        raise ValueError("LUAD release contains a non-primary tumor sample")
    numeric = frame.loc[:, ["t_cell_score", *RPPA_COLUMNS]].to_numpy(float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("LUAD data contain missing or non-finite values")
    if np.any(frame["t_cell_score"].to_numpy(float) <= 0.0):
        raise ValueError("T-cell scores must be positive before log transformation")
    return frame


def prepare_luad(
    frame: pd.DataFrame,
    *,
    seed: int = 20260808,
) -> PreparedRPPA:
    """Use middle T-cell tertile for tuning and extreme tertiles for testing.

    Tertiles are determined from the outcome-free T-cell score, with the
    participant ID as a deterministic tie breaker. The 120 middle-tertile
    patients freeze protein coordinates, the reference density, and the
    kernel bandwidth. Fits and tests use the disjoint 120 low and 120 high
    patients. The seed is retained in the object for downstream bootstrap and
    fold assignment; it does not affect this deterministic split.
    """

    n = frame.shape[0]
    if n % 3:
        raise ValueError("The locked tertile design requires n divisible by three")

    log_t = np.log(frame["t_cell_score"].to_numpy(float))
    participant = frame["participant_id"].astype(str).to_numpy()
    order = np.lexsort((participant, log_t))
    third = n // 3
    low_indices = order[:third]
    pilot_indices = np.sort(order[third : 2 * third])
    high_indices = order[2 * third :]
    analysis_indices = np.concatenate((np.sort(low_indices), np.sort(high_indices)))
    analysis_groups = np.concatenate(
        (np.zeros(third, dtype=int), np.ones(third, dtype=int))
    )
    # Re-sort the coordinates and labels together by source-row index. This
    # makes output lineage easy to audit without changing group membership.
    source_order = np.argsort(analysis_indices)
    analysis_indices = analysis_indices[source_order]
    analysis_groups = analysis_groups[source_order]
    pilot_groups = np.full(third, -1, dtype=int)
    low_cut = float(0.5 * (log_t[order[third - 1]] + log_t[order[third]]))
    high_cut = float(
        0.5 * (log_t[order[2 * third - 1]] + log_t[order[2 * third]])
    )

    y = frame.loc[:, RPPA_COLUMNS].to_numpy(float)
    pilot_y = y[pilot_indices]
    center = np.median(pilot_y, axis=0)
    scale = _robust_scale(pilot_y)
    x = (y - center) / scale
    pilot_x = x[pilot_indices]
    analysis_x = x[analysis_indices]

    # The common t_5 reference is completely frozen on the middle tertile.
    # Group-specific bounded marginal terms are estimated on the analysis
    # observations and included in the influence correction.
    reference_location = np.median(pilot_x, axis=0)
    reference_scale = _robust_scale(pilot_x)
    base_location = np.repeat(reference_location[None, :], 2, axis=0)
    base_scale = np.repeat(reference_scale[None, :], 2, axis=0)

    return PreparedRPPA(
        source=frame,
        proteins=PROTEINS,
        pilot_indices=pilot_indices,
        analysis_indices=analysis_indices,
        pilot_x=pilot_x,
        analysis_x=analysis_x,
        pilot_groups=pilot_groups,
        analysis_groups=analysis_groups,
        center=center,
        scale=scale,
        base_location=base_location,
        base_scale=base_scale,
        log_t_low_cut=low_cut,
        log_t_high_cut=high_cut,
        seed=seed,
    )


def complete_edges(dimension: int | None = None) -> tuple[tuple[int, int], ...]:
    d = len(PROTEINS) if dimension is None else int(dimension)
    return tuple((i, j) for i in range(d - 1) for j in range(i + 1, d))
