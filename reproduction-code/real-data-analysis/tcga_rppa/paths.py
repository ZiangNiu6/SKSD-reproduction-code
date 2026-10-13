"""Repository paths shared by the TCGA-LUAD reproduction entry points."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
REPRODUCTION_ROOT = PACKAGE_ROOT.parents[1]


def _results_root() -> Path:
    configured = os.environ.get("TCGA_RPPA_RESULTS_DIR")
    if not configured:
        return REPRODUCTION_ROOT / "results" / "real-data-analysis-full" / "tcga-rppa"
    path = Path(configured).expanduser()
    return path if path.is_absolute() else REPRODUCTION_ROOT / path


RESULTS_ROOT = _results_root()
