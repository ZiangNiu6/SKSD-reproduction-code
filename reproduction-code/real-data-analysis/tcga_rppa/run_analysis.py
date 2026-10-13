#!/usr/bin/env python3
"""Evaluate shared versus T-cell-varying continuous RPPA interaction models."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy


ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
if str(PARENT) not in sys.path:
    sys.path.insert(0, str(PARENT))

from tcga_rppa.data import (  # noqa: E402
    PATHWAY_EDGES,
    PROTEINS,
    complete_edges,
    load_luad_csv,
    prepare_luad,
)
from tcga_rppa.ksd import (  # noqa: E402
    evaluate_ksd,
    median_within_group_bandwidth,
    stratified_efron_counts,
    stratified_gaussian_multipliers,
)
from tcga_rppa.model import (  # noqa: E402
    DEFAULT_MARGINAL_KNOTS,
    DEFAULT_MARGINAL_WIDTH,
    MODEL_LABELS,
    StudentTBase,
    fit_model,
)
from tcga_rppa.paths import RESULTS_ROOT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test whether low/high T-cell LUAD tumors share one continuous "
            "RTK-PI3K/AKT-mTOR interaction matrix."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "derived" / "luad_rppa.csv",
    )
    parser.add_argument(
        "--graph",
        choices=("pathway", "complete"),
        default="pathway",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_LABELS),
        default=list(MODEL_LABELS),
    )
    parser.add_argument("--bootstrap", type=int, default=999)
    parser.add_argument("--student-df", type=float, default=5.0)
    parser.add_argument("--phi-scale", type=float, default=1.0)
    parser.add_argument(
        "--marginal-knots",
        nargs="+",
        type=float,
        default=list(DEFAULT_MARGINAL_KNOTS),
    )
    parser.add_argument(
        "--marginal-width", type=float, default=DEFAULT_MARGINAL_WIDTH
    )
    parser.add_argument("--bandwidth-multiplier", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "default: results/real-data-analysis-full/tcga-rppa/outputs/<graph>"
        ),
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="write numerical outputs without invoking the R/ggplot helper",
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def holm_adjust(p_values: list[float | None]) -> list[float | None]:
    available = [
        (index, float(value))
        for index, value in enumerate(p_values)
        if value is not None and np.isfinite(float(value))
    ]
    adjusted: list[float | None] = [None] * len(p_values)
    ordered = sorted(available, key=lambda item: item[1])
    running = 0.0
    # Preserve the full predeclared family even when a nonregular candidate
    # has no formal p-value; this is equivalent to assigning it p=1 for the
    # multiplicity calculation while retaining NA in the output cell.
    total = len(p_values)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * value))
        adjusted[index] = running
    return adjusted


def make_plots(output: Path) -> None:
    """Render the manuscript model-evaluation figure."""

    helper = ROOT.parent / "plotting-code" / "plot_tcga_rppa.R"
    if not helper.is_file():
        raise FileNotFoundError(f"Missing TCGA plotting helper: {helper}")
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "Rscript is required for figures; rerun with --skip-plots to "
            "write numerical outputs only"
        )
    completed = subprocess.run(
        [rscript, str(helper), str(output.resolve())],
        check=False,
        text=True,
        capture_output=True,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"TCGA R plotting helper failed with exit code {completed.returncode}"
        )
    for name in ("model_evaluation.pdf", "model_evaluation.png"):
        figure = output / name
        if not figure.is_file() or figure.stat().st_size == 0:
            raise RuntimeError(f"R plotting helper did not create {figure}")


def main() -> None:
    args = parse_args()
    if args.bootstrap < 19:
        raise ValueError("Use at least 19 bootstrap replicates")
    if args.bandwidth_multiplier <= 0.0:
        raise ValueError("bandwidth_multiplier must be positive")
    output = args.output_dir or (RESULTS_ROOT / "outputs" / args.graph)
    output.mkdir(parents=True, exist_ok=True)

    frame = load_luad_csv(args.data)
    prepared = prepare_luad(frame, seed=args.seed)
    base = StudentTBase(
        prepared.base_location,
        prepared.base_scale,
        degrees_of_freedom=args.student_df,
    )
    edges = {
        "pathway": PATHWAY_EDGES,
        "complete": complete_edges(),
    }[args.graph]
    bandwidth = (
        median_within_group_bandwidth(prepared.pilot_x, prepared.pilot_groups)
        * args.bandwidth_multiplier
    )
    rng = np.random.default_rng(args.seed + 1009)
    counts = stratified_efron_counts(
        prepared.analysis_groups, args.bootstrap, rng
    )
    gaussian_weights = stratified_gaussian_multipliers(
        prepared.analysis_groups, args.bootstrap, rng
    )
    marginal_knots = tuple(args.marginal_knots)

    result_rows = []
    for model in args.models:
        print(f"Fitting {model} ({args.graph} graph)...", flush=True)
        fitted = fit_model(
            model,
            prepared.analysis_x,
            prepared.analysis_groups,
            base,
            edges,
            marginal_knots=marginal_knots,
            marginal_width=args.marginal_width,
            phi_scale=args.phi_scale,
        )
        ksd, _ = evaluate_ksd(
            fitted,
            prepared.analysis_x,
            prepared.analysis_groups,
            bandwidth,
            counts,
            gaussian_weights,
            brueck_correction="midpoint",
        )
        result_rows.append(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "graph": args.graph,
                "edge_count": len(edges),
                "parameter_count": fitted.beta.size,
                "n_low": int(np.sum(prepared.analysis_groups == 0)),
                "n_high": int(np.sum(prepared.analysis_groups == 1)),
                "ksd_v": ksd.v_statistic,
                "ksd_u": ksd.u_statistic,
                "ksd": ksd.ksd,
                "p_value_influence": ksd.influence_p_value,
                "p_value_brueck": ksd.brueck_p_value,
                "influence_calibration": ksd.influence_calibration,
                "brueck_calibration": ksd.brueck_calibration,
                "bootstrap_replicates": ksd.bootstrap_replicates,
                "influence_seconds": ksd.influence_seconds,
                "brueck_seconds": ksd.brueck_seconds,
                "brueck_refits": ksd.brueck_refits,
                "brueck_max_hessian_condition": (
                    ksd.brueck_max_hessian_condition
                ),
                "brueck_max_estimating_residual": (
                    ksd.brueck_max_estimating_residual
                ),
                "brueck_bootstrap_min": ksd.brueck_bootstrap_min,
                "brueck_bootstrap_q05": ksd.brueck_bootstrap_q05,
                "brueck_bootstrap_median": ksd.brueck_bootstrap_median,
                "brueck_bootstrap_q95": ksd.brueck_bootstrap_q95,
                "brueck_bootstrap_max": ksd.brueck_bootstrap_max,
                "converged": fitted.converged,
                "regular": fitted.regular,
                "normalizable": fitted.normalizable,
                "hessian_condition": fitted.hessian_condition,
                "fit_message": fitted.message,
            }
        )

    results = pd.DataFrame(result_rows)
    results["p_value_influence_holm"] = holm_adjust(
        results["p_value_influence"].tolist()
    )
    results["p_value_brueck_holm"] = holm_adjust(
        results["p_value_brueck"].tolist()
    )
    results.to_csv(output / "model_evaluation.csv", index=False)
    if not args.skip_plots:
        make_plots(output)

    metadata = {
        "analysis": "TCGA LUAD RPPA low/high T-cell continuous EBM",
        "source_commit": "95d8a6710950920ca0e2eb5001896b8d0c47d4df",
        "data_path": str(args.data.resolve()),
        "data_sha256": file_sha256(args.data),
        "graph": args.graph,
        "graph_manifest_sha256": file_sha256(ROOT / "graph_manifest.csv"),
        "proteins": list(PROTEINS),
        "edges": [[PROTEINS[i], PROTEINS[j]] for i, j in edges],
        "tuning_n": int(prepared.pilot_x.shape[0]),
        "analysis_n": int(prepared.analysis_x.shape[0]),
        "log_t_low_cut": prepared.log_t_low_cut,
        "log_t_high_cut": prepared.log_t_high_cut,
        "bandwidth": bandwidth,
        "brueck_correction": "midpoint",
        "composite_calibrations": {
            "proposed": (
                "influence-adjusted stratified Gaussian multiplier V-bootstrap"
            ),
            "benchmark": (
                "project factor-corrected midpoint diagnostic built on the "
                "Brueck refitted empirical U-bootstrap, with fixed-stratum "
                "resampling and score-matching refits; not an author-confirmed erratum"
            ),
            "brueck_reference": "arXiv:2510.22792v2",
            "brueck_scope_note": (
                "fixed-stratum extension; the paper states an iid empirical "
                "bootstrap"
            ),
            "brueck_condition_diagnostic": (
                "maximum LAPACK 1-norm condition estimate for each diagonally "
                "normalized bootstrap score-matching Hessian; no ridge or "
                "failed-replicate redraw"
            ),
        },
        "arguments": vars(args) | {"data": str(args.data), "output_dir": str(output)},
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
        },
    }
    (output / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        results[
            [
                "model",
                "ksd_v",
                "ksd_u",
                "p_value_influence",
                "p_value_influence_holm",
                "p_value_brueck",
                "p_value_brueck_holm",
            ]
        ].to_string(index=False)
    )
    print(f"Wrote {output / 'model_evaluation.csv'}")


if __name__ == "__main__":
    main()
