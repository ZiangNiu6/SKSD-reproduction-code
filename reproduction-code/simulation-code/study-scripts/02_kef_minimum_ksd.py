#!/usr/bin/env python3
"""Run the paired KEF simulation or plot its existing standardized results.

The default runs all six methods using numerical-helpers/, with
checkpointed joint statistical/timing records. ``--check`` selects a small
fresh simulation. ``--plot-only`` validates the saved CSVs and regenerates
figures without running any simulation or reading source archives.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence

import numpy as np


SIMULATION_ID = "02_kef_minimum_ksd"
SCHEMA_VERSION = "semiparametric_ksd_kef_minimum_ksd_paired_aggregate_v2"
LEGACY_SCHEMA_VERSION = "semiparametric_ksd_kef_minimum_ksd_v1"
FRESH_SCHEMA_VERSION = "semiparametric_ksd_kef_minimum_ksd_fresh_v1"
PUBLICATION_DIMENSIONS = (1, 2, 3)
PUBLICATION_N = 100
PUBLICATION_B = 500
PUBLICATION_REPLICATIONS = 500
PUBLICATION_THETA1 = 10.0
PUBLICATION_SIGMA = 3.0
PUBLICATION_ALPHA = 0.05
PUBLICATION_FRESH_SEED = 20260731
PUBLICATION_BASE_GRID = np.array(
    [-4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4],
    dtype=float,
)
PAIRED_ARM = "fresh_paired_observed_datasets_aggregate_only"
SAME_RUN_TIMING_BASIS = (
    "same statistical Monte Carlo replications as rejection-rate aggregates"
)
HISTORICAL_TIMING_BASIS = (
    "separate explicitly selected historical timing aggregate; rejection "
    "fields ignored"
)


class ProvenanceError(ValueError):
    """Raised when saved results violate the recorded simulation contract."""


@dataclass(frozen=True)
class RunConfig:
    output_dir: Path
    dimensions: tuple[int, ...]
    n: int
    bootstrap_draws: int
    replications: int
    theta1: float
    sigma: float
    alpha: float
    fresh_seed: int
    quick: bool
    plot_only: bool
    resume: bool
    force: bool
    plot: bool
    require_plot: bool
    workflow: str = "simulate"
    base_grid: tuple[float, ...] = tuple(float(value) for value in PUBLICATION_BASE_GRID)

    @property
    def effective_dimensions(self) -> tuple[int, ...]:
        return self.dimensions

    @property
    def publication_design(self) -> bool:
        return (
            self.dimensions == PUBLICATION_DIMENSIONS
            and self.n == PUBLICATION_N
            and self.bootstrap_draws == PUBLICATION_B
            and self.replications == PUBLICATION_REPLICATIONS
            and math.isclose(self.theta1, PUBLICATION_THETA1)
            and math.isclose(self.sigma, PUBLICATION_SIGMA)
            and math.isclose(self.alpha, PUBLICATION_ALPHA)
            and self.fresh_seed == PUBLICATION_FRESH_SEED
            and self.base_grid == tuple(PUBLICATION_BASE_GRID)
            and not self.quick
        )


METHODS = (
    {
        "method": "SKSD-parametric",
        "display_method": "SKSD-parametric",
        "test": "KSD",
        "calibration": "parametric bootstrap",
    },
    {
        "method": "SKSD-influence",
        "display_method": "SKSD-influence",
        "test": "KSD",
        "calibration": "influence-adjusted Gaussian multiplier bootstrap",
    },
    {
        "method": "Brueck-corrected",
        "display_method": "Brück et al.",
        "test": "KSD",
        "calibration": "factor-corrected midpoint Efron diagnostic",
    },
    {
        "method": "SMMD",
        "display_method": "SMMD (Key et al.)",
        "test": "MMD",
        "calibration": "parametric bootstrap",
    },
    {
        "method": "SKS",
        "display_method": "SKS",
        "test": "KS",
        "calibration": "parametric bootstrap",
    },
    {
        "method": "SKSD-naive-wild",
        "display_method": "SKSD-naive wild",
        "test": "KSD",
        "calibration": "naive wild Gaussian multiplier bootstrap (no adjustment)",
    },
)


PAIRED_METHODS = METHODS
SAVED_PAIRED_METHODS = tuple(
    {**spec, "calibration": "influence-adjusted Efron bootstrap"}
    if spec["method"] == "SKSD-influence" else spec
    for spec in METHODS if spec["method"] != "SKSD-naive-wild"
)


PAIRED_TIME_KEYS = {
    "t_KSD_K",
    "t_KSD_IF_K",
    "t_BRUECK_K",
    "t_MMD_K",
    "t_KS_K",
}
REPLICATION_COLUMNS = (
    "simulation",
    "dimension",
    "theta_index",
    "theta2",
    "theta2_base",
    "replication",
    "method",
    "display_method",
    "test",
    "calibration",
    "nuisance_estimator",
    "p_value",
    "reject",
    "alpha",
    "data_seed",
    "bootstrap_seed",
    "paired_replication_key",
    "monte_carlo_arm",
    "paired_with_legacy",
    "source_archive",
)

SUMMARY_COLUMNS = (
    "simulation",
    "dimension",
    "theta_index",
    "theta2",
    "theta2_base",
    "method",
    "display_method",
    "test",
    "calibration",
    "nuisance_estimator",
    "rejection_rate",
    "rejected",
    "monte_carlo_repetitions",
    "mcse",
    "ci_lower",
    "ci_upper",
    "mean_seconds_per_replication",
    "monte_carlo_arm",
    "paired_within_arm",
    "paired_with_legacy",
    "replications_available",
    "source_archive",
)

PAIRED_REPLICATION_COLUMNS = (
    "simulation",
    "dimension",
    "theta_index",
    "theta2",
    "theta2_base",
    "replication",
    "method",
    "display_method",
    "test",
    "calibration",
    "nuisance_estimator",
    "p_value",
    "reject",
    "alpha",
    "source_archive",
)

PAIRED_SUMMARY_COLUMNS = (
    "simulation",
    "dimension",
    "theta_index",
    "theta2",
    "theta2_base",
    "method",
    "display_method",
    "test",
    "calibration",
    "nuisance_estimator",
    "rejection_rate",
    "rejected",
    "monte_carlo_repetitions",
    "mcse",
    "ci_lower",
    "ci_upper",
    "mean_seconds_per_replication",
    "monte_carlo_arm",
    "paired_observed_datasets",
    "shared_bootstrap_randomness_across_all_methods",
    "paired_contrasts_reconstructible",
    "replications_available",
    "source_archive",
)


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_output_dir(quick: bool = False) -> Path:
    return (
        repository_root()
        / "results"
        / ("simulation-analysis-check" if quick else "simulation-analysis")
        / "02-kef-minimum-ksd"
    )


def parse_dimensions(text: str) -> tuple[int, ...]:
    try:
        values = tuple(int(value.strip()) for value in text.split(",") if value.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--dimensions must be a comma-separated list of positive integers"
        ) from error
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError(
            "--dimensions must contain at least one positive integer"
        )
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("--dimensions contains duplicates")
    return values


def parse_base_grid(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(value.strip()) for value in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("--base-grid must be comma-separated numbers") from error
    if not values or any(not math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("--base-grid must contain finite numbers")
    if any(left >= right for left, right in zip(values, values[1:])):
        raise argparse.ArgumentTypeError("--base-grid must be strictly increasing")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the paired KEF simulation or plot existing standardized results."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--simulate", action="store_true", help="run the local six-method simulation (default)")
    mode.add_argument(
        "--plot-only", action="store_true",
        help="validate saved CSVs in --output-dir and regenerate figures without simulation",
    )
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=f"result directory (default: {default_output_dir()}; --check uses simulation-analysis-check/)")
    parser.add_argument("--dimensions", type=parse_dimensions,
                        help="comma-separated dimensions (publication: 1,2,3)")
    parser.add_argument("--sample-size", type=int, help="observed sample size (full: 100; check: 12)")
    parser.add_argument("--bootstrap-draws", "--B", dest="bootstrap_draws", type=int,
                        help="bootstrap draws per method/dataset (full: 500; check: 20)")
    parser.add_argument("--replications", type=int,
                        help="datasets per cell for statistics and timing (full: 500; check: 2)")
    parser.add_argument("--base-grid", type=parse_base_grid,
                        help="theta2*sqrt(d) grid; use --base-grid=-0.5,0,0.5 for negatives")
    parser.add_argument("--theta1", type=float, default=PUBLICATION_THETA1)
    parser.add_argument("--sigma", type=float, default=PUBLICATION_SIGMA)
    parser.add_argument("--alpha", type=float, default=PUBLICATION_ALPHA)
    parser.add_argument("--fresh-seed", type=int, default=PUBLICATION_FRESH_SEED)
    parser.add_argument("--check", dest="quick", action="store_true",
                        help="small simulation (d=1,n=12,B=20,R=2,three signals)")
    parser.add_argument("--quick", dest="quick", action="store_true", help=argparse.SUPPRESS)
    reuse = parser.add_mutually_exclusive_group()
    reuse.add_argument("--resume", action="store_true",
                       help="resume a compatible simulation checkpoint")
    reuse.add_argument("--force", action="store_true",
                       help="replace this entry point's standard simulation output files")
    parser.add_argument("--no-plot", action="store_true",
                        help="write and validate tabular outputs without R plotting")
    parser.add_argument("--require-plot", action="store_true",
                        help="fail if Rscript or the plotting packages are unavailable")
    return parser


def config_from_args(args: argparse.Namespace) -> RunConfig:
    n = args.sample_size if args.sample_size is not None else (12 if args.quick else PUBLICATION_N)
    bootstrap_draws = (
        args.bootstrap_draws if args.bootstrap_draws is not None
        else (20 if args.quick else PUBLICATION_B)
    )
    replications = (
        args.replications if args.replications is not None
        else (2 if args.quick else PUBLICATION_REPLICATIONS)
    )
    dimensions = args.dimensions or ((1,) if args.quick else PUBLICATION_DIMENSIONS)
    base_grid = (
        args.base_grid if args.base_grid is not None
        else ((-0.5, 0.0, 0.5) if args.quick else tuple(PUBLICATION_BASE_GRID))
    )
    if n < 2:
        raise ValueError("--sample-size must be at least 2")
    if bootstrap_draws < 20:
        raise ValueError(
            "--bootstrap-draws must be at least 20 at alpha=0.05; smaller "
            "values cannot attain the manuscript rejection threshold"
        )
    if replications < 1:
        raise ValueError("--replications must be positive")
    if not math.isfinite(args.sigma) or args.sigma <= 0:
        raise ValueError("--sigma must be finite and positive")
    if not math.isfinite(args.theta1):
        raise ValueError("--theta1 must be finite")
    if not 0 < args.alpha < 1:
        raise ValueError("--alpha must lie strictly between 0 and 1")
    if not math.isclose(args.alpha, PUBLICATION_ALPHA, abs_tol=1e-15):
        raise ValueError(
            "the manuscript plotting/calibration workflow uses alpha=0.05; "
            "fresh runs retain p-values for separate threshold analyses"
        )
    if args.fresh_seed < 0:
        raise ValueError("--fresh-seed must be nonnegative")
    if args.require_plot and args.no_plot:
        raise ValueError("--require-plot and --no-plot cannot be combined")
    if args.plot_only and args.no_plot:
        raise ValueError("--plot-only and --no-plot cannot be combined")
    if args.plot_only and (args.force or args.resume):
        raise ValueError("--plot-only cannot be combined with --force or --resume")
    return RunConfig(
        output_dir=(args.output_dir or default_output_dir(args.quick)).expanduser().resolve(),
        dimensions=tuple(dimensions),
        n=n,
        bootstrap_draws=bootstrap_draws,
        replications=replications,
        theta1=args.theta1,
        sigma=args.sigma,
        alpha=args.alpha,
        fresh_seed=args.fresh_seed,
        quick=args.quick,
        plot_only=args.plot_only,
        resume=args.resume,
        force=args.force,
        plot=not args.no_plot,
        require_plot=args.require_plot or args.plot_only,
        base_grid=tuple(float(value) for value in base_grid),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_rows(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]
) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    os.replace(temporary, path)


def _atomic_write_metadata(path: Path, metadata: Mapping[str, object]) -> None:
    rows = ({"key": key, "value": value} for key, value in metadata.items())
    _atomic_write_rows(path, ("key", "value"), list(rows))


def _read_metadata(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"key", "value"}:
        raise ProvenanceError(f"{path} is not a key/value metadata CSV")
    metadata: dict[str, str] = {}
    for row in rows:
        if row["key"] in metadata:
            raise ProvenanceError(f"{path} has duplicate metadata key {row['key']!r}")
        metadata[row["key"]] = row["value"]
    return metadata


def _read_rows(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise ProvenanceError(f"{path} has an invalid CSV header")
        return [dict(row) for row in reader]


def _require_output_hash(metadata: Mapping[str, str], key: str, path: Path) -> None:
    recorded = metadata.get(key)
    if not recorded:
        raise ProvenanceError(f"run_metadata.csv is missing {key}")
    observed = _sha256(path)
    if recorded != observed:
        raise ProvenanceError(f"{path.name} does not match its recorded SHA-256")


def _load_fresh_simulation_module():
    name = "_kef_reproduction_simulation"
    path = Path(__file__).resolve().parent.parent / "numerical-helpers" / "_kef_simulation.py"
    existing = sys.modules.get(name)
    if existing is not None and Path(existing.__file__).resolve() == path.resolve():
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load KEF simulation helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def validate_fresh_standardized_tables(
    metadata: Mapping[str, str], replications_path: Path, summary_path: Path,
) -> None:
    _require_output_hash(metadata, "replications_sha256", replications_path)
    _require_output_hash(metadata, "summary_sha256", summary_path)
    rows = _read_rows(replications_path, PAIRED_REPLICATION_COLUMNS + ("seconds",))
    summary = _read_rows(summary_path, PAIRED_SUMMARY_COLUMNS)
    _load_fresh_simulation_module().validate_tables(
        metadata, rows, summary, sys.modules[__name__]
    )


def _validate_standardized_tables(
    metadata: Mapping[str, str],
    replications_path: Path,
    summary_path: Path,
) -> None:
    _require_output_hash(metadata, "replications_sha256", replications_path)
    _require_output_hash(metadata, "summary_sha256", summary_path)
    replications = _read_rows(replications_path, REPLICATION_COLUMNS)
    summaries = _read_rows(summary_path, SUMMARY_COLUMNS)
    if len(replications) != int(metadata.get("replications_csv_rows", "-1")):
        raise ProvenanceError("replications.csv row count disagrees with metadata")
    if len(summaries) != int(metadata.get("summary_csv_rows", "-1")):
        raise ProvenanceError("summary.csv row count disagrees with metadata")

    method_specs = {str(spec["method"]): spec for spec in METHODS}
    cells: dict[tuple[int, int], set[str]] = {}
    for row in summaries:
        method = row["method"]
        spec = method_specs.get(method)
        if spec is None:
            raise ProvenanceError(f"summary.csv contains unknown method {method!r}")
        if (
            row["display_method"] != spec["display_method"]
            or row["test"] != spec["test"]
            or row["calibration"] != spec["calibration"]
            or row["nuisance_estimator"] != "minimum KSD"
        ):
            raise ProvenanceError(f"summary.csv method metadata is wrong for {method}")
        try:
            dimension = int(row["dimension"])
            theta_index = int(row["theta_index"])
            rate = float(row["rejection_rate"])
            repetitions = int(row["monte_carlo_repetitions"])
            rejected = int(row["rejected"])
            seconds = float(row["mean_seconds_per_replication"])
            lower = float(row["ci_lower"])
            upper = float(row["ci_upper"])
        except ValueError as error:
            raise ProvenanceError("summary.csv contains malformed numerical values") from error
        if not (
            0.0 <= rate <= 1.0
            and 0 <= rejected <= repetitions
            and repetitions > 0
            and math.isclose(rate, rejected / repetitions, abs_tol=1e-12)
            and math.isfinite(seconds)
            and seconds >= 0.0
            and 0.0 <= lower <= upper <= 1.0
        ):
            raise ProvenanceError("summary.csv contains invalid numerical results")
        cell_methods = cells.setdefault((dimension, theta_index), set())
        if method in cell_methods:
            raise ProvenanceError("summary.csv contains a duplicate method/cell row")
        cell_methods.add(method)
    expected_methods = set(method_specs)
    if not cells or any(methods != expected_methods for methods in cells.values()):
        raise ProvenanceError("summary.csv does not contain five methods per cell")
    expected_cells: set[tuple[int, int]] = set()
    try:
        for dimension_block in metadata["selected_theta_indices"].split(";"):
            dimension_text, indices_text = dimension_block.split(":", 1)
            dimension = int(dimension_text.removeprefix("d"))
            expected_cells.update(
                (dimension, int(index))
                for index in indices_text.split(",")
                if index
            )
    except (KeyError, ValueError) as error:
        raise ProvenanceError("metadata has invalid selected_theta_indices") from error
    if set(cells) != expected_cells:
        raise ProvenanceError("summary.csv cells disagree with selected_theta_indices")

    fresh_methods = {"SKSD-influence", "Brueck-corrected"}
    paired: dict[tuple[int, int, int], set[str]] = {}
    for row in replications:
        method = row["method"]
        spec = method_specs.get(method)
        if method not in fresh_methods or spec is None:
            raise ProvenanceError("replications.csv contains a non-fresh method")
        if (
            row["display_method"] != spec["display_method"]
            or row["test"] != spec["test"]
            or row["calibration"] != spec["calibration"]
            or row["nuisance_estimator"] != "minimum KSD"
        ):
            raise ProvenanceError("replications.csv contains inconsistent method metadata")
        try:
            key = (
                int(row["dimension"]),
                int(row["theta_index"]),
                int(row["replication"]),
            )
            p_value = float(row["p_value"])
            alpha = float(row["alpha"])
        except ValueError as error:
            raise ProvenanceError("replications.csv contains malformed numbers") from error
        if not (0.0 <= p_value <= 1.0 and 0.0 < alpha < 1.0):
            raise ProvenanceError("replications.csv contains an invalid p-value")
        expected_reject = str(p_value < alpha)
        if row["reject"] != expected_reject or row["paired_with_legacy"] != "False":
            raise ProvenanceError("replications.csv has inconsistent provenance flags")
        methods = paired.setdefault(key, set())
        if method in methods:
            raise ProvenanceError("replications.csv contains a duplicate method/cell row")
        methods.add(method)
    if not paired or any(methods != fresh_methods for methods in paired.values()):
        raise ProvenanceError("replications.csv does not contain paired fresh methods")
    if {(dimension, theta_index) for dimension, theta_index, _ in paired} != expected_cells:
        raise ProvenanceError("replications.csv cells disagree with summary.csv")


def _selected_cells_from_metadata(metadata: Mapping[str, str]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    try:
        for dimension_block in metadata["selected_theta_indices"].split(";"):
            dimension_text, indices_text = dimension_block.split(":", 1)
            dimension = int(dimension_text.removeprefix("d"))
            cells.update(
                (dimension, int(index))
                for index in indices_text.split(",")
                if index
            )
    except (KeyError, ValueError) as error:
        raise ProvenanceError("metadata has invalid selected_theta_indices") from error
    return cells


def _paired_timing_protocol_is_unrecorded(metadata: Mapping[str, str]) -> bool:
    """Identify the older CSV contract, not partially populated modern metadata."""
    return not any(
        key in {"timing_replication_basis", "timing_protocol", "timing_clock"}
        or key.endswith(("_timing_protocol", "_timing_clock"))
        for key in metadata
    )


def _validate_paired_timing_provenance(metadata: Mapping[str, str]) -> None:
    """Validate recorded CSV provenance without opening the original aggregates.

    Earlier paired-v2 CSVs identify a separate timing run and its replication
    count, but predate timing protocol/clock fields. Keep that limitation
    explicit: accepting these saved plots neither invents the missing fields
    nor certifies their timings as the current standalone benchmarking protocol.
    Once any newer protocol field is present, require the full modern contract.
    """
    if metadata.get("timing_rejection_fields_used") != "false":
        raise ProvenanceError("timing results must not supply rejection results")
    timing_mode = metadata.get("timing_source_mode")
    if timing_mode not in {"result-aggregate", "dedicated-short-run"}:
        raise ProvenanceError(
            "run_metadata.csv has an invalid timing_source_mode"
        )
    if metadata.get("timing_fields_imported") != ",".join(
        sorted(PAIRED_TIME_KEYS)
    ):
        raise ProvenanceError(
            "run_metadata.csv does not identify the five timing fields"
        )
    try:
        timing_replications = int(metadata["timing_monte_carlo_replications"])
        result_replications = int(metadata["monte_carlo_replications"])
        dimensions = tuple(
            int(value) for value in metadata["dimensions"].split(",")
        )
    except (KeyError, ValueError) as error:
        raise ProvenanceError(
            "run_metadata.csv has malformed timing provenance"
        ) from error
    if (
        timing_replications <= 0
        or result_replications <= 0
        or not dimensions
        or len(set(dimensions)) != len(dimensions)
        or any(dimension <= 0 for dimension in dimensions)
        or not metadata.get("timing_source_dir")
    ):
        raise ProvenanceError(
            "run_metadata.csv has invalid timing replication/source metadata"
        )
    if (
        timing_mode == "result-aggregate"
        and timing_replications != result_replications
    ):
        raise ProvenanceError(
            "result-aggregate timings must use the result replication count"
        )
    expected_timing_basis = (
        SAME_RUN_TIMING_BASIS
        if timing_mode == "result-aggregate"
        else HISTORICAL_TIMING_BASIS
    )
    historical_unrecorded_protocol = _paired_timing_protocol_is_unrecorded(metadata)
    if historical_unrecorded_protocol and timing_mode != "dedicated-short-run":
        raise ProvenanceError(
            "unrecorded timing protocol/clock is supported only for historical "
            "dedicated-short-run saved CSVs"
        )
    if not historical_unrecorded_protocol and (
        metadata.get("timing_replication_basis") != expected_timing_basis
    ):
        raise ProvenanceError(
            "run_metadata.csv has inconsistent timing_replication_basis"
        )
    timing_protocol = metadata.get("timing_protocol", "")
    timing_clock = metadata.get("timing_clock", "")
    if not historical_unrecorded_protocol and (not timing_protocol or not timing_clock):
        raise ProvenanceError(
            "run_metadata.csv has incomplete timing protocol/clock metadata"
        )
    if timing_mode == "result-aggregate" and (
        metadata.get("timing_source_dir") != metadata.get("source_dir")
    ):
        raise ProvenanceError(
            "result-aggregate timings must use the result source directory"
        )
    if historical_unrecorded_protocol and (
        not metadata.get("source_dir")
        or metadata.get("timing_source_dir") == metadata.get("source_dir")
    ):
        raise ProvenanceError(
            "historical dedicated timing results must identify a separate source"
        )
    for dimension in dimensions:
        file_key = f"d{dimension}_timing_aggregate_file"
        hash_key = f"d{dimension}_timing_aggregate_sha256"
        nuisance_key = f"d{dimension}_timing_archive_nuisance_mode"
        result_file_key = f"d{dimension}_aggregate_file"
        result_hash_key = f"d{dimension}_aggregate_sha256"
        result_nuisance_key = f"d{dimension}_archive_nuisance_mode"
        protocol_key = f"d{dimension}_timing_archive_timing_protocol"
        clock_key = f"d{dimension}_timing_archive_timing_clock"
        result_protocol_key = f"d{dimension}_archive_timing_protocol"
        result_clock_key = f"d{dimension}_archive_timing_clock"
        digest = metadata.get(hash_key, "")
        if (
            not metadata.get(file_key)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ProvenanceError(
                f"run_metadata.csv has invalid d={dimension} timing source metadata"
            )
        if timing_mode == "result-aggregate" and (
            metadata.get(file_key) != metadata.get(result_file_key)
            or metadata.get(hash_key) != metadata.get(result_hash_key)
            or metadata.get(nuisance_key) != metadata.get(result_nuisance_key)
            or metadata.get(protocol_key) != metadata.get(result_protocol_key)
            or metadata.get(clock_key) != metadata.get(result_clock_key)
        ):
            raise ProvenanceError(
                f"run_metadata.csv d={dimension} same-run timing source/hash "
                "must match the result aggregate"
            )
        if not historical_unrecorded_protocol and (
            metadata.get(protocol_key) != timing_protocol
            or metadata.get(clock_key) != timing_clock
        ):
            raise ProvenanceError(
                f"run_metadata.csv d={dimension} timing protocol/clock "
                "disagrees across dimensions"
            )
        if historical_unrecorded_protocol:
            result_digest = metadata.get(result_hash_key, "")
            if (
                not metadata.get(result_file_key)
                or len(result_digest) != 64
                or any(character not in "0123456789abcdef" for character in result_digest)
                or metadata.get(result_nuisance_key) != "minKSD"
            ):
                raise ProvenanceError(
                    f"run_metadata.csv has invalid d={dimension} historical result provenance"
                )
        if (
            timing_mode == "dedicated-short-run"
            and metadata.get(nuisance_key) != "minKSD"
        ):
            raise ProvenanceError(
                f"run_metadata.csv d={dimension} dedicated timing archive "
                "must use minimum-KSD nuisance estimation"
            )


def _validate_paired_standardized_tables(
    metadata: Mapping[str, str],
    replications_path: Path,
    summary_path: Path,
) -> None:
    _require_output_hash(metadata, "replications_sha256", replications_path)
    _require_output_hash(metadata, "summary_sha256", summary_path)
    replications = _read_rows(replications_path, PAIRED_REPLICATION_COLUMNS)
    summaries = _read_rows(summary_path, PAIRED_SUMMARY_COLUMNS)
    if replications:
        raise ProvenanceError(
            "paired aggregate inputs contain no replication-level records; "
            "replications.csv must be header-only"
        )
    if metadata.get("replications_csv_rows") != "0":
        raise ProvenanceError("replications.csv row count disagrees with metadata")
    if len(summaries) != int(metadata.get("summary_csv_rows", "-1")):
        raise ProvenanceError("summary.csv row count disagrees with metadata")
    provenance_requirements = {
        "archive_format": "paired-aggregate",
        "paired_observed_datasets": "true",
        "shared_bootstrap_randomness_across_all_methods": "false",
        "paired_contrasts_reconstructible": "false",
        "replication_level_pvalues_available": "false",
        "replication_level_decisions_available": "false",
        "replications_csv_is_header_only": "true",
        "timing_rejection_fields_used": "false",
    }
    for key, expected in provenance_requirements.items():
        if metadata.get(key) != expected:
            raise ProvenanceError(
                f"run_metadata.csv must record {key}={expected}"
            )
    _validate_paired_timing_provenance(metadata)

    method_specs = {str(spec["method"]): spec for spec in SAVED_PAIRED_METHODS}
    cells: dict[tuple[int, int], set[str]] = {}
    for row in summaries:
        method = row["method"]
        spec = method_specs.get(method)
        if spec is None:
            raise ProvenanceError(f"summary.csv contains unknown method {method!r}")
        if (
            row["display_method"] != spec["display_method"]
            or row["test"] != spec["test"]
            or row["calibration"] != spec["calibration"]
            or row["nuisance_estimator"] != "minimum KSD"
            or row["monte_carlo_arm"] != PAIRED_ARM
            or row["paired_observed_datasets"] != "True"
            or row["shared_bootstrap_randomness_across_all_methods"] != "False"
            or row["paired_contrasts_reconstructible"] != "False"
            or row["replications_available"] != "False"
        ):
            raise ProvenanceError(
                f"summary.csv method/provenance metadata is wrong for {method}"
            )
        try:
            dimension = int(row["dimension"])
            theta_index = int(row["theta_index"])
            rate = float(row["rejection_rate"])
            repetitions = int(row["monte_carlo_repetitions"])
            rejected = int(row["rejected"])
            seconds = float(row["mean_seconds_per_replication"])
            mcse = float(row["mcse"])
            lower = float(row["ci_lower"])
            upper = float(row["ci_upper"])
        except ValueError as error:
            raise ProvenanceError(
                "summary.csv contains malformed numerical values"
            ) from error
        if (
            repetitions != int(metadata["monte_carlo_replications"])
            or repetitions <= 0
            or not 0.0 <= rate <= 1.0
        ):
            raise ProvenanceError("summary.csv has invalid repetitions or rejection rate")
        expected_mcse = math.sqrt(rate * (1 - rate) / repetitions)
        if not (
            0.0 <= rate <= 1.0
            and 0 <= rejected <= repetitions
            and repetitions > 0
            and math.isclose(rate, rejected / repetitions, abs_tol=1e-12)
            and math.isclose(mcse, expected_mcse, abs_tol=1e-12)
            and math.isfinite(seconds)
            and seconds > 0.0
            and -1e-12 <= lower <= rate + 1e-12
            and rate - 1e-12 <= upper <= 1.0 + 1e-12
        ):
            raise ProvenanceError("summary.csv contains invalid numerical results")
        cell_methods = cells.setdefault((dimension, theta_index), set())
        if method in cell_methods:
            raise ProvenanceError("summary.csv contains a duplicate method/cell row")
        cell_methods.add(method)
    expected_methods = set(method_specs)
    if not cells or any(methods != expected_methods for methods in cells.values()):
        raise ProvenanceError("summary.csv does not contain five methods per cell")
    if set(cells) != _selected_cells_from_metadata(metadata):
        raise ProvenanceError("summary.csv cells disagree with selected_theta_indices")
    if {dimension for dimension, _ in cells} != {
        int(value) for value in metadata["dimensions"].split(",")
    }:
        raise ProvenanceError("summary.csv dimensions disagree with timing provenance")


def run_plot(config: RunConfig, summary_path: Path) -> str:
    if not config.plot:
        return "skipped by --no-plot"
    helper = Path(__file__).resolve().parent.parent / "figure-code" / "plot_02_kef_minimum_ksd.R"
    rscript = shutil.which("Rscript")
    if not helper.is_file():
        message = f"plot helper is missing: {helper}"
        if config.require_plot:
            raise FileNotFoundError(message)
        print(f"WARNING: {message}; tabular outputs are complete", file=sys.stderr)
        return "unavailable: helper missing"
    if rscript is None:
        message = "Rscript is not available"
        if config.require_plot:
            raise RuntimeError(message)
        print(f"WARNING: {message}; tabular outputs are complete", file=sys.stderr)
        return "unavailable: Rscript missing"

    pdf_path = config.output_dir / "figure.pdf"
    png_path = config.output_dir / "figure.png"
    process = subprocess.run(
        [rscript, str(helper), str(summary_path), str(pdf_path), str(png_path)],
        cwd=repository_root(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        message = (
            f"plot helper failed with exit code {process.returncode}: "
            f"{process.stderr.strip() or process.stdout.strip()}"
        )
        if config.require_plot:
            raise RuntimeError(message)
        print(f"WARNING: {message}; tabular outputs are complete", file=sys.stderr)
        return "unavailable: plotting dependencies or device failed"
    if not pdf_path.is_file() or pdf_path.stat().st_size == 0:
        raise RuntimeError("plot helper returned success without a nonempty figure.pdf")
    if not png_path.is_file() or png_path.stat().st_size == 0:
        raise RuntimeError("plot helper returned success without a nonempty figure.png")
    if process.stdout.strip():
        print(process.stdout.strip())
    return "generated figure.pdf and figure.png"


def execute_plot_only(config: RunConfig) -> int:
    replications_path = config.output_dir / "replications.csv"
    summary_path = config.output_dir / "summary.csv"
    metadata_path = config.output_dir / "run_metadata.csv"
    missing = [
        str(path)
        for path in (replications_path, summary_path, metadata_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "--plot-only requires existing standardized output(s):\n  "
            + "\n  ".join(missing)
            + "\nRun this simulation entry point without --plot-only first."
        )
    metadata = _read_metadata(metadata_path)
    if metadata.get("status", "complete") != "complete":
        raise ProvenanceError("--plot-only requires completed simulation results")
    if metadata.get("simulation") != SIMULATION_ID:
        raise ProvenanceError("run_metadata.csv belongs to a different simulation")
    schema = metadata.get("schema_version")
    if schema not in {FRESH_SCHEMA_VERSION, SCHEMA_VERSION, LEGACY_SCHEMA_VERSION}:
        raise ProvenanceError(
            f"{metadata_path}: unknown KEF schema_version {schema!r}"
        )
    if not metadata.get("run_fingerprint"):
        raise ProvenanceError(
            f"{metadata_path}: required run_fingerprint metadata is missing"
        )
    if schema == FRESH_SCHEMA_VERSION:
        validate_fresh_standardized_tables(metadata, replications_path, summary_path)
    elif schema == SCHEMA_VERSION:
        _validate_paired_standardized_tables(
            metadata, replications_path, summary_path
        )
        if _paired_timing_protocol_is_unrecorded(metadata):
            print(
                "NOTE: saved KEF timing averages use "
                f"{metadata['timing_monte_carlo_replications']} dedicated-run "
                "replications; rejection rates use "
                f"{metadata['monte_carlo_replications']}. Timing protocol/clock "
                "were not recorded in these older CSVs.",
                file=sys.stderr,
            )
    else:
        _validate_standardized_tables(metadata, replications_path, summary_path)

    plot_status = run_plot(config, summary_path)
    metadata["figure_status"] = plot_status
    metadata["plotted_at_utc"] = datetime.now(timezone.utc).isoformat()
    helper = Path(__file__).resolve().parent.parent / "figure-code" / "plot_02_kef_minimum_ksd.R"
    if helper.is_file():
        metadata["plot_helper_sha256"] = _sha256(helper)
    pdf_path = config.output_dir / "figure.pdf"
    png_path = config.output_dir / "figure.png"
    if pdf_path.is_file():
        metadata["figure_pdf_sha256"] = _sha256(pdf_path)
    if png_path.is_file():
        metadata["figure_png_sha256"] = _sha256(png_path)
    _atomic_write_metadata(metadata_path, metadata)
    print(f"Plot-only KEF reproduction complete in {config.output_dir}: {plot_status}.")
    return 0


def execute(config: RunConfig) -> int:
    if config.plot_only:
        return execute_plot_only(config)
    return _load_fresh_simulation_module().execute(config, sys.modules[__name__])


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = config_from_args(args)
        return execute(config)
    except (FileNotFoundError, FileExistsError, ProvenanceError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
