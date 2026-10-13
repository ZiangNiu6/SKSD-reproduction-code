#!/usr/bin/env python
"""Reproduce the Gaussian--Bernoulli RBM score-matching experiment.

This is the fourth and most computationally demanding simulation in the
manuscript.  It provides a conventional command-line interface around the
self-contained scientific implementation in ``numerical-helpers/_rbm_core.py``
module.  The publication settings are the defaults; ``--check`` selects a
small deterministic smoke run.  No simulator framework, workflow engine, or
input configuration file is used.

Outputs are deliberately stable and human-readable::

    replications.csv
    summary.csv
    run_metadata.csv
    figure.pdf
    figure.png

During a run, ``checkpoint.csv`` is updated atomically after every dataset.
Use ``--resume`` to continue a compatible checkpoint or ``--force`` to start
the requested configuration again.

Figures are rendered from the saved summary by the R script in figure-code/.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import importlib.util
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPRODUCTION_ROOT = SCRIPT_PATH.parents[2]
SCIENTIFIC_SOURCE = SCRIPT_PATH.parent.parent / "numerical-helpers" / "_rbm_core.py"
HISTORICAL_SOURCE = (
    "code/rbm-python/experiments/gaussian_bernoulli_influence_vs_brueck.py"
)
DEFAULT_OUTPUT_DIR = (
    REPRODUCTION_ROOT / "results" / "simulation-analysis" / "04-rbm-score-matching"
)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    default_dir = DEFAULT_OUTPUT_DIR.parent.parent / (
        "simulation-analysis-check" if args.quick else "simulation-analysis"
    ) / DEFAULT_OUTPUT_DIR.name
    return (args.output_dir or default_dir).expanduser().resolve()

PUBLICATION = {
    "visible_dimension": 30,
    "hidden_dimension": 6,
    "sample_size": 300,
    "bootstrap_draws": 200,
    "reps": 300,
    "alpha": 0.05,
    "seed": 20260801,
    "model_seed": 2016,
    "rotation_grid": (0.0, 0.069, 0.138, 0.207, 0.276, 0.345),
    "omitted_grid": (0.0, 0.302, 0.604, 0.906, 1.208, 1.51),
    "tail_grid": (0.0, 0.25, 0.5, 0.75, 1.0, 1.25),
    "true_scale_multiplier": 1.5,
    "scale_lower": 0.05,
    "scale_upper": 4.0,
    "optimizer_maxiter": 300,
    "optimizer_gtol": 1e-7,
}

QUICK = {
    **PUBLICATION,
    "visible_dimension": 6,
    "hidden_dimension": 2,
    "sample_size": 60,
    "bootstrap_draws": 5,
    "reps": 2,
    "rotation_grid": (0.0, 0.1),
    "omitted_grid": (0.0, 0.25),
    "tail_grid": (0.0, 0.05),
    "true_scale_multiplier": 1.0,
}


def _installed_version(distribution: str) -> str:
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return "not installed"


def _load_scientific_module() -> Any:
    """Load the RBM implementation without altering ``sys.path``."""
    if not SCIENTIFIC_SOURCE.is_file():
        raise FileNotFoundError(
            "Scientific RBM implementation is missing: " f"{SCIENTIFIC_SOURCE}"
        )
    specification = importlib.util.spec_from_file_location(
        "semiparametric_ksd_rbm", SCIENTIFIC_SOURCE
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not load {SCIENTIFIC_SOURCE}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _grid(value: str | Iterable[float]) -> tuple[float, ...]:
    if isinstance(value, str):
        values = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    else:
        values = tuple(float(item) for item in value)
    if not values:
        raise argparse.ArgumentTypeError("a signal grid cannot be empty")
    if any(not math.isfinite(item) or item < 0.0 for item in values):
        raise argparse.ArgumentTypeError("signal grids must be finite and nonnegative")
    if len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("signal grids cannot contain duplicates")
    values = tuple(sorted(values))
    seed_codes = [int(round(value * 10000)) for value in values]
    if len(seed_codes) != len(set(seed_codes)):
        raise argparse.ArgumentTypeError(
            "signal values collide under the 1e-4 deterministic seed encoding"
        )
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Gaussian--Bernoulli RBM goodness-of-fit simulation with "
            "score-matching nuisance estimation"
        )
    )
    parser.add_argument(
        "--check",
        dest="quick",
        action="store_true",
        help="use the small deterministic check settings",
    )
    parser.add_argument("--quick", dest="quick", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="validate summary.csv and recreate the PDF/PNG without simulating",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"result directory (default: {DEFAULT_OUTPUT_DIR}; --check uses simulation-analysis-check/)",
    )
    parser.add_argument("--visible-dimension", type=int)
    parser.add_argument("--hidden-dimension", type=int)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--bootstrap-draws", type=int)
    parser.add_argument("--reps", type=int)
    parser.add_argument("--bandwidth", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model-seed", type=int)
    parser.add_argument("--rotation-grid", type=_grid)
    parser.add_argument("--omitted-grid", type=_grid)
    parser.add_argument("--tail-grid", type=_grid)
    parser.add_argument("--true-scale-multiplier", type=float)
    parser.add_argument("--scale-lower", type=float)
    parser.add_argument("--scale-upper", type=float)
    parser.add_argument("--optimizer-maxiter", type=int)
    parser.add_argument("--optimizer-gtol", type=float)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--resume", action="store_true", help="continue a validated checkpoint.csv"
    )
    action.add_argument(
        "--force", action="store_true", help="replace outputs for this directory"
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="write numerical outputs without running the R/ggplot2 plotter",
    )
    return parser


def _setting(args: argparse.Namespace, name: str) -> Any:
    value = getattr(args, name)
    if value is not None:
        return value
    return (QUICK if args.quick else PUBLICATION)[name]


def config_from_args(args: argparse.Namespace, rbm: Any) -> Any:
    dimension = int(_setting(args, "visible_dimension"))
    bandwidth = args.bandwidth
    if bandwidth is None:
        bandwidth = math.sqrt(2.0 * dimension)
    config = rbm.Config(
        visible_dimension=dimension,
        hidden_dimension=int(_setting(args, "hidden_dimension")),
        sample_size=int(_setting(args, "sample_size")),
        bootstrap_draws=int(_setting(args, "bootstrap_draws")),
        monte_carlo_reps=int(_setting(args, "reps")),
        fixed_bandwidth=float(bandwidth),
        alpha=float(_setting(args, "alpha")),
        seed=int(_setting(args, "seed")),
        model_seed=int(_setting(args, "model_seed")),
        signal_grid=_grid(_setting(args, "rotation_grid")),
        omitted_unit_grid=_grid(_setting(args, "omitted_grid")),
        heavy_tail_grid=_grid(_setting(args, "tail_grid")),
        true_loading_scale_multiplier=float(
            _setting(args, "true_scale_multiplier")
        ),
        loading_scale_lower=float(_setting(args, "scale_lower")),
        loading_scale_upper=float(_setting(args, "scale_upper")),
        optimizer_maxiter=int(_setting(args, "optimizer_maxiter")),
        optimizer_gtol=float(_setting(args, "optimizer_gtol")),
        output_dir=str(resolve_output_dir(args)),
    )
    rbm.validate_config(config)
    return config


def _fingerprint(config: Any, rbm: Any) -> str:
    identity = asdict(config)
    identity.pop("output_dir", None)
    identity["design_version"] = rbm.DESIGN_VERSION
    identity["schema_version"] = rbm.OUTPUT_SCHEMA_VERSION
    identity["entry_point_sha256"] = hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest()
    encoded = repr(sorted(identity.items())).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:12]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _read_metadata(path: Path) -> dict[str, str]:
    rows = _read_csv(path)
    if not rows or set(rows[0]) != {"key", "value"}:
        raise ValueError(f"metadata must contain exactly key,value columns: {path}")
    metadata = {row["key"]: row["value"] for row in rows}
    if len(metadata) != len(rows):
        raise ValueError(f"metadata contains duplicate keys: {path}")
    return metadata


def _atomic_write_metadata_values(
    path: Path, values: Iterable[tuple[str, object]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("key", "value"))
        writer.writerows(values)
    temporary.replace(path)


def _record_figure_hashes(metadata_path: Path, output_dir: Path) -> None:
    pdf_path = output_dir / "figure.pdf"
    png_path = output_dir / "figure.png"
    if not pdf_path.is_file() or not png_path.is_file():
        raise FileNotFoundError("cannot record hashes before both figures exist")
    metadata = _read_metadata(metadata_path)
    metadata["figure_pdf_sha256"] = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    metadata["figure_png_sha256"] = hashlib.sha256(png_path.read_bytes()).hexdigest()
    _atomic_write_metadata_values(metadata_path, metadata.items())


def _write_metadata(
    path: Path,
    config: Any,
    rbm: Any,
    fingerprint: str,
    source_sha256: str,
    computed_datasets: int,
    quick: bool,
    replications_sha256: str,
    summary_sha256: str,
) -> None:
    values: list[tuple[str, object]] = [
        ("schema_version", rbm.OUTPUT_SCHEMA_VERSION),
        ("design_version", rbm.DESIGN_VERSION),
        ("run_fingerprint", fingerprint),
        ("profile", "quick" if quick else "publication-or-overridden"),
        ("scientific_source", SCIENTIFIC_SOURCE.name),
        ("scientific_source_sha256", source_sha256),
        ("historical_source", HISTORICAL_SOURCE),
        ("entry_point_sha256", hashlib.sha256(SCRIPT_PATH.read_bytes()).hexdigest()),
        ("computed_datasets", computed_datasets),
        ("replications_sha256", replications_sha256),
        ("summary_sha256", summary_sha256),
        ("python_version", platform.python_version()),
        ("numpy_version", np.__version__),
        ("scipy_version", _installed_version("scipy")),
        ("plotting_backend", "R/ggplot2"),
        ("brueck_correction", rbm.BRUECK_CORRECTION_VERSION),
        ("nuisance_estimator", "Hyvarinen score matching"),
        ("fitted_model_draws", 0),
        ("checkpoint", "atomic CSV keyed by alternative, signal, rep"),
        ("timing_protocol", "standalone-test-v1"),
        ("timing_clock", "perf_counter"),
        ("timing_unit", "seconds per simulated dataset"),
        (
            "timing_shared_scope",
            "RBF geometry|observed score/derivative/Stein Gram|Efron count "
            "generation",
        ),
        ("timing_observed_setup_scope", "observed nuisance fit|fixed bandwidth"),
        (
            "timing_total_formula",
            "observed fit + shared preprocessing + method statistic/bootstrap/p-value",
        ),
        ("timing_excludes", "observed-data simulation|checkpoint and output I/O"),
        ("timing_method_order", "SKSD-influence|Brueck-corrected"),
        ("platform", platform.platform()),
        ("cpu_architecture", platform.machine() or "unknown"),
        ("cpu_processor", platform.processor() or "unknown"),
        ("logical_cpu_count", os.cpu_count() or "unknown"),
        ("OMP_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "unset")),
        (
            "OPENBLAS_NUM_THREADS",
            os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        ),
        ("MKL_NUM_THREADS", os.environ.get("MKL_NUM_THREADS", "unset")),
        (
            "VECLIB_MAXIMUM_THREADS",
            os.environ.get("VECLIB_MAXIMUM_THREADS", "unset"),
        ),
    ]
    for key, value in asdict(config).items():
        if key == "output_dir":
            rendered: object = str(Path(value).resolve())
        elif isinstance(value, (tuple, list)):
            rendered = "|".join(f"{float(item):g}" for item in value)
        else:
            rendered = value
        values.append((f"config.{key}", rendered))
    _atomic_write_metadata_values(path, values)


def _check_collisions(output_dir: Path, resume: bool, force: bool) -> list[dict[str, str]]:
    finals = [
        output_dir / "replications.csv",
        output_dir / "summary.csv",
        output_dir / "run_metadata.csv",
        output_dir / "figure.pdf",
        output_dir / "figure.png",
    ]
    checkpoint = output_dir / "checkpoint.csv"
    existing_finals = [path for path in finals if path.exists()]
    if resume:
        replications_path = output_dir / "replications.csv"
        if replications_path.exists():
            raise FileExistsError(
                f"cannot resume a completed run: {replications_path}"
            )
        if not checkpoint.exists():
            raise FileNotFoundError(f"resume checkpoint is absent: {checkpoint}")
        return _read_csv(checkpoint)
    collisions = existing_finals + ([checkpoint] if checkpoint.exists() else [])
    if collisions and not force:
        raise FileExistsError(
            "result directory already contains this run; use --resume or --force: "
            + ", ".join(str(path) for path in collisions)
        )
    if force:
        for path in collisions:
            path.unlink()
    return []


def _draw_efron_counts(
    rbm: Any,
    config: Any,
    alternative: str,
    signal: float,
    repetition: int,
) -> tuple[np.ndarray, float]:
    """Draw the shared bootstrap randomness and measure its wall-clock cost.

    Efron counts feed the Brueck refits; centered Gaussian multipliers feed
    the influence-adjusted test.
    """
    started = time.perf_counter()
    rng = np.random.default_rng(
        rbm._seed(config, alternative, signal, repetition, 1)
    )
    counts = rng.multinomial(
        config.sample_size,
        np.full(config.sample_size, 1.0 / config.sample_size),
        size=config.bootstrap_draws,
    )
    weights = rng.standard_normal(
        (config.bootstrap_draws, config.sample_size)
    )
    weights -= weights.mean(axis=1, keepdims=True)
    return counts, weights, time.perf_counter() - started


def run_simulation(args: argparse.Namespace, rbm: Any, config: Any) -> tuple[Path, Path]:
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "checkpoint.csv"
    replications_path = output_dir / "replications.csv"
    summary_path = output_dir / "summary.csv"
    metadata_path = output_dir / "run_metadata.csv"
    raw_rows: list[dict[str, object]] = list(
        _check_collisions(output_dir, args.resume, args.force)
    )
    source_sha256 = hashlib.sha256(SCIENTIFIC_SOURCE.read_bytes()).hexdigest()
    fingerprint = _fingerprint(config, rbm)
    if raw_rows:
        rbm._validate_resume_rows(raw_rows, config, source_sha256)
        wrong_fingerprint = [
            row
            for row in raw_rows
            if row.get("run_fingerprint", "") != fingerprint
        ]
        if wrong_fingerprint:
            raise ValueError("checkpoint belongs to a different configuration")
        # A crash during finalization can leave derived finals beside the
        # authoritative checkpoint.  They are regenerated from that validated
        # checkpoint and must not be mistaken for a completed run.
        for name in ("summary.csv", "run_metadata.csv", "figure.pdf", "figure.png"):
            derived = output_dir / name
            if derived.exists():
                derived.unlink()

    completed_keys = {
        (str(row["alternative"]), float(row["signal"]), int(row["rep"]))
        for row in raw_rows
    }
    cells = rbm.computational_cells(config)
    total = len(cells) * config.monte_carlo_reps
    design = rbm.RBMDesign(config)
    completed = len(raw_rows)
    print(
        "RBM score-matching simulation: "
        f"d={config.visible_dimension}, m={config.hidden_dimension}, "
        f"n={config.sample_size}, B={config.bootstrap_draws}, "
        f"reps={config.monte_carlo_reps}, cells={len(cells)}",
        flush=True,
    )

    for alternative, signal in cells:
        dgp_alternative = (
            rbm.ALTERNATIVE_ROTATION
            if alternative == rbm.SHARED_NULL
            else alternative
        )
        for repetition in range(config.monte_carlo_reps):
            key = (alternative, signal, repetition)
            if key in completed_keys:
                continue
            data_rng = np.random.default_rng(
                rbm._seed(config, alternative, signal, repetition, 0)
            )
            x = design.sample_exact(
                data_rng, config.sample_size, signal, dgp_alternative
            )
            fit_started = time.perf_counter()
            fitted = rbm.fit_score_matching(x, design, config)
            retried = False
            if fitted.boundary_hit or not fitted.success:
                retried = True
                fitted = rbm.fit_score_matching(
                    x,
                    design,
                    config,
                    start=design.initial_theta(),
                    strict=True,
                )
            fit_seconds = time.perf_counter() - fit_started
            if not fitted.success or fitted.boundary_hit:
                raise RuntimeError(
                    "invalid observed score-matching fit at "
                    f"{alternative}, signal={signal:g}, rep={repetition}: "
                    f"{fitted.message}"
                )
            counts, multiplier_weights, counts_seconds = _draw_efron_counts(
                rbm, config, alternative, signal, repetition
            )
            result = rbm.paired_empirical_tests(
                x, design, config, fitted, counts, multiplier_weights
            )
            raw_rows.append(
                {
                    "schema_version": rbm.OUTPUT_SCHEMA_VERSION,
                    "design_version": rbm.DESIGN_VERSION,
                    "source_sha256": source_sha256,
                    "run_fingerprint": fingerprint,
                    "alternative": alternative,
                    "signal": signal,
                    "rep": repetition,
                    "pvalue_influence": result.pvalue_influence,
                    "pvalue_brueck_corrected": result.pvalue_brueck_corrected,
                    "seconds_influence": result.seconds_influence,
                    "seconds_brueck_corrected": result.seconds_brueck_corrected,
                    "observed_fit_seconds": fit_seconds,
                    "shared_preprocess_seconds": (
                        counts_seconds + result.shared_preprocess_seconds
                    ),
                    "observed_fit_iterations": fitted.iterations,
                    "observed_fit_retried": int(retried),
                    "observed_fit_gradient_norm": fitted.gradient_norm,
                    "observed_fit_boundary": int(fitted.boundary_hit),
                    "observed_v": result.observed_v,
                    "observed_u": result.observed_u,
                    "influence_hessian_condition": result.influence_hessian_condition,
                    "brueck_refits": result.brueck_refits,
                    "brueck_mean_iterations": result.brueck_mean_iterations,
                    "brueck_max_iterations": result.brueck_max_iterations,
                    "brueck_retry_rate": result.brueck_retry_rate,
                }
            )
            completed_keys.add(key)
            _write_csv(checkpoint, raw_rows)
            completed += 1
            print(
                f"[{completed}/{total}] {alternative} signal={signal:g} "
                f"rep={repetition} p_IF={result.pvalue_influence:.3f} "
                f"p_B={result.pvalue_brueck_corrected:.3f}",
                flush=True,
            )

    _write_csv(summary_path, rbm.summarize(raw_rows, config))
    replications_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    summary_sha256 = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    _write_metadata(
        metadata_path,
        config,
        rbm,
        fingerprint,
        source_sha256,
        total,
        args.quick,
        replications_sha256,
        summary_sha256,
    )
    checkpoint.replace(replications_path)
    print(f"wrote {replications_path}")
    print(f"wrote {summary_path}")
    print(f"wrote {metadata_path}")
    return summary_path, metadata_path


def _as_number(row: dict[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, ValueError) as error:
        raise ValueError(f"summary.csv has an invalid {key!r} value") from error
    if not math.isfinite(value):
        raise ValueError(f"summary.csv has a non-finite {key!r} value")
    return value


def validate_saved_results(
    output_dir: Path,
) -> tuple[list[dict[str, str]], Any, dict[str, str]]:
    """Validate historical CSVs and their recorded timing, without simulating.

    The flattened ``source_metadata.csv`` keeps original settings/provenance;
    numeric grids are pipe-separated. Saved numerical tables stay untouched.
    """
    output_dir = Path(output_dir)
    manifest = _read_metadata(output_dir / "run_metadata.csv")
    expected_manifest = {
        "schema_version": "saved-csv-v1", "study": "04-rbm-score-matching",
        "workflow": "saved-results", "run_complete": "true",
        "timing_protocol": "historical-recorded",
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise ValueError("invalid saved RBM CSV manifest")
    for filename, key in (
        ("replications.csv", "replications_sha256"),
        ("summary.csv", "summary_sha256"),
        ("source_metadata.csv", "source_metadata_sha256"),
    ):
        path = output_dir / filename
        if not path.is_file() or manifest.get(key) != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"saved {filename} checksum does not match manifest")
    source = _read_metadata(output_dir / "source_metadata.csv")
    rbm = _load_scientific_module()
    if (
        source.get("schema_version") != str(rbm.OUTPUT_SCHEMA_VERSION)
        or source.get("design_version") != rbm.DESIGN_VERSION
        or source.get("calibration.brueck_correction_version") != rbm.BRUECK_CORRECTION_VERSION
        or not source.get("config_fingerprint")
        or len(source.get("source_sha256_at_start", "")) != 64
    ):
        raise ValueError("saved RBM source lacks corrected-run provenance")
    integer_fields = {
        "visible_dimension", "hidden_dimension", "sample_size", "bootstrap_draws",
        "monte_carlo_reps", "seed", "model_seed", "optimizer_maxiter",
    }
    grid_fields = {"signal_grid", "omitted_unit_grid", "heavy_tail_grid"}
    values: dict[str, object] = {}
    try:
        for name in rbm.Config.__dataclass_fields__:
            if name == "output_dir":
                values[name] = str(output_dir.resolve())
                continue
            value = source[f"config.{name}"]
            if name in grid_fields:
                grid = tuple(float(item) for item in value.split("|"))
                if not grid or tuple(sorted(set(grid))) != grid or any(not math.isfinite(item) or item < 0.0 for item in grid):
                    raise ValueError(f"invalid {name}")
                values[name] = grid
            elif name in integer_fields:
                values[name] = int(value)
            else:
                values[name] = float(value)
                if not math.isfinite(values[name]):
                    raise ValueError(f"non-finite {name}")
        config = rbm.Config(**values)
        rbm.validate_config(config)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("saved RBM source has invalid recorded settings") from error
    raw = _read_csv(output_dir / "replications.csv")
    keys = rbm._validate_resume_rows(raw, config, source["source_sha256_at_start"])
    expected_cells = len(rbm.computational_cells(config))
    expected_count = expected_cells * config.monte_carlo_reps
    if (
        len(keys) != expected_count
        or source.get("execution.computed_datasets") != str(expected_count)
        or source.get("execution.computational_cells") != str(expected_cells)
    ):
        raise ValueError("saved RBM CSVs do not cover the complete recorded design")
    for row in raw:
        for field, value in row.items():
            if field in {"alternative", "design_version", "source_sha256"}:
                continue
            numeric = float(value)
            if not math.isfinite(numeric):
                raise ValueError(f"saved RBM replication contains non-finite {field}")
            if field.startswith("pvalue_") and (
                not 0.0 < numeric <= 1.0
                or not math.isclose(numeric * (config.bootstrap_draws + 1), round(numeric * (config.bootstrap_draws + 1)), abs_tol=1e-9, rel_tol=0.0)
            ):
                raise ValueError(f"saved RBM replication contains invalid {field}")
            if "seconds" in field and numeric < 0.0:
                raise ValueError("saved RBM replication contains negative timing")
    expected = rbm.summarize(raw, config)
    summary = _read_csv(output_dir / "summary.csv")
    if len(summary) != len(expected):
        raise ValueError("saved RBM summary has missing or extra rows")
    for stored, target in zip(summary, expected):
        if tuple(stored) != tuple(target):
            raise ValueError("saved RBM summary has an invalid header")
        for field, value in target.items():
            if isinstance(value, str):
                agrees = stored[field] == value
            else:
                numeric = _as_number(stored, field)
                agrees = math.isclose(numeric, float(value), rel_tol=1e-12, abs_tol=1e-14)
            if not agrees:
                raise ValueError(f"saved RBM summary disagrees with replications on {field}")
    return summary, config, manifest


def plot_summary(
    summary_path: Path,
    output_dir: Path,
    alpha: float,
    expected_grids: dict[str, tuple[float, ...]] | None = None,
) -> None:
    """Validate and plot the two-method, three-alternative summary."""
    rows = _read_csv(summary_path)
    if not rows:
        raise ValueError(f"summary is empty: {summary_path}")
    required = {
        "alternative",
        "signal",
        "method",
        "rejection_rate",
        "rejection_ci_lower",
        "rejection_ci_upper",
        "total_seconds_per_dataset",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"summary is missing columns: {sorted(missing)}")
    methods = ("SKSD-influence", "Brueck-corrected")
    alternatives = (
        "loading-rotation",
        "omitted-hidden-unit",
        "heavy-tailed-visible",
    )
    if {row["method"] for row in rows} != set(methods):
        raise ValueError("summary does not contain exactly the two expected methods")
    if {row["alternative"] for row in rows} != set(alternatives):
        raise ValueError("summary does not contain exactly the three RBM alternatives")
    keyed: set[tuple[str, float, str]] = set()
    for row in rows:
        for key in (
            "signal",
            "rejection_rate",
            "rejection_ci_lower",
            "rejection_ci_upper",
            "total_seconds_per_dataset",
        ):
            _as_number(row, key)
        alternative = row["alternative"]
        method = row["method"]
        signal = _as_number(row, "signal")
        key = (alternative, signal, method)
        if key in keyed:
            raise ValueError(f"summary contains a duplicate method/cell row: {key}")
        keyed.add(key)
        rate = _as_number(row, "rejection_rate")
        lower = _as_number(row, "rejection_ci_lower")
        upper = _as_number(row, "rejection_ci_upper")
        seconds = _as_number(row, "total_seconds_per_dataset")
        if not (0.0 <= lower <= rate <= upper <= 1.0):
            raise ValueError("summary contains an invalid rejection rate or interval")
        if seconds <= 0.0:
            raise ValueError("summary contains a nonpositive runtime")
    for alternative in alternatives:
        signals = {
            _as_number(row, "signal")
            for row in rows
            if row["alternative"] == alternative
        }
        if expected_grids is not None and signals != set(expected_grids[alternative]):
            raise ValueError(f"summary grid is incomplete for {alternative}")
        for signal in signals:
            present = {
                row["method"]
                for row in rows
                if row["alternative"] == alternative
                and _as_number(row, "signal") == signal
            }
            if present != set(methods):
                raise ValueError(
                    f"summary does not contain both methods for {alternative}/{signal:g}"
                )

    plot_script = SCRIPT_PATH.parent.parent / "figure-code" / "plot_04_rbm_score_matching.R"
    if not plot_script.is_file():
        raise FileNotFoundError(f"R plotting script is missing: {plot_script}")
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "Rscript is required for figures. Install R and the packages in "
            "reproduction-code/README.md, then rerun with --plot-only."
        )
    if not summary_path.is_file():
        raise FileNotFoundError(f"plotting summary is missing: {summary_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "figure.pdf"
    png_path = output_dir / "figure.png"
    try:
        subprocess.run(
            [
                rscript, str(plot_script),
                str(summary_path.resolve()), str(pdf_path.resolve()),
                str(png_path.resolve()), format(alpha, ".17g"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or "no diagnostic output").strip()
        raise RuntimeError(f"R plotting failed ({plot_script.name}): {details}") from error
    if not all(
        path.is_file() and path.stat().st_size > 0 for path in (pdf_path, png_path)
    ):
        raise RuntimeError(f"R plotting did not create both figures in {output_dir}")
    print(f"wrote {pdf_path}")
    print(f"wrote {png_path}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = resolve_output_dir(args)
    if args.plot_only:
        if args.resume or args.skip_plot:
            raise ValueError("--plot-only cannot be combined with --resume/--skip-plot")
        summary_path = output_dir / "summary.csv"
        if not summary_path.is_file():
            raise FileNotFoundError(f"plot-only summary is absent: {summary_path}")
        metadata_path = output_dir / "run_metadata.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"plot-only metadata is absent: {metadata_path}")
        metadata = _read_metadata(metadata_path)
        if metadata.get("schema_version") == "saved-csv-v1":
            _summary, saved_config, metadata = validate_saved_results(output_dir)
            # Keep the existing override checks and plotter below. These
            # transient fields describe saved settings; never rewrite them as
            # fresh-run provenance in the historical manifest.
            metadata = dict(metadata)
            metadata["run_fingerprint"] = "saved-results"
            metadata["config.alpha"] = str(saved_config.alpha)
            for key in ("signal_grid", "omitted_unit_grid", "heavy_tail_grid"):
                metadata[f"config.{key}"] = "|".join(str(value) for value in getattr(saved_config, key))
        if not metadata.get("schema_version") or not metadata.get("run_fingerprint"):
            raise ValueError("plot-only metadata lacks schema/fingerprint provenance")
        replications_path = output_dir / "replications.csv"
        if not replications_path.is_file():
            raise FileNotFoundError(
                f"plot-only replications are absent: {replications_path}"
            )
        for key, path in (
            ("replications_sha256", replications_path),
            ("summary_sha256", summary_path),
        ):
            recorded = metadata.get(key)
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
            if not recorded or recorded != observed:
                raise ValueError(f"plot-only {path.name} checksum does not match metadata")
        recorded_alpha = float(metadata.get("config.alpha", PUBLICATION["alpha"]))
        if args.alpha is not None and not math.isclose(
            float(args.alpha), recorded_alpha, rel_tol=0.0, abs_tol=1e-15
        ):
            raise ValueError(
                "--alpha cannot change in plot-only mode because rejection rates "
                "were computed at the recorded alpha"
            )
        override_names = (
            "visible_dimension",
            "hidden_dimension",
            "sample_size",
            "bootstrap_draws",
            "reps",
            "bandwidth",
            "seed",
            "model_seed",
            "rotation_grid",
            "omitted_grid",
            "tail_grid",
            "true_scale_multiplier",
            "scale_lower",
            "scale_upper",
            "optimizer_maxiter",
            "optimizer_gtol",
        )
        supplied = [name for name in override_names if getattr(args, name) is not None]
        if supplied:
            raise ValueError(
                "plot-only reads the recorded configuration; remove simulation "
                "overrides: " + ", ".join(supplied)
            )

        def recorded_grid(key: str) -> tuple[float, ...]:
            value = metadata.get(key)
            if not value:
                raise ValueError(f"plot-only metadata is missing {key}")
            return tuple(float(item) for item in value.split("|") if item)

        expected_grids = {
            "loading-rotation": recorded_grid("config.signal_grid"),
            "omitted-hidden-unit": recorded_grid("config.omitted_unit_grid"),
            "heavy-tailed-visible": recorded_grid("config.heavy_tail_grid"),
        }
        plot_summary(
            summary_path, output_dir, recorded_alpha, expected_grids=expected_grids
        )
        _record_figure_hashes(metadata_path, output_dir)
        return 0

    rbm = _load_scientific_module()
    config = config_from_args(args, rbm)
    summary_path, _ = run_simulation(args, rbm, config)
    if not args.skip_plot:
        plot_summary(
            summary_path,
            output_dir,
            config.alpha,
            expected_grids=rbm.alternative_grids(config),
        )
        _record_figure_hashes(output_dir / "run_metadata.csv", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
