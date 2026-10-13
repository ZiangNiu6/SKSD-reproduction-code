#!/usr/bin/env python3
"""Reproduce the conditional-Gaussian score-matching simulation.

This is a direct, single-process entry point for the third simulation in the
manuscript. It loads its self-contained numerical implementation from
``numerical-helpers/_conditional_gaussian_core.py`` and does not require the
development code tree.

Publication defaults are d=8, n=100, B=300, R=300, Gibbs burn-in 600, fixed
RBF bandwidth 2.35, and base seed 20260731.  ``--check`` selects a small,
deterministic smoke configuration.  Every completed replication is written
atomically, so ``--resume`` can continue an interrupted matching run without
changing any data or bootstrap stream.

Figures are rendered from the saved summary by the R script in figure-code/.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import csv
from datetime import datetime, timezone
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
from typing import Iterable, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_ROOT = SCRIPT_PATH.parents[2]
SCIENTIFIC_SOURCE = SCRIPT_PATH.parent.parent / "numerical-helpers" / "_conditional_gaussian_core.py"
OUTPUT_SCHEMA_VERSION = 1


def _installed_version(distribution: str) -> str:
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return "not installed"


DEFAULT_OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "results"
    / "simulation-analysis"
    / "03-conditional-gaussian-score-matching"
)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    default_dir = DEFAULT_OUTPUT_DIR.parent.parent / (
        "simulation-analysis-check" if args.quick else "simulation-analysis"
    ) / DEFAULT_OUTPUT_DIR.name
    return (args.output_dir or default_dir).expanduser().resolve()

PUBLICATION_DEFAULTS = {
    "dimension": 8,
    "sample_size": 100,
    "bootstrap_draws": 300,
    "monte_carlo_reps": 300,
    "gibbs_burn": 600,
    "fixed_bandwidth": 2.35,
    "alpha": 0.05,
    "seed": 20260731,
    "model_seed": 2026,
    "score_matching_lower": -50.0,
    "score_matching_upper": 0.0,
    "rewire_grid": (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5),
    "community_grid": (0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
    "matching_grid": (0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
}

QUICK_DEFAULTS = {
    **PUBLICATION_DEFAULTS,
    "sample_size": 40,
    "bootstrap_draws": 19,
    "monte_carlo_reps": 2,
    "gibbs_burn": 60,
    "rewire_grid": (0.0, 0.5),
    "community_grid": (0.0, 1.0),
    "matching_grid": (0.0, 1.0),
}

OUTPUT_NAMES = (
    "replications.csv",
    "summary.csv",
    "run_metadata.csv",
    "figure.pdf",
    "figure.png",
)


def _load_scientific_core():
    """Load the numerical helper without modifying ``sys.path``."""
    if not SCIENTIFIC_SOURCE.is_file():
        raise FileNotFoundError(
            f"conditional-Gaussian numerical helper not found: "
            f"{SCIENTIFIC_SOURCE}"
        )
    module_name = "_semiksd_conditional_gaussian_scientific_core"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, SCIENTIFIC_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {SCIENTIFIC_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


CORE = _load_scientific_core()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_grid(text: str, name: str) -> tuple[float, ...]:
    try:
        values = tuple(
            float(piece.strip())
            for piece in text.split(",")
            if piece.strip()
        )
    except ValueError as error:
        raise ValueError(f"{name} must be a comma-separated numeric grid") from error
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"{name} must contain finite nonnegative values")
    values = tuple(sorted(set(values)))
    if 0.0 not in values:
        raise ValueError(f"{name} must contain the common null value 0")
    seed_codes = [int(round(value * 10000)) for value in values]
    if len(seed_codes) != len(set(seed_codes)):
        raise ValueError(
            f"{name} contains values that collide under the 1e-4 seed encoding"
        )
    return values


def _grid_text(values: Iterable[float]) -> str:
    return ",".join(format(float(value), ".17g") for value in values)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the conditional-Gaussian structural goodness-of-fit "
            "simulation with score-matching nuisance estimation."
        )
    )
    parser.add_argument(
        "--check",
        dest="quick",
        action="store_true",
        help="use a small deterministic check configuration",
    )
    parser.add_argument("--quick", dest="quick", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="validate completed outputs and regenerate only figure.pdf/png",
    )
    parser.add_argument("--dimension", type=int, help="dimension (publication: 8)")
    parser.add_argument("--sample-size", type=int, help="sample size (publication: 100)")
    parser.add_argument(
        "--bootstrap-draws", type=int, help="bootstrap draws B (publication: 300)"
    )
    parser.add_argument(
        "--replications", type=int, help="Monte Carlo replications R (publication: 300)"
    )
    parser.add_argument("--burn", type=int, help="Gibbs burn-in (publication: 600)")
    parser.add_argument(
        "--bandwidth", type=float, help="fixed RBF bandwidth (publication: 2.35)"
    )
    parser.add_argument("--alpha", type=float, help="nominal level (publication: 0.05)")
    parser.add_argument("--seed", type=int, help="base seed (publication: 20260731)")
    parser.add_argument("--model-seed", type=int, help="model seed (publication: 2026)")
    parser.add_argument(
        "--sm-lower", type=float, help="score-matching lower bound (publication: -50)"
    )
    parser.add_argument(
        "--sm-upper", type=float, help="score-matching upper bound (publication: 0)"
    )
    parser.add_argument(
        "--rewire-grid",
        help="comma-separated second-neighbor rewiring signals",
    )
    parser.add_argument(
        "--community-grid",
        help="comma-separated within-block closure signals",
    )
    parser.add_argument(
        "--matching-grid",
        help="comma-separated diametric matching signals",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"output directory (default: {DEFAULT_OUTPUT_DIR}; --check uses simulation-analysis-check/)",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--resume",
        action="store_true",
        help="validate and continue a matching interrupted run",
    )
    action.add_argument(
        "--force",
        action="store_true",
        help="replace outputs for this exact entry point and start over",
    )
    return parser


def resolve_config(args: argparse.Namespace):
    defaults = QUICK_DEFAULTS if args.quick else PUBLICATION_DEFAULTS

    def selected(argument: str, key: str):
        value = getattr(args, argument)
        return defaults[key] if value is None else value

    rewire_grid = (
        defaults["rewire_grid"]
        if args.rewire_grid is None
        else _parse_grid(args.rewire_grid, "--rewire-grid")
    )
    community_grid = (
        defaults["community_grid"]
        if args.community_grid is None
        else _parse_grid(args.community_grid, "--community-grid")
    )
    matching_grid = (
        defaults["matching_grid"]
        if args.matching_grid is None
        else _parse_grid(args.matching_grid, "--matching-grid")
    )
    config = CORE.Config(
        dimension=selected("dimension", "dimension"),
        sample_size=selected("sample_size", "sample_size"),
        bootstrap_draws=selected("bootstrap_draws", "bootstrap_draws"),
        monte_carlo_reps=selected("replications", "monte_carlo_reps"),
        gibbs_burn=selected("burn", "gibbs_burn"),
        fixed_bandwidth=selected("bandwidth", "fixed_bandwidth"),
        alpha=selected("alpha", "alpha"),
        seed=selected("seed", "seed"),
        model_seed=selected("model_seed", "model_seed"),
        score_matching_ridge=0.0,
        score_matching_lower=selected("sm_lower", "score_matching_lower"),
        score_matching_upper=selected("sm_upper", "score_matching_upper"),
        rewire_grid=rewire_grid,
        community_grid=community_grid,
        matching_grid=matching_grid,
        output_dir=str(resolve_output_dir(args)),
    )
    CORE.validate_config(config)
    design = CORE.StructuralDesign(config)
    CORE.validate_design_grids(config, design)
    return config, design


def _config_items(config) -> list[tuple[str, str]]:
    values = asdict(config)
    values.pop("output_dir", None)
    answer: list[tuple[str, str]] = []
    for key in sorted(values):
        value = values[key]
        if isinstance(value, (tuple, list)):
            rendered = _grid_text(value)
        elif isinstance(value, float):
            rendered = format(value, ".17g")
        else:
            rendered = str(value)
        answer.append((key, rendered))
    return answer


def config_fingerprint(config) -> str:
    payload = "\n".join(f"{key}={value}" for key, value in _config_items(config))
    payload += f"\ndesign_version={CORE.DESIGN_VERSION}"
    payload += f"\nscientific_schema={CORE.OUTPUT_SCHEMA_VERSION}"
    payload += f"\nscientific_source_sha256={_sha256(SCIENTIFIC_SOURCE)}"
    payload += f"\nentry_point_sha256={_sha256(SCRIPT_PATH)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sort_rows(rows: Iterable[dict]) -> list[dict]:
    alternative_order = {
        alternative: index for index, alternative in enumerate(CORE.ALTERNATIVES)
    }
    return sorted(
        rows,
        key=lambda row: (
            alternative_order[row["alternative"]],
            float(row["delta"]),
            int(row["rep"]),
        ),
    )


def _display_rows(computational_rows: Iterable[dict]) -> list[dict]:
    return _sort_rows(CORE.expand_shared_null_rows(list(computational_rows)))


def _summary_rows(computational_rows: Iterable[dict], config) -> list[dict]:
    rows = _display_rows(computational_rows)
    return CORE.summarize_rows(rows, config) if rows else []


def _metadata_rows(metadata: dict[str, object]) -> list[dict[str, str]]:
    return [
        {"key": key, "value": str(metadata[key])}
        for key in sorted(metadata)
    ]


def _read_metadata(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["key", "value"]:
            raise ValueError(f"invalid metadata header in {path}")
        rows = list(reader)
    if any(not row["key"] for row in rows):
        raise ValueError(f"blank metadata key in {path}")
    keys = [row["key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate metadata key in {path}")
    return {row["key"]: row["value"] for row in rows}


def _coerce_replication_row(row: dict[str, str]) -> dict:
    answer: dict[str, object] = {"alternative": row["alternative"]}
    answer["rep"] = int(row["rep"])
    for field in CORE.RAW_FIELDS:
        if field in {"alternative", "rep"}:
            continue
        answer[field] = float(row[field])
    return answer


def _load_resume_rows(path: Path, config) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(CORE.RAW_FIELDS):
            raise ValueError(f"invalid replication header in {path}")
        rows = [_coerce_replication_row(row) for row in reader]

    expected_cells = set(CORE.computational_cells(config))
    canonical: list[dict] = []
    canonical_null: dict[int, dict] = {}
    clones: list[dict] = []
    seen: set[tuple[str, float, int]] = set()
    for row in rows:
        alternative = str(row["alternative"])
        delta = float(row["delta"])
        repetition = int(row["rep"])
        key = (alternative, delta, repetition)
        if key in seen:
            raise ValueError(f"duplicate replication key in {path}: {key}")
        seen.add(key)
        if alternative not in CORE.ALTERNATIVES:
            raise ValueError(f"unknown alternative in {path}: {alternative}")
        if not 0 <= repetition < config.monte_carlo_reps:
            raise ValueError(f"replication index outside configured range: {key}")
        if delta == 0.0 and alternative != CORE.ALTERNATIVES[0]:
            clones.append(row)
            continue
        if (alternative, delta) not in expected_cells:
            raise ValueError(f"unexpected simulation cell in {path}: {key}")
        canonical.append(row)
        if alternative == CORE.ALTERNATIVES[0] and delta == 0.0:
            canonical_null[repetition] = row

    comparison_fields = [
        field for field in CORE.RAW_FIELDS if field != "alternative"
    ]
    for clone in clones:
        source = canonical_null.get(int(clone["rep"]))
        if source is None or any(clone[field] != source[field] for field in comparison_fields):
            raise ValueError("shared-null clone does not match its canonical row")
    numeric_fields = CORE.RAW_FIELDS[3:]
    bootstrap_scale = config.bootstrap_draws + 1
    for row in canonical:
        values = np.asarray([row[field] for field in numeric_fields], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("replications.csv contains a non-finite value")
        for field in (
            "pvalue_influence",
            "pvalue_brueck",
            "pvalue_brueck_corrected",
        ):
            pvalue = float(row[field])
            if not 0.0 < pvalue <= 1.0 or not np.isclose(
                pvalue * bootstrap_scale,
                round(pvalue * bootstrap_scale),
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(f"replications.csv has an invalid {field}")
        if any(
            float(row[field]) < 0.0
            for field in (
                "seconds_influence",
                "seconds_brueck",
                "seconds_brueck_corrected",
            )
        ):
            raise ValueError("replications.csv contains a negative runtime")
        if float(row["score_matching_condition"]) <= 0.0:
            raise ValueError("replications.csv contains an invalid condition number")
        if not (
            config.score_matching_lower - 1e-10
            <= float(row["min_fitted_ring"])
            <= float(row["max_fitted_ring"])
            <= config.score_matching_upper + 1e-10
        ):
            raise ValueError("replications.csv contains an invalid fitted ring")
        for field in (
            "score_matching_boundary_hit",
            "brueck_boundary_refit_rate",
        ):
            if not 0.0 <= float(row[field]) <= 1.0:
                raise ValueError(f"replications.csv has an invalid {field}")
    return _sort_rows(canonical)


def _base_metadata(
    config,
    profile: str,
    started_utc: str,
    elapsed_seconds: float,
    resume_count: int,
    computational_rows: Sequence[dict],
    run_complete: bool,
) -> dict[str, object]:
    expected_computational = (
        len(CORE.computational_cells(config)) * config.monte_carlo_reps
    )
    expected_display = sum(
        len(CORE.grid_for(config, alternative))
        for alternative in CORE.ALTERNATIVES
    ) * config.monte_carlo_reps
    metadata: dict[str, object] = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "scientific_output_schema_version": CORE.OUTPUT_SCHEMA_VERSION,
        "design_version": CORE.DESIGN_VERSION,
        "brueck_correction_version": CORE.BRUECK_CORRECTION_VERSION,
        "profile": profile,
        "config_fingerprint": config_fingerprint(config),
        "run_fingerprint": config_fingerprint(config),
        "started_utc": started_utc,
        "updated_utc": _utc_now(),
        "elapsed_seconds": format(elapsed_seconds, ".9g"),
        "resume_count": resume_count,
        "run_complete": str(bool(run_complete)).lower(),
        "expected_computational_rows": expected_computational,
        "completed_computational_rows": len(computational_rows),
        "expected_display_rows": expected_display,
        "completed_display_rows": len(_display_rows(computational_rows)),
        "scientific_source": SCIENTIFIC_SOURCE.name,
        "scientific_source_sha256": _sha256(SCIENTIFIC_SOURCE),
        "entry_point_sha256": _sha256(SCRIPT_PATH),
        "seed_scheme": (
            "SeedSequence([base_seed, alternative_code, "
            "round(delta*10000), replication, stream])"
        ),
        "data_stream": 1,
        "bootstrap_stream": 2,
        "nuisance_estimator": "score matching",
        "methods": "SKSD-influence|Brueck-corrected",
        "comparison": "nominal rejection rates; no size adjustment",
        "timing_protocol": "standalone-test-v1",
        "timing_clock": "perf_counter",
        "timing_unit": "seconds per simulated dataset",
        "timing_shared_scope": (
            "score-matching geometry|observed nuisance fit|observed Stein Gram|"
            "Efron count generation"
        ),
        "timing_observed_setup_scope": "observed nuisance fit|fixed bandwidth",
        "timing_total_formula": "shared scope + method statistic/bootstrap/p-value",
        "timing_excludes": "observed-data simulation|checkpoint and output I/O",
        "timing_method_order": "SKSD-influence|Brueck-corrected",
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": _installed_version("scipy"),
        "plotting_backend": "R/ggplot2",
        "platform": platform.platform(),
        "cpu_architecture": platform.machine() or "unknown",
        "cpu_processor": platform.processor() or "unknown",
        "logical_cpu_count": os.cpu_count() or "unknown",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "unset"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get(
            "VECLIB_MAXIMUM_THREADS", "unset"
        ),
    }
    metadata.update(dict(_config_items(config)))
    return metadata


def _write_checkpoint(
    output_dir: Path,
    computational_rows: Sequence[dict],
    config,
    metadata: dict[str, object],
) -> tuple[Path, Path, Path]:
    raw_path = output_dir / "replications.csv"
    summary_path = output_dir / "summary.csv"
    metadata_path = output_dir / "run_metadata.csv"
    display_rows = _display_rows(computational_rows)
    summaries = _summary_rows(computational_rows, config)
    _atomic_csv(raw_path, CORE.RAW_FIELDS, display_rows)
    metadata["replications_sha256"] = _sha256(raw_path)
    if summaries:
        _atomic_csv(summary_path, tuple(summaries[0]), summaries)
        metadata["summary_sha256"] = _sha256(summary_path)
    _atomic_csv(metadata_path, ("key", "value"), _metadata_rows(metadata))
    return raw_path, summary_path, metadata_path


def _simulate_replication(alternative: str, delta: float, repetition: int, config, design) -> dict:
    model = design.model(alternative, delta)
    sample = CORE.gibbs_iid_qim(
        model,
        config.sample_size,
        burn=config.gibbs_burn,
        seed=CORE.seed_for(config, alternative, delta, repetition, stream=1),
    )
    result = CORE.paired_empirical_tests(
        sample,
        design,
        config,
        bootstrap_seed=CORE.seed_for(
            config, alternative, delta, repetition, stream=2
        ),
    )
    return {
        "alternative": alternative,
        "delta": delta,
        "rep": repetition,
        **asdict(result),
    }


def _plot(summary: Sequence[dict], output_dir: Path, alpha: float) -> tuple[Path, Path]:
    """Render the saved summary with the shared R/ggplot2 template."""
    first_alternative = tuple(CORE.ALTERNATIVES)[0]
    methods = (CORE.METHOD_INFLUENCE, CORE.METHOD_BRUECK_CORRECTED)
    timing_values = [
        float(row["mean_seconds"])
        for row in summary
        if row["alternative"] == first_alternative and row["method"] in methods
    ]
    if any(not math.isfinite(value) or value <= 0.0 for value in timing_values):
        raise ValueError("timing panels require finite, strictly positive runtimes")
    summary_path = output_dir / "summary.csv"
    plot_script = SCRIPT_PATH.parent.parent / "figure-code" / "plot_03_conditional_gaussian_score_matching.R"
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
    return pdf_path, png_path


def _prepare_run(args: argparse.Namespace, config) -> tuple[list[dict], dict[str, str], int]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / name for name in OUTPUT_NAMES]
    existing = [path for path in paths if path.exists()]
    if args.force:
        for path in paths:
            if path.exists():
                path.unlink()
        return [], {}, 0
    if not args.resume:
        if existing:
            raise FileExistsError(
                "output already exists; use --resume for a matching run or "
                "--force to replace it: " + ", ".join(str(path) for path in existing)
            )
        return [], {}, 0

    raw_path = output_dir / "replications.csv"
    metadata_path = output_dir / "run_metadata.csv"
    if not raw_path.exists() or not metadata_path.exists():
        if existing:
            raise FileNotFoundError(
                "resume requires both replications.csv and run_metadata.csv"
            )
        return [], {}, 1
    prior_metadata = _read_metadata(metadata_path)
    expected_fingerprint = config_fingerprint(config)
    if (
        prior_metadata.get("schema_version") != str(OUTPUT_SCHEMA_VERSION)
        or prior_metadata.get("config_fingerprint") != expected_fingerprint
        or prior_metadata.get("run_fingerprint") != expected_fingerprint
    ):
        raise ValueError(
            "resume configuration does not match the existing run: "
            f"{prior_metadata.get('config_fingerprint')} != {expected_fingerprint}"
        )
    recorded_raw_hash = prior_metadata.get("replications_sha256")
    if not recorded_raw_hash or recorded_raw_hash != _sha256(raw_path):
        raise ValueError("resume replications.csv checksum does not match metadata")
    summary_path = output_dir / "summary.csv"
    recorded_summary_hash = prior_metadata.get("summary_sha256")
    if summary_path.is_file() and (
        not recorded_summary_hash or recorded_summary_hash != _sha256(summary_path)
    ):
        raise ValueError("resume summary.csv checksum does not match metadata")
    rows = _load_resume_rows(raw_path, config)
    resume_count = int(prior_metadata.get("resume_count", "0")) + 1
    return rows, prior_metadata, resume_count


def _recorded_plot_config(args: argparse.Namespace, metadata: dict[str, str]):
    """Read settings and reject plotting-time changes to the saved design."""
    defaults = {**PUBLICATION_DEFAULTS, "score_matching_ridge": 0.0}
    values: dict[str, object] = {}
    try:
        for key, default in defaults.items():
            recorded = metadata[key]
            if isinstance(default, tuple):
                value = _parse_grid(recorded, key)
            elif isinstance(default, int):
                value = int(recorded)
            else:
                value = float(recorded)
                if not math.isfinite(value):
                    raise ValueError(f"non-finite {key}")
            values[key] = value
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("plot-only requires valid recorded simulation settings") from error
    values["output_dir"] = str(resolve_output_dir(args))
    config = CORE.Config(**values)
    CORE.validate_config(config)

    # User-specified plotting-time overrides must agree with the recorded
    # numerical run. Unspecified defaults (including --check) are irrelevant.
    option_fields = {
        "replications": "monte_carlo_reps", "burn": "gibbs_burn",
        "bandwidth": "fixed_bandwidth", "sm_lower": "score_matching_lower",
        "sm_upper": "score_matching_upper",
    }
    for key in defaults:
        option = next((name for name, field in option_fields.items() if field == key), key)
        supplied = getattr(args, option, None)
        if supplied is None:
            continue
        if isinstance(defaults[key], tuple):
            supplied = _parse_grid(supplied, option)
        if supplied != values[key]:
            raise ValueError(f"plot-only --{option.replace('_', '-')} conflicts with recorded {key}")

    return config


def _plot_config_from_metadata(args: argparse.Namespace, metadata: dict[str, str]):
    """Recover a fresh-run design and validate its recorded provenance."""
    config = _recorded_plot_config(args, metadata)

    required_provenance = (
        "design_version", "scientific_output_schema_version",
        "scientific_source_sha256", "entry_point_sha256",
    )
    if any(not metadata.get(key) for key in required_provenance):
        raise ValueError("plot-only metadata lacks source provenance")
    payload = "\n".join(f"{key}={value}" for key, value in _config_items(config))
    payload += f"\ndesign_version={metadata['design_version']}"
    payload += f"\nscientific_schema={metadata['scientific_output_schema_version']}"
    payload += f"\nscientific_source_sha256={metadata['scientific_source_sha256']}"
    payload += f"\nentry_point_sha256={metadata['entry_point_sha256']}"
    fingerprint = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    if (
        metadata.get("schema_version") != str(OUTPUT_SCHEMA_VERSION)
        or metadata.get("scientific_output_schema_version") != str(CORE.OUTPUT_SCHEMA_VERSION)
        or metadata.get("run_fingerprint") != fingerprint
        or metadata.get("config_fingerprint") != fingerprint
    ):
        raise ValueError("plot-only recorded configuration/provenance is inconsistent")
    return config


def _load_plot_summary(path: Path, expected: Sequence[dict]) -> list[dict]:
    """Validate and type the saved summary without rewriting it."""
    if not expected:
        raise ValueError("plot-only requires a nonempty summary")
    fields = tuple(expected[0])
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError("invalid summary.csv header")
        stored = list(reader)
    if len(stored) != len(expected):
        raise ValueError("summary.csv is incomplete or contains extra rows")
    summary: list[dict] = []
    for stored_row, expected_row in zip(stored, expected):
        typed: dict[str, object] = {}
        for field in fields:
            value = stored_row[field]
            target = expected_row[field]
            if isinstance(target, str):
                if value != target:
                    raise ValueError(f"summary.csv disagrees with replications.csv on {field}")
                typed[field] = value
            else:
                try:
                    numeric = float(value)
                except (TypeError, ValueError) as error:
                    raise ValueError(f"invalid summary.csv {field}") from error
                if not math.isfinite(numeric) or not math.isclose(numeric, float(target), rel_tol=1e-12, abs_tol=1e-14):
                    raise ValueError(f"summary.csv disagrees with replications.csv on {field}")
                typed[field] = numeric
        summary.append(typed)
    return summary


def validate_saved_results(
    output_dir: Path, args: argparse.Namespace | None = None,
) -> tuple[list[dict], object, dict[str, str]]:
    """Validate previously saved CSVs without rewriting data or running a model.

    ``source_metadata.csv`` is a flattened key/value copy of the original
    run's metadata; numeric grids use ``|`` separators. Its manifest records
    historical timing explicitly, never the fresh standalone timing protocol.
    """
    output_dir = Path(output_dir)
    manifest = _read_metadata(output_dir / "run_metadata.csv")
    expected_manifest = {
        "schema_version": "saved-csv-v1",
        "study": "03-conditional-gaussian-score-matching",
        "workflow": "saved-results",
        "run_complete": "true",
        "timing_protocol": "historical-recorded",
    }
    if any(manifest.get(key) != value for key, value in expected_manifest.items()):
        raise ValueError("invalid saved conditional-Gaussian CSV manifest")
    for filename, key in (
        ("replications.csv", "replications_sha256"),
        ("summary.csv", "summary_sha256"),
        ("source_metadata.csv", "source_metadata_sha256"),
    ):
        path = output_dir / filename
        if not path.is_file() or manifest.get(key) != _sha256(path):
            raise ValueError(f"saved {filename} checksum does not match manifest")
    source = _read_metadata(output_dir / "source_metadata.csv")
    if (
        source.get("design_version") != CORE.DESIGN_VERSION
        or source.get("output_schema_version") != str(CORE.OUTPUT_SCHEMA_VERSION)
        or source.get("brueck_production_correction") != "factor-corrected-midpoint-v1"
        or source.get("run_complete") != "true"
    ):
        raise ValueError("saved conditional-Gaussian source is not a completed corrected run")
    settings = dict(source)
    for key in ("rewire_grid", "community_grid", "matching_grid"):
        settings[key] = settings.get(key, "").replace("|", ",")
    if args is None:
        args = build_parser().parse_args(["--output-dir", str(output_dir)])
    config = _recorded_plot_config(args, settings)
    raw_path = output_dir / "replications.csv"
    computational_rows = _load_resume_rows(raw_path, config)
    expected_computational = len(CORE.computational_cells(config)) * config.monte_carlo_reps
    expected_display = sum(len(CORE.grid_for(config, alternative)) for alternative in CORE.ALTERNATIVES) * config.monte_carlo_reps
    with raw_path.open(newline="", encoding="utf-8") as handle:
        display_count = sum(1 for _row in csv.DictReader(handle))
    if (
        len(computational_rows) != expected_computational
        or display_count != expected_display
        or source.get("completed_raw_rows") != str(expected_display)
        or source.get("expected_raw_rows") != str(expected_display)
    ):
        raise ValueError("saved conditional-Gaussian CSVs do not cover the complete recorded design")
    summary = _load_plot_summary(
        output_dir / "summary.csv", _summary_rows(computational_rows, config)
    )
    return summary, config, manifest


def _run_plot_only(args: argparse.Namespace) -> dict[str, Path]:
    if args.resume:
        raise ValueError("--plot-only cannot be combined with --resume")
    output_dir = resolve_output_dir(args)
    raw_path = output_dir / "replications.csv"
    summary_path = output_dir / "summary.csv"
    metadata_path = output_dir / "run_metadata.csv"
    missing = [
        path for path in (raw_path, summary_path, metadata_path) if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "plot-only requires completed replications, summary, and metadata: "
            + ", ".join(str(path) for path in missing)
        )
    metadata = _read_metadata(metadata_path)
    if metadata.get("schema_version") == "saved-csv-v1":
        summary, config, metadata = validate_saved_results(output_dir, args)
        pdf_path, png_path = _plot(summary, output_dir, config.alpha)
        metadata.update({
            "updated_utc": _utc_now(),
            "figure_pdf_sha256": _sha256(pdf_path),
            "figure_png_sha256": _sha256(png_path),
        })
        _atomic_csv(metadata_path, ("key", "value"), _metadata_rows(metadata))
        print(f"Validated saved CSVs in {output_dir}; numerical data unchanged")
        print(f"Saved {pdf_path}")
        print(f"Saved {png_path}")
        return {
            "replications": raw_path, "summary": summary_path,
            "metadata": metadata_path, "figure_pdf": pdf_path, "figure_png": png_path,
        }
    config = _plot_config_from_metadata(args, metadata)
    if metadata.get("run_complete") != "true":
        raise ValueError("plot-only requires run_complete=true")
    recorded_raw_hash = metadata.get("replications_sha256")
    if not recorded_raw_hash or recorded_raw_hash != _sha256(raw_path):
        raise ValueError("replications.csv checksum does not match run_metadata.csv")
    if metadata.get("summary_sha256") != _sha256(summary_path):
        raise ValueError("summary.csv checksum does not match run_metadata.csv")
    computational_rows = _load_resume_rows(raw_path, config)
    expected_rows = (
        len(CORE.computational_cells(config)) * config.monte_carlo_reps
    )
    if len(computational_rows) != expected_rows:
        raise ValueError(
            "plot-only requires every configured replication: "
            f"{len(computational_rows)} != {expected_rows}"
        )
    expected_display = sum(len(CORE.grid_for(config, alternative)) for alternative in CORE.ALTERNATIVES) * config.monte_carlo_reps
    with raw_path.open(newline="", encoding="utf-8") as handle:
        display_count = sum(1 for _row in csv.DictReader(handle))
    expected_counts = {
        "expected_computational_rows": expected_rows,
        "completed_computational_rows": expected_rows,
        "expected_display_rows": expected_display,
        "completed_display_rows": expected_display,
    }
    if display_count != expected_display or any(metadata.get(key) != str(value) for key, value in expected_counts.items()):
        raise ValueError("plot-only requires complete saved computational and display rows")
    summary = _load_plot_summary(summary_path, _summary_rows(computational_rows, config))
    pdf_path, png_path = _plot(summary, output_dir, config.alpha)
    metadata.update(
        {
            "updated_utc": _utc_now(),
            "figure_pdf_sha256": _sha256(pdf_path),
            "figure_png_sha256": _sha256(png_path),
        }
    )
    _atomic_csv(metadata_path, ("key", "value"), _metadata_rows(metadata))
    print(f"Validated {raw_path}")
    print(f"Validated {summary_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    return {
        "replications": raw_path,
        "summary": summary_path,
        "metadata": metadata_path,
        "figure_pdf": pdf_path,
        "figure_png": png_path,
    }


def run(args: argparse.Namespace) -> dict[str, Path]:
    if args.plot_only:
        return _run_plot_only(args)
    config, design = resolve_config(args)
    profile = "quick" if args.quick else "publication"
    output_dir = Path(config.output_dir)
    computational_rows, prior_metadata, resume_count = _prepare_run(args, config)
    started_utc = prior_metadata.get("started_utc", _utc_now())
    elapsed_before = float(prior_metadata.get("elapsed_seconds", "0"))
    current_started = time.perf_counter()

    completed = {
        (row["alternative"], float(row["delta"]), int(row["rep"]))
        for row in computational_rows
    }
    cells = CORE.computational_cells(config)
    total_rows = len(cells) * config.monte_carlo_reps
    print(
        "Conditional-Gaussian score-matching simulation: "
        f"d={config.dimension}, n={config.sample_size}, "
        f"B={config.bootstrap_draws}, R={config.monte_carlo_reps}, "
        f"burn={config.gibbs_burn}, h={config.fixed_bandwidth:g}",
        flush=True,
    )
    if completed:
        print(f"Resuming {len(completed)}/{total_rows} completed replications.", flush=True)

    for alternative, delta in cells:
        for repetition in range(config.monte_carlo_reps):
            key = (alternative, float(delta), repetition)
            if key in completed:
                continue
            row = _simulate_replication(
                alternative, float(delta), repetition, config, design
            )
            computational_rows.append(row)
            completed.add(key)
            elapsed = elapsed_before + time.perf_counter() - current_started
            metadata = _base_metadata(
                config,
                profile,
                started_utc,
                elapsed,
                resume_count,
                computational_rows,
                run_complete=False,
            )
            _write_checkpoint(output_dir, computational_rows, config, metadata)
            print(
                f"[{len(completed):>5}/{total_rows}] "
                f"{alternative} delta={delta:g} rep={repetition}",
                flush=True,
            )

    computational_rows = _sort_rows(computational_rows)
    summary = _summary_rows(computational_rows, config)
    elapsed = elapsed_before + time.perf_counter() - current_started
    metadata = _base_metadata(
        config,
        profile,
        started_utc,
        elapsed,
        resume_count,
        computational_rows,
        run_complete=True,
    )
    raw_path, summary_path, metadata_path = _write_checkpoint(
        output_dir, computational_rows, config, metadata
    )
    pdf_path, png_path = _plot(summary, output_dir, config.alpha)
    metadata.update(
        {
            "finished_utc": _utc_now(),
            "replications_sha256": _sha256(raw_path),
            "summary_sha256": _sha256(summary_path),
            "figure_pdf_sha256": _sha256(pdf_path),
            "figure_png_sha256": _sha256(png_path),
        }
    )
    _atomic_csv(metadata_path, ("key", "value"), _metadata_rows(metadata))
    print(f"Saved {raw_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {metadata_path}")
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")
    return {
        "replications": raw_path,
        "summary": summary_path,
        "metadata": metadata_path,
        "figure_pdf": pdf_path,
        "figure_png": png_path,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except (FileExistsError, FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
