"""Fresh-run orchestration for the KEF/minimum-KSD reproduction."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
from typing import Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "semiparametric_ksd_kef_minimum_ksd_fresh_v1"
FRESH_ARM = "fresh_paired_observed_datasets"
TIMING_REPLICATION_BASIS = (
    "same statistical Monte Carlo replications as rejection-rate results"
)
CHECKPOINT_DIRECTORY_NAME = "checkpoints"
PUBLICATION_CHECKPOINT_BATCH = 20
OWNED_OUTPUT_NAMES = (
    CHECKPOINT_DIRECTORY_NAME,
    "replications.csv",
    "summary.csv",
    "run_metadata.csv",
    "figure.pdf",
    "figure.png",
)
THREAD_ENVIRONMENT_KEYS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
HELPER_PATH = Path(__file__).resolve()
CORE_PATH = HELPER_PATH.with_name("_kef_core.py")


class FreshSimulationError(ValueError):
    """Raised when a fresh-run configuration or checkpoint is inconsistent."""


def _load_core():
    if not CORE_PATH.is_file():
        raise FileNotFoundError(f"KEF scientific core is missing: {CORE_PATH}")
    module_name = "_semiksd_kef_scientific_core"
    existing = sys.modules.get(module_name)
    if existing is not None and Path(existing.__file__).resolve() == CORE_PATH:
        return existing
    specification = importlib.util.spec_from_file_location(module_name, CORE_PATH)
    if specification is None or specification.loader is None:
        raise ImportError(f"could not load KEF scientific core: {CORE_PATH}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _float_text(value: float) -> str:
    return format(float(value), ".17g")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime_environment() -> dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "thread_environment": {
            key: os.environ.get(key, "unset")
            for key in THREAD_ENVIRONMENT_KEYS
        },
    }


def _fingerprint(items: Mapping[str, object]) -> str:
    payload = json.dumps(items, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _method_specs(api) -> tuple[Mapping[str, object], ...]:
    specs = tuple(api.PAIRED_METHODS)
    methods = [str(spec["method"]) for spec in specs]
    if len(specs) != 6 or len(set(methods)) != 6:
        raise FreshSimulationError(
            "the KEF API must expose exactly six distinct paired methods"
        )
    return specs


def _raw_columns(api) -> tuple[str, ...]:
    return tuple(api.PAIRED_REPLICATION_COLUMNS) + ("seconds",)


def _dimensions(config) -> tuple[int, ...]:
    try:
        dimensions = tuple(config.effective_dimensions)
    except (AttributeError, TypeError) as error:
        raise FreshSimulationError(
            "fresh KEF configuration has invalid effective dimensions"
        ) from error
    if (
        not dimensions
        or any(isinstance(value, bool) or not isinstance(value, int) for value in dimensions)
        or any(value <= 0 for value in dimensions)
        or len(set(dimensions)) != len(dimensions)
    ):
        raise FreshSimulationError(
            "fresh KEF dimensions must be distinct positive integers"
        )
    return dimensions


def _base_grid(config) -> tuple[float, ...]:
    try:
        grid = tuple(float(value) for value in config.base_grid)
    except (AttributeError, TypeError, ValueError) as error:
        raise FreshSimulationError("fresh KEF base grid is invalid") from error
    if (
        not grid
        or any(not math.isfinite(value) for value in grid)
        or any(left >= right for left, right in zip(grid, grid[1:]))
    ):
        raise FreshSimulationError(
            "fresh KEF base grid must be finite and strictly increasing"
        )
    return grid


def _validate_config(config, api) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if getattr(config, "workflow", None) != "simulate":
        raise FreshSimulationError("fresh KEF engine requires workflow='simulate'")
    dimensions = _dimensions(config)
    base_grid = _base_grid(config)
    integer_fields = {
        "sample size": config.n,
        "bootstrap draws": config.bootstrap_draws,
        "replications": config.replications,
        "seed": config.fresh_seed,
    }
    for label, value in integer_fields.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise FreshSimulationError(f"fresh KEF {label} must be an integer")
    if config.n < 2:
        raise FreshSimulationError("fresh KEF sample size must be at least 2")
    if config.bootstrap_draws < 20:
        raise FreshSimulationError(
            "fresh KEF bootstrap draws must be at least 20 at alpha=0.05"
        )
    if config.replications < 1:
        raise FreshSimulationError("fresh KEF replications must be positive")
    if config.fresh_seed < 0:
        raise FreshSimulationError("fresh KEF seed must be nonnegative")
    if not math.isfinite(config.theta1):
        raise FreshSimulationError("fresh KEF theta1 must be finite")
    if not math.isfinite(config.sigma) or config.sigma <= 0.0:
        raise FreshSimulationError("fresh KEF sigma must be finite and positive")
    if not math.isfinite(config.alpha) or not 0.0 < config.alpha < 1.0:
        raise FreshSimulationError("fresh KEF alpha must lie between zero and one")
    if getattr(config, "resume", False) and getattr(config, "force", False):
        raise FreshSimulationError("fresh KEF resume and force are mutually exclusive")
    _method_specs(api)
    return dimensions, base_grid


def _mode(config) -> str:
    if config.quick:
        return "quick-fresh-simulation"
    if config.publication_design:
        return "full-fresh-simulation"
    return "custom-fresh-simulation"


def _fingerprint_metadata(config, api, core, dimensions, base_grid):
    runtime = _runtime_environment()
    core_hash = _sha256(CORE_PATH)
    helper_hash = _sha256(HELPER_PATH)
    driver_path = Path(api.__file__).resolve()
    driver_hash = _sha256(driver_path)
    items = {
        "schema_version": SCHEMA_VERSION,
        "workflow": "simulate",
        "mode": _mode(config),
        "dimensions": list(dimensions),
        "base_grid": list(base_grid),
        "sample_size": config.n,
        "bootstrap_draws": config.bootstrap_draws,
        "monte_carlo_replications": config.replications,
        "theta1": _float_text(config.theta1),
        "sigma": _float_text(config.sigma),
        "alpha": _float_text(config.alpha),
        "seed": config.fresh_seed,
        "scientific_core_sha256": core_hash,
        "simulation_helper_sha256": helper_hash,
        "entry_point_sha256": driver_hash,
        "runtime_environment": runtime,
        "timing_protocol": str(core.TIMING_PROTOCOL),
        "timing_clock": str(core.TIMING_CLOCK),
    }
    return _fingerprint(items), items


def _base_metadata(config, api, core, dimensions, base_grid):
    fingerprint, items = _fingerprint_metadata(
        config, api, core, dimensions, base_grid
    )
    runtime_json = json.dumps(
        items["runtime_environment"], sort_keys=True, separators=(",", ":")
    )
    dataset_count = len(dimensions) * len(base_grid) * config.replications
    method_count = len(_method_specs(api))
    return {
        "simulation": getattr(api, "SIMULATION_ID", "02_kef_minimum_ksd"),
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "workflow": "simulate",
        "mode": items["mode"],
        "quick": str(config.quick).lower(),
        "publication_design": str(config.publication_design).lower(),
        "run_fingerprint": fingerprint,
        "configuration_fingerprint": fingerprint,
        "generated_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "output_dir": config.output_dir,
        "dimensions": ",".join(str(value) for value in dimensions),
        "base_grid": ",".join(_float_text(value) for value in base_grid),
        "sample_size": config.n,
        "bootstrap_draws": config.bootstrap_draws,
        "monte_carlo_replications": config.replications,
        "theta1": _float_text(config.theta1),
        "sigma": _float_text(config.sigma),
        "alpha": _float_text(config.alpha),
        "seed": config.fresh_seed,
        "fresh_seed": config.fresh_seed,
        "seed_scheme": (
            "SeedSequence([seed,dimension,signal_index,replication,stream])"
        ),
        "nuisance_estimator": "minimum KSD",
        "bootstrap_mode": "refit",
        "empirical_bootstrap": "gaussian-multiplier (influence, naive); efron (Brueck)",
        "brueck_correction": "factor-corrected midpoint diagnostic",
        "method_count": method_count,
        "expected_datasets": dataset_count,
        "completed_datasets": 0,
        "expected_replication_rows": dataset_count * method_count,
        "completed_replication_rows": 0,
        "replications_csv_rows": 0,
        "summary_csv_rows": 0,
        "paired_observed_datasets": "true",
        "shared_bootstrap_randomness_across_all_methods": "false",
        "paired_contrasts_reconstructible": "true",
        "replication_level_pvalues_available": "true",
        "replication_level_decisions_available": "true",
        "replications_csv_is_header_only": "false",
        "monte_carlo_arm": FRESH_ARM,
        "timing_protocol": items["timing_protocol"],
        "timing_clock": items["timing_clock"],
        "timing_replication_basis": TIMING_REPLICATION_BASIS,
        "timing_monte_carlo_replications": config.replications,
        "timing_scope": (
            "observed bandwidth and minimum-KSD fit plus each method's test "
            "and calibration; observed-data generation excluded"
        ),
        "timing_method_order": ",".join(
            str(spec["method"]) for spec in _method_specs(api)
        ),
        "benchmark_environment": runtime_json,
        "runtime_environment_sha256": hashlib.sha256(
            runtime_json.encode("utf-8")
        ).hexdigest(),
        "scientific_core_file": CORE_PATH.name,
        "scientific_core_sha256": items["scientific_core_sha256"],
        "simulation_helper_file": HELPER_PATH.name,
        "simulation_helper_sha256": items["simulation_helper_sha256"],
        "entry_point_file": Path(api.__file__).name,
        "entry_point_sha256": items["entry_point_sha256"],
        "historical_scientific_function_count": getattr(
            core, "HISTORICAL_SCIENTIFIC_FUNCTION_COUNT", "unavailable"
        ),
        "historical_scientific_ast_sha256": getattr(
            core, "HISTORICAL_SCIENTIFIC_AST_SHA256", "unavailable"
        ),
        "checkpoint_directory": CHECKPOINT_DIRECTORY_NAME,
        "checkpoint_strategy": "per-cell-batched-atomic-snapshot-v1",
        "checkpoint_index": "{}",
        "checkpoint_files": 0,
        "checkpoint_batch_datasets": (
            1 if config.quick else PUBLICATION_CHECKPOINT_BATCH
        ),
        "maximum_uncheckpointed_datasets": (
            0 if config.quick else PUBLICATION_CHECKPOINT_BATCH - 1
        ),
    }


def _read_rows(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise FreshSimulationError(
                f"{path.name} has an incompatible column schema"
            )
        return list(reader)


def _as_int(value, label: str) -> int:
    try:
        numeric = float(value)
        integer = int(numeric)
    except (TypeError, ValueError, OverflowError) as error:
        raise FreshSimulationError(f"{label} must be an integer") from error
    if not math.isfinite(numeric) or numeric != integer:
        raise FreshSimulationError(f"{label} must be an integer")
    return integer


def _as_float(value, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise FreshSimulationError(f"{label} must be numerical") from error
    if not math.isfinite(result):
        raise FreshSimulationError(f"{label} must be finite")
    return result


def _as_bool(value, label: str) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    text = str(value).strip().lower()
    if text == "true":
        return True
    if text == "false":
        return False
    raise FreshSimulationError(f"{label} must be true or false")


def _metadata_dimensions(metadata: Mapping[str, object]) -> tuple[int, ...]:
    try:
        dimensions = tuple(
            int(value) for value in str(metadata["dimensions"]).split(",")
        )
    except (KeyError, ValueError) as error:
        raise FreshSimulationError("metadata dimensions are invalid") from error
    if not dimensions or any(value <= 0 for value in dimensions):
        raise FreshSimulationError("metadata dimensions are invalid")
    return dimensions


def _metadata_grid(metadata: Mapping[str, object]) -> tuple[float, ...]:
    try:
        grid = tuple(
            float(value) for value in str(metadata["base_grid"]).split(",")
        )
    except (KeyError, ValueError) as error:
        raise FreshSimulationError("metadata base_grid is invalid") from error
    if (
        not grid
        or any(not math.isfinite(value) for value in grid)
        or any(left >= right for left, right in zip(grid, grid[1:]))
    ):
        raise FreshSimulationError("metadata base_grid is invalid")
    return grid


def _dataset_keys(
    dimensions: Sequence[int], base_grid: Sequence[float], replications: int
) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (dimension, signal_index, replication)
        for dimension in dimensions
        for signal_index in range(len(base_grid))
        for replication in range(1, replications + 1)
    )


def _validate_core_result(result, specs) -> None:
    expected = tuple(str(spec["method"]) for spec in specs)
    if not isinstance(result, Mapping) or tuple(result) != expected:
        raise FreshSimulationError(
            "KEF scientific core did not return the five methods in canonical order"
        )
    for method in expected:
        record = result[method]
        if not isinstance(record, Mapping) or set(record) != {"p_value", "seconds"}:
            raise FreshSimulationError(
                f"KEF scientific core returned malformed output for {method}"
            )
        p_value = _as_float(record["p_value"], f"{method} p_value")
        seconds = _as_float(record["seconds"], f"{method} seconds")
        if not 0.0 <= p_value <= 1.0 or seconds < 0.0:
            raise FreshSimulationError(
                f"KEF scientific core returned invalid output for {method}"
            )


def _dataset_rows(
    config,
    api,
    specs,
    dimension: int,
    signal_index: int,
    base_signal: float,
    replication_zero_based: int,
    result,
) -> list[dict[str, object]]:
    _validate_core_result(result, specs)
    signal = base_signal / math.sqrt(dimension)
    simulation = getattr(api, "SIMULATION_ID", "02_kef_minimum_ksd")
    rows: list[dict[str, object]] = []
    for spec in specs:
        method = str(spec["method"])
        record = result[method]
        p_value = float(record["p_value"])
        rows.append({
            "simulation": simulation,
            "dimension": dimension,
            "theta_index": signal_index,
            "theta2": signal,
            "theta2_base": base_signal,
            "replication": replication_zero_based + 1,
            "method": method,
            "display_method": spec["display_method"],
            "test": spec["test"],
            "calibration": spec["calibration"],
            "nuisance_estimator": "minimum KSD",
            "p_value": p_value,
            "reject": p_value < config.alpha,
            "alpha": config.alpha,
            "source_archive": "",
            "seconds": float(record["seconds"]),
        })
    return rows


def _wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    if trials <= 0 or successes < 0 or successes > trials:
        raise FreshSimulationError("invalid binomial count for Wilson interval")
    z = 1.959963984540054
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(
        probability * (1.0 - probability) / trials
        + z * z / (4.0 * trials * trials)
    ) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _summarize(rows: Sequence[Mapping[str, object]], api) -> list[dict[str, object]]:
    specs = _method_specs(api)
    methods = tuple(str(spec["method"]) for spec in specs)
    spec_by_method = {str(spec["method"]): spec for spec in specs}
    grouped: dict[tuple[int, int, str], list[Mapping[str, object]]] = {}
    for row in rows:
        key = (
            _as_int(row["dimension"], "dimension"),
            _as_int(row["theta_index"], "theta_index"),
            str(row["method"]),
        )
        grouped.setdefault(key, []).append(row)

    summaries: list[dict[str, object]] = []
    cells = sorted({(dimension, index) for dimension, index, _ in grouped})
    for dimension, signal_index in cells:
        for method in methods:
            group = grouped.get((dimension, signal_index, method), [])
            if not group:
                raise FreshSimulationError(
                    "cannot summarize an incomplete KEF method/cell"
                )
            group = sorted(
                group,
                key=lambda row: _as_int(row["replication"], "replication"),
            )
            repetitions = len(group)
            rejected = sum(
                _as_bool(row["reject"], "reject") for row in group
            )
            rate = rejected / repetitions
            lower, upper = _wilson_interval(rejected, repetitions)
            spec = spec_by_method[method]
            first = group[0]
            summaries.append({
                "simulation": first["simulation"],
                "dimension": dimension,
                "theta_index": signal_index,
                "theta2": _as_float(first["theta2"], "theta2"),
                "theta2_base": _as_float(first["theta2_base"], "theta2_base"),
                "method": method,
                "display_method": spec["display_method"],
                "test": spec["test"],
                "calibration": spec["calibration"],
                "nuisance_estimator": "minimum KSD",
                "rejection_rate": rate,
                "rejected": rejected,
                "monte_carlo_repetitions": repetitions,
                "mcse": math.sqrt(rate * (1.0 - rate) / repetitions),
                "ci_lower": lower,
                "ci_upper": upper,
                "mean_seconds_per_replication": float(np.mean([
                    _as_float(row["seconds"], "seconds") for row in group
                ])),
                "monte_carlo_arm": FRESH_ARM,
                "paired_observed_datasets": True,
                "shared_bootstrap_randomness_across_all_methods": False,
                "paired_contrasts_reconstructible": True,
                "replications_available": True,
                "source_archive": "",
            })
    return summaries


def _validate_digest(value, label: str) -> None:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise FreshSimulationError(f"metadata {label} is not a SHA-256 digest")


def _validate_raw_rows(
    rows: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object],
    api,
    *,
    allow_prefix: bool,
) -> int:
    specs = _method_specs(api)
    methods = tuple(str(spec["method"]) for spec in specs)
    spec_by_method = {str(spec["method"]): spec for spec in specs}
    columns = set(_raw_columns(api))
    dimensions = _metadata_dimensions(metadata)
    base_grid = _metadata_grid(metadata)
    replications = _as_int(
        metadata.get("monte_carlo_replications"),
        "metadata monte_carlo_replications",
    )
    alpha = _as_float(metadata.get("alpha"), "metadata alpha")
    expected_datasets = _dataset_keys(dimensions, base_grid, replications)
    expected_identifiers = tuple(
        (dimension, signal_index, replication, method)
        for dimension, signal_index, replication in expected_datasets
        for method in methods
    )
    if len(rows) % len(methods):
        raise FreshSimulationError(
            "KEF checkpoint contains a partial five-method dataset"
        )
    observed_identifiers: list[tuple[int, int, int, str]] = []
    simulation = str(metadata.get("simulation", ""))
    for row_number, row in enumerate(rows, start=1):
        if set(row) != columns:
            raise FreshSimulationError(
                f"KEF replication row {row_number} has an incompatible schema"
            )
        dimension = _as_int(row["dimension"], "dimension")
        signal_index = _as_int(row["theta_index"], "theta_index")
        replication = _as_int(row["replication"], "replication")
        method = str(row["method"])
        identifier = (dimension, signal_index, replication, method)
        observed_identifiers.append(identifier)
        if method not in spec_by_method:
            raise FreshSimulationError(f"unknown KEF method in row: {method}")
        spec = spec_by_method[method]
        if (
            str(row["simulation"]) != simulation
            or str(row["display_method"]) != str(spec["display_method"])
            or str(row["test"]) != str(spec["test"])
            or str(row["calibration"]) != str(spec["calibration"])
            or str(row["nuisance_estimator"]) != "minimum KSD"
            or str(row["source_archive"]) != ""
        ):
            raise FreshSimulationError(
                f"KEF replication row {row_number} has inconsistent method metadata"
            )
        if dimension not in dimensions or not 0 <= signal_index < len(base_grid):
            raise FreshSimulationError(
                f"KEF replication row {row_number} has an unknown simulation cell"
            )
        expected_base = base_grid[signal_index]
        expected_signal = expected_base / math.sqrt(dimension)
        if not math.isclose(
            _as_float(row["theta2_base"], "theta2_base"),
            expected_base,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) or not math.isclose(
            _as_float(row["theta2"], "theta2"),
            expected_signal,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise FreshSimulationError(
                f"KEF replication row {row_number} has the wrong signal"
            )
        p_value = _as_float(row["p_value"], "p_value")
        seconds = _as_float(row["seconds"], "seconds")
        row_alpha = _as_float(row["alpha"], "alpha")
        if (
            not 0.0 <= p_value <= 1.0
            or seconds < 0.0
            or not math.isclose(row_alpha, alpha, rel_tol=0.0, abs_tol=1e-15)
            or _as_bool(row["reject"], "reject") != (p_value < alpha)
        ):
            raise FreshSimulationError(
                f"KEF replication row {row_number} has invalid numerical results"
            )
    expected_slice = expected_identifiers[: len(observed_identifiers)]
    if tuple(observed_identifiers) != expected_slice:
        kind = "checkpoint prefix" if allow_prefix else "complete result grid"
        raise FreshSimulationError(
            f"KEF replication rows do not form the canonical {kind}"
        )
    if not allow_prefix and len(observed_identifiers) != len(expected_identifiers):
        raise FreshSimulationError(
            "KEF replication rows do not cover the complete result grid"
        )
    return len(rows) // len(methods)


def _assert_summary_matches(
    observed: Sequence[Mapping[str, object]],
    expected: Sequence[Mapping[str, object]],
    api,
) -> None:
    if len(observed) != len(expected):
        raise FreshSimulationError("KEF summary row count is inconsistent")
    columns = set(api.PAIRED_SUMMARY_COLUMNS)
    integer_fields = {
        "dimension", "theta_index", "rejected", "monte_carlo_repetitions",
    }
    float_fields = {
        "theta2",
        "theta2_base",
        "rejection_rate",
        "mcse",
        "ci_lower",
        "ci_upper",
        "mean_seconds_per_replication",
    }
    boolean_fields = {
        "paired_observed_datasets",
        "shared_bootstrap_randomness_across_all_methods",
        "paired_contrasts_reconstructible",
        "replications_available",
    }
    for row_number, (found, wanted) in enumerate(
        zip(observed, expected), start=1
    ):
        if set(found) != columns:
            raise FreshSimulationError(
                f"KEF summary row {row_number} has an incompatible schema"
            )
        for field in api.PAIRED_SUMMARY_COLUMNS:
            if field in integer_fields:
                matches = _as_int(found[field], field) == int(wanted[field])
            elif field in float_fields:
                matches = math.isclose(
                    _as_float(found[field], field),
                    float(wanted[field]),
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
            elif field in boolean_fields:
                matches = _as_bool(found[field], field) is bool(wanted[field])
            else:
                matches = str(found[field]) == str(wanted[field])
            if not matches:
                raise FreshSimulationError(
                    f"KEF summary row {row_number} disagrees in {field}"
                )


def validate_tables(
    metadata: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    summary: Sequence[Mapping[str, object]],
    api,
) -> None:
    """Validate complete fresh-run tables and their statistical semantics."""
    required_values = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "workflow": "simulate",
        "paired_observed_datasets": "true",
        "shared_bootstrap_randomness_across_all_methods": "false",
        "paired_contrasts_reconstructible": "true",
        "replication_level_pvalues_available": "true",
        "replication_level_decisions_available": "true",
        "replications_csv_is_header_only": "false",
        "monte_carlo_arm": FRESH_ARM,
        "timing_replication_basis": TIMING_REPLICATION_BASIS,
    }
    for key, expected in required_values.items():
        if str(metadata.get(key, "")).lower() != expected.lower():
            raise FreshSimulationError(
                f"fresh KEF metadata must record {key}={expected}"
            )
    mode = str(metadata.get("mode", ""))
    if mode not in {
        "quick-fresh-simulation",
        "full-fresh-simulation",
        "custom-fresh-simulation",
    }:
        raise FreshSimulationError("fresh KEF metadata has an invalid mode")
    publication = _as_bool(
        metadata.get("publication_design"), "publication_design"
    )
    quick = _as_bool(metadata.get("quick"), "quick")
    if (
        publication != (mode == "full-fresh-simulation")
        or quick != (mode == "quick-fresh-simulation")
    ):
        raise FreshSimulationError(
            "fresh KEF publication/quick design and mode are inconsistent"
        )
    dimensions = _metadata_dimensions(metadata)
    base_grid = _metadata_grid(metadata)
    replications = _as_int(
        metadata.get("monte_carlo_replications"),
        "metadata monte_carlo_replications",
    )
    sample_size = _as_int(metadata.get("sample_size"), "metadata sample_size")
    bootstrap_draws = _as_int(
        metadata.get("bootstrap_draws"), "metadata bootstrap_draws"
    )
    seed = _as_int(metadata.get("seed"), "metadata seed")
    theta1 = _as_float(metadata.get("theta1"), "metadata theta1")
    sigma = _as_float(metadata.get("sigma"), "metadata sigma")
    alpha = _as_float(metadata.get("alpha"), "metadata alpha")
    if (
        sample_size < 2
        or bootstrap_draws < 20
        or replications < 1
        or seed < 0
        or sigma <= 0.0
        or not 0.0 < alpha < 1.0
    ):
        raise FreshSimulationError("fresh KEF metadata has an invalid design")
    actual_publication_design = (
        dimensions == tuple(api.PUBLICATION_DIMENSIONS)
        and len(base_grid) == len(api.PUBLICATION_BASE_GRID)
        and np.allclose(
            base_grid, api.PUBLICATION_BASE_GRID, rtol=0.0, atol=1e-12
        )
        and sample_size == api.PUBLICATION_N
        and bootstrap_draws == api.PUBLICATION_B
        and replications == api.PUBLICATION_REPLICATIONS
        and math.isclose(theta1, api.PUBLICATION_THETA1, abs_tol=1e-12)
        and math.isclose(sigma, api.PUBLICATION_SIGMA, abs_tol=1e-12)
        and math.isclose(alpha, api.PUBLICATION_ALPHA, abs_tol=1e-15)
        and seed == api.PUBLICATION_FRESH_SEED
        and not quick
    )
    if publication != actual_publication_design:
        raise FreshSimulationError(
            "fresh KEF publication_design does not match the actual design"
        )
    expected_datasets = len(dimensions) * len(base_grid) * replications
    expected_rows = expected_datasets * len(_method_specs(api))
    expected_summary_rows = len(dimensions) * len(base_grid) * len(
        _method_specs(api)
    )
    integer_expectations = {
        "expected_datasets": expected_datasets,
        "completed_datasets": expected_datasets,
        "expected_replication_rows": expected_rows,
        "completed_replication_rows": expected_rows,
        "replications_csv_rows": expected_rows,
        "summary_csv_rows": expected_summary_rows,
        "timing_monte_carlo_replications": replications,
        "method_count": len(_method_specs(api)),
    }
    for key, expected in integer_expectations.items():
        if _as_int(metadata.get(key), f"metadata {key}") != expected:
            raise FreshSimulationError(
                f"fresh KEF metadata has the wrong {key}"
            )
    if (
        metadata.get("timing_protocol") != "standalone-test-v1"
        or metadata.get("timing_clock") != "perf_counter"
    ):
        raise FreshSimulationError(
            "fresh KEF metadata has the wrong timing protocol/clock"
        )
    expected_method_order = ",".join(
        str(spec["method"]) for spec in _method_specs(api)
    )
    if metadata.get("timing_method_order") != expected_method_order:
        raise FreshSimulationError(
            "fresh KEF metadata has the wrong timing method order"
        )
    try:
        environment = json.loads(str(metadata["benchmark_environment"]))
    except (KeyError, json.JSONDecodeError) as error:
        raise FreshSimulationError(
            "fresh KEF benchmark_environment is not valid JSON"
        ) from error
    if not isinstance(environment, dict):
        raise FreshSimulationError(
            "fresh KEF benchmark_environment must be a JSON object"
        )
    environment_digest = hashlib.sha256(
        str(metadata["benchmark_environment"]).encode("utf-8")
    ).hexdigest()
    if environment_digest != metadata.get("runtime_environment_sha256"):
        raise FreshSimulationError(
            "fresh KEF benchmark environment hash is inconsistent"
        )
    for key in (
        "run_fingerprint",
        "configuration_fingerprint",
        "scientific_core_sha256",
        "simulation_helper_sha256",
        "entry_point_sha256",
        "runtime_environment_sha256",
        "replications_sha256",
        "summary_sha256",
    ):
        _validate_digest(metadata.get(key, ""), key)
    if metadata["run_fingerprint"] != metadata["configuration_fingerprint"]:
        raise FreshSimulationError("fresh KEF fingerprints disagree")
    completed = _validate_raw_rows(
        rows, metadata, api, allow_prefix=False
    )
    if completed != expected_datasets:
        raise FreshSimulationError("fresh KEF completed dataset count is wrong")
    expected_summary = _summarize(rows, api)
    _assert_summary_matches(summary, expected_summary, api)


validate_fresh_standardized_tables = validate_tables


CHECKPOINT_PATTERN = re.compile(
    r"d(?P<dimension>[1-9][0-9]*)_theta(?P<signal>[0-9]{3})_"
    r"rep(?P<start>[0-9]{6})-(?P<end>[0-9]{6})[.]csv"
)


def _checkpoint_columns(api) -> tuple[str, ...]:
    return _raw_columns(api) + ("run_fingerprint",)


def _checkpoint_name(
    dimension: int, signal_index: int, start: int, end: int
) -> str:
    return (
        f"d{dimension}_theta{signal_index:03d}_"
        f"rep{start:06d}-{end:06d}.csv"
    )


def _checkpoint_directory(config) -> Path:
    return Path(config.output_dir) / CHECKPOINT_DIRECTORY_NAME


def _checkpoint_sort_key(path: Path) -> tuple[int, int, int, int]:
    match = CHECKPOINT_PATTERN.fullmatch(path.name)
    if match is None:
        raise FreshSimulationError(
            f"unexpected fresh KEF checkpoint filename: {path.name}"
        )
    return tuple(
        int(match.group(name))
        for name in ("dimension", "signal", "start", "end")
    )


def _is_checkpoint_temporary(path: Path) -> bool:
    return (
        path.is_file()
        and path.name.endswith(".tmp")
        and CHECKPOINT_PATTERN.fullmatch(path.name[:-4]) is not None
    )


def _checkpoint_index(metadata: Mapping[str, object]) -> dict[str, str]:
    try:
        value = json.loads(str(metadata.get("checkpoint_index", "{}")))
    except json.JSONDecodeError as error:
        raise FreshSimulationError(
            "fresh KEF checkpoint index is malformed"
        ) from error
    if not isinstance(value, dict) or any(
        not isinstance(name, str) or not isinstance(digest, str)
        for name, digest in value.items()
    ):
        raise FreshSimulationError("fresh KEF checkpoint index is malformed")
    for name, digest in value.items():
        if CHECKPOINT_PATTERN.fullmatch(name) is None:
            raise FreshSimulationError(
                "fresh KEF checkpoint index contains an invalid filename"
            )
        _validate_digest(digest, "checkpoint index digest")
    return value


def _load_checkpoint_rows(config, metadata, dimensions, base_grid, api):
    directory = _checkpoint_directory(config)
    if not directory.is_dir():
        raise FreshSimulationError(
            "fresh KEF running metadata exists without its checkpoint directory"
        )
    for path in tuple(directory.iterdir()):
        if _is_checkpoint_temporary(path):
            path.unlink()
    entries = tuple(directory.iterdir())
    unexpected = sorted(
        path.name
        for path in entries
        if not path.is_file() or CHECKPOINT_PATTERN.fullmatch(path.name) is None
    )
    if unexpected:
        raise FreshSimulationError(
            f"fresh KEF checkpoint directory has unexpected entries: {unexpected}"
        )
    recorded_index = _checkpoint_index(metadata)
    actual_names = {path.name for path in entries}
    if not set(recorded_index).issubset(actual_names):
        raise FreshSimulationError(
            "fresh KEF checkpoint index references a missing file"
        )
    rows: list[dict[str, str]] = []
    actual_index: dict[str, str] = {}
    seen_ranges: set[tuple[int, int, int, int]] = set()
    dimension_order = {
        dimension: index for index, dimension in enumerate(dimensions)
    }
    for path in sorted(
        entries,
        key=lambda item: (
            dimension_order.get(_checkpoint_sort_key(item)[0], len(dimensions)),
            *_checkpoint_sort_key(item)[1:],
        ),
    ):
        dimension, signal_index, start, end = _checkpoint_sort_key(path)
        if (
            dimension not in dimensions
            or not 0 <= signal_index < len(base_grid)
            or start < 1
            or end < start
            or end > _as_int(
                metadata["monte_carlo_replications"], "replications"
            )
            or (dimension, signal_index, start, end) in seen_ranges
        ):
            raise FreshSimulationError(
                f"fresh KEF checkpoint has an invalid cell/range: {path.name}"
            )
        seen_ranges.add((dimension, signal_index, start, end))
        digest = api._sha256(path)
        actual_index[path.name] = digest
        if path.name in recorded_index and recorded_index[path.name] != digest:
            raise FreshSimulationError(
                f"fresh KEF checkpoint hash mismatch: {path.name}"
            )
        checkpoint_rows = _read_rows(path, _checkpoint_columns(api))
        expected_count = (end - start + 1) * len(_method_specs(api))
        if len(checkpoint_rows) != expected_count:
            raise FreshSimulationError(
                f"fresh KEF checkpoint has a partial method block: {path.name}"
            )
        for row in checkpoint_rows:
            if row.pop("run_fingerprint") != metadata["run_fingerprint"]:
                raise FreshSimulationError(
                    "fresh KEF checkpoint was produced by another configuration"
                )
            if (
                _as_int(row["dimension"], "dimension") != dimension
                or _as_int(row["theta_index"], "theta_index") != signal_index
                or not start <= _as_int(
                    row["replication"], "replication"
                ) <= end
            ):
                raise FreshSimulationError(
                    f"fresh KEF checkpoint rows disagree with {path.name}"
                )
            rows.append(row)
    completed = _validate_raw_rows(
        rows, metadata, api, allow_prefix=True
    )
    # A crash may occur after an immutable batch is committed but before the
    # progress metadata is rewritten. Rebuild that lagging index only after
    # every batch has passed its fingerprint and canonical-prefix checks.
    metadata["checkpoint_index"] = json.dumps(
        actual_index, sort_keys=True, separators=(",", ":")
    )
    metadata["checkpoint_files"] = len(actual_index)
    metadata["completed_datasets"] = completed
    metadata["completed_replication_rows"] = len(rows)
    return rows


def _remove_owned_outputs(config, dimensions, base_grid) -> None:
    output_dir = Path(config.output_dir)
    checkpoint_directory = _checkpoint_directory(config)
    if checkpoint_directory.exists():
        if not checkpoint_directory.is_dir() or checkpoint_directory.is_symlink():
            raise FreshSimulationError(
                "refusing to replace a non-directory KEF checkpoint path"
            )
        entries = tuple(checkpoint_directory.iterdir())
        unexpected = sorted(
            path.name
            for path in entries
            if not path.is_file()
            or (
                CHECKPOINT_PATTERN.fullmatch(path.name) is None
                and not _is_checkpoint_temporary(path)
            )
        )
        if unexpected:
            raise FreshSimulationError(
                "refusing to remove checkpoint directory with unexpected "
                f"entries: {unexpected}"
            )
        for path in entries:
            path.unlink()
        checkpoint_directory.rmdir()
    for name in OWNED_OUTPUT_NAMES:
        if name == CHECKPOINT_DIRECTORY_NAME:
            continue
        path = output_dir / name
        if path.exists():
            if not path.is_file() and not path.is_symlink():
                raise FreshSimulationError(
                    f"refusing to replace non-file owned output: {path}"
                )
            path.unlink()


def _known_output_exists(config) -> bool:
    output_dir = Path(config.output_dir)
    return any((output_dir / name).exists() for name in OWNED_OUTPUT_NAMES)


def _write_progress_metadata(
    config, metadata, rows, api, updated: tuple[Path, int] | None = None
) -> None:
    method_count = _as_int(metadata["method_count"], "method_count")
    metadata["status"] = "running"
    metadata["completed_datasets"] = len(rows) // method_count
    metadata["completed_replication_rows"] = len(rows)
    metadata["updated_at_utc"] = _utc_now()
    index = _checkpoint_index(metadata)
    if updated is not None:
        path, _ = updated
        index[path.name] = api._sha256(path)
    metadata["checkpoint_index"] = json.dumps(
        index, sort_keys=True, separators=(",", ":")
    )
    metadata["checkpoint_files"] = len(index)
    api._atomic_write_metadata(
        Path(config.output_dir) / "run_metadata.csv", metadata
    )


def _write_cell_checkpoint(
    config,
    batch_rows,
    metadata,
    dimension,
    signal_index,
    start,
    end,
    api,
) -> Path:
    path = (
        _checkpoint_directory(config)
        / _checkpoint_name(dimension, signal_index, start, end)
    )
    if path.exists():
        raise FreshSimulationError(
            f"refusing to overwrite immutable KEF checkpoint: {path.name}"
        )
    checkpoint_rows = [
        {**row, "run_fingerprint": metadata["run_fingerprint"]}
        for row in batch_rows
    ]
    api._atomic_write_rows(
        path,
        _checkpoint_columns(api),
        checkpoint_rows,
    )
    return path


def _validate_complete_files(config, metadata, api) -> list[dict[str, str]]:
    replications_path = Path(config.output_dir) / "replications.csv"
    summary_path = Path(config.output_dir) / "summary.csv"
    if not replications_path.is_file() or not summary_path.is_file():
        raise FreshSimulationError(
            "completed fresh KEF metadata is missing standardized tables"
        )
    if metadata.get("replications_sha256") != api._sha256(replications_path):
        raise FreshSimulationError(
            "fresh KEF replications.csv does not match its recorded hash"
        )
    if metadata.get("summary_sha256") != api._sha256(summary_path):
        raise FreshSimulationError(
            "fresh KEF summary.csv does not match its recorded hash"
        )
    rows = _read_rows(replications_path, _raw_columns(api))
    summary = _read_rows(summary_path, api.PAIRED_SUMMARY_COLUMNS)
    validate_tables(metadata, rows, summary, api)
    return rows


def _resume_state(config, expected_metadata, dimensions, base_grid, api):
    metadata_path = Path(config.output_dir) / "run_metadata.csv"
    if not metadata_path.is_file():
        if _known_output_exists(config):
            raise FreshSimulationError(
                "fresh KEF outputs exist without run_metadata.csv"
            )
        return None, []
    metadata = api._read_metadata(metadata_path)
    if metadata.get("run_fingerprint") != expected_metadata["run_fingerprint"]:
        raise FreshSimulationError(
            "fresh KEF resume configuration, code, or runtime does not match"
        )
    status = metadata.get("status")
    if status == "complete":
        checkpoint_rows = _load_checkpoint_rows(
            config, metadata, dimensions, base_grid, api
        )
        standardized_rows = _validate_complete_files(config, metadata, api)
        if checkpoint_rows != standardized_rows:
            raise FreshSimulationError(
                "fresh KEF checkpoints disagree with completed replications.csv"
            )
        return metadata, None
    if status != "running":
        raise FreshSimulationError("fresh KEF metadata has an invalid status")
    rows = _load_checkpoint_rows(
        config, metadata, dimensions, base_grid, api
    )
    return metadata, rows


def execute(config, api) -> int:
    """Run, checkpoint, summarize, and validate a fresh KEF simulation."""
    dimensions, base_grid = _validate_config(config, api)
    core = _load_core()
    expected_metadata = _base_metadata(
        config, api, core, dimensions, base_grid
    )
    output_dir = Path(config.output_dir)

    if config.force:
        _remove_owned_outputs(config, dimensions, base_grid)
    elif not config.resume and _known_output_exists(config):
        raise FileExistsError(
            "fresh KEF outputs already exist; use --resume or --force"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    if config.resume:
        metadata, rows = _resume_state(
            config, expected_metadata, dimensions, base_grid, api
        )
        if rows is None:
            print(f"Reused validated complete fresh KEF output in {output_dir}.")
            if config.plot:
                return api.execute_plot_only(config)
            return 0
    else:
        metadata, rows = None, []

    checkpoint_directory = _checkpoint_directory(config)
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    if metadata is None:
        metadata = expected_metadata
    _write_progress_metadata(config, metadata, rows, api)

    specs = _method_specs(api)
    method_count = len(specs)
    completed_datasets = len(rows) // method_count
    dataset_keys = _dataset_keys(dimensions, base_grid, config.replications)
    batch_size = _as_int(
        metadata["checkpoint_batch_datasets"], "checkpoint batch datasets"
    )
    pending_rows: list[dict[str, object]] = []
    pending_start = 0
    pending_cell: tuple[int, int] | None = None

    for dataset_offset, (dimension, signal_index, replication) in enumerate(
        dataset_keys
    ):
        if dataset_offset < completed_datasets:
            continue
        cell = (dimension, signal_index)
        if pending_cell is None:
            pending_cell = cell
            pending_start = replication
        elif pending_cell != cell:
            raise FreshSimulationError(
                "internal KEF checkpoint batching crossed a signal cell"
            )
        base_signal = base_grid[signal_index]
        result = core.run_replication(
            n=config.n,
            dimension=dimension,
            theta1=config.theta1,
            theta2=base_signal / math.sqrt(dimension),
            sigma=config.sigma,
            bootstrap_draws=config.bootstrap_draws,
            seed=config.fresh_seed,
            signal_index=signal_index,
            replication=replication - 1,
        )
        new_rows = _dataset_rows(
            config,
            api,
            specs,
            dimension,
            signal_index,
            base_signal,
            replication - 1,
            result,
        )
        rows.extend(new_rows)
        pending_rows.extend(new_rows)
        cell_complete = replication == config.replications
        batch_complete = len(pending_rows) // method_count >= batch_size
        if batch_complete or cell_complete:
            path = _write_cell_checkpoint(
                config,
                pending_rows,
                metadata,
                dimension,
                signal_index,
                pending_start,
                replication,
                api,
            )
            _write_progress_metadata(
                config, metadata, rows, api, (path, len(pending_rows))
            )
            print(
                f"Checkpointed d={dimension}, signal={signal_index}, "
                f"replications {pending_start}-{replication} "
                f"({len(rows) // method_count}/{len(dataset_keys)} datasets)."
            )
            pending_rows = []
            pending_cell = None

    if pending_rows:
        raise FreshSimulationError(
            "internal KEF checkpoint batching left an uncommitted dataset"
        )
    _validate_raw_rows(rows, metadata, api, allow_prefix=False)
    summary = _summarize(rows, api)
    replications_path = output_dir / "replications.csv"
    summary_path = output_dir / "summary.csv"
    api._atomic_write_rows(
        replications_path, _raw_columns(api), rows
    )
    api._atomic_write_rows(
        summary_path, api.PAIRED_SUMMARY_COLUMNS, summary
    )
    metadata["replications_sha256"] = api._sha256(replications_path)
    metadata["summary_sha256"] = api._sha256(summary_path)
    metadata["replications_csv_rows"] = len(rows)
    metadata["summary_csv_rows"] = len(summary)
    metadata["completed_datasets"] = len(dataset_keys)
    metadata["completed_replication_rows"] = len(rows)

    plot_status = api.run_plot(config, summary_path)
    metadata["figure_status"] = plot_status
    for filename, key in (
        ("figure.pdf", "figure_pdf_sha256"),
        ("figure.png", "figure_png_sha256"),
    ):
        path = output_dir / filename
        if path.is_file() and path.stat().st_size > 0:
            metadata[key] = api._sha256(path)
    metadata["status"] = "complete"
    metadata["completed_at_utc"] = _utc_now()
    metadata["updated_at_utc"] = metadata["completed_at_utc"]
    api._atomic_write_metadata(output_dir / "run_metadata.csv", metadata)
    _validate_complete_files(config, metadata, api)
    print(
        f"Completed fresh KEF simulation: {len(dataset_keys)} paired datasets, "
        f"{len(rows)} method rows, {len(summary)} summary rows in {output_dir}."
    )
    print(f"Figure status: {plot_status}")
    return 0
