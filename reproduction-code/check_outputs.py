#!/usr/bin/env python
"""Strictly validate the six standardized reproduction output directories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import math
from pathlib import Path
import sys
from typing import Iterable


SIMULATIONS = (
    "01-gaussianity-mle",
    "02-kef-minimum-ksd",
    "02b-kef-score-matching",
    "03-conditional-gaussian-score-matching",
    "04-rbm-score-matching",
    "05-sphere-watson",
)
KEF_PAIRED_SCHEMA_VERSION = (
    "semiparametric_ksd_kef_minimum_ksd_paired_aggregate_v2"
)
KEF_FRESH_SCHEMA_VERSION = "semiparametric_ksd_kef_minimum_ksd_fresh_v1"
KEF_LEGACY_SCHEMA_VERSION = "semiparametric_ksd_kef_minimum_ksd_v1"
REQUIRED = (
    "replications.csv",
    "summary.csv",
    "run_metadata.csv",
    "figure.pdf",
    "figure.png",
)
EXPECTED_METHODS = {
    "01-gaussianity-mle": {
        "SKSD-parametric", "SKSD-influence", "SKS", "SMMD", "Brueck-corrected",
        "SKSD-naive-wild",
    },
    "02-kef-minimum-ksd": {
        "SKSD-parametric", "SKSD-influence", "SKS", "SMMD", "Brueck-corrected",
        "SKSD-naive-wild",
    },
    "03-conditional-gaussian-score-matching": {
        "SKSD-influence", "Brueck-corrected"
    },
    "04-rbm-score-matching": {"SKSD-influence", "Brueck-corrected"},
    "05-sphere-watson": {"SKSD-influence", "SKSD-parametric", "Brueck-corrected"},
}
KEF_DISPLAY_METHODS = {
    "SKSD-parametric": "SKSD-parametric",
    "SKSD-influence": "SKSD-influence",
    "SKS": "SKS",
    "SMMD": "SMMD (Key et al.)",
    "Brueck-corrected": "Brück et al.",
    "SKSD-naive-wild": "SKSD-naive wild",
}
REPLICATION_SCHEMAS = {
    "01-gaussianity-mle": (
        "alternative", "signal", "replication", "method", "method_label",
        "test", "calibration", "p_value", "reject", "seconds",
        "source_record", "aggregate_rejection_rate",
    ),
    "02-kef-minimum-ksd": (
        "simulation", "dimension", "theta_index", "theta2", "theta2_base",
        "replication", "method", "display_method", "test", "calibration",
        "nuisance_estimator", "p_value", "reject", "alpha", "data_seed",
        "bootstrap_seed", "paired_replication_key", "monte_carlo_arm",
        "paired_with_legacy", "source_archive",
    ),
    "03-conditional-gaussian-score-matching": (
        "alternative", "delta", "rep", "pvalue_influence", "pvalue_brueck",
        "pvalue_brueck_corrected", "seconds_influence", "seconds_brueck",
        "seconds_brueck_corrected", "score_matching_condition",
        "min_fitted_ring", "max_fitted_ring", "score_matching_boundary_hit",
        "brueck_boundary_refit_rate",
    ),
    "04-rbm-score-matching": (
        "schema_version", "design_version", "source_sha256", "run_fingerprint",
        "alternative", "signal", "rep", "pvalue_influence",
        "pvalue_brueck_corrected", "seconds_influence",
        "seconds_brueck_corrected", "observed_fit_seconds",
        "shared_preprocess_seconds", "observed_fit_iterations",
        "observed_fit_retried", "observed_fit_gradient_norm",
        "observed_fit_boundary", "observed_v", "observed_u",
        "influence_hessian_condition", "brueck_refits",
        "brueck_mean_iterations", "brueck_max_iterations", "brueck_retry_rate",
    ),
}
KEF_PAIRED_REPLICATION_SCHEMA = (
    "simulation", "dimension", "theta_index", "theta2", "theta2_base",
    "replication", "method", "display_method", "test", "calibration",
    "nuisance_estimator", "p_value", "reject", "alpha", "source_archive",
)
KEF_FRESH_REPLICATION_SCHEMA = KEF_PAIRED_REPLICATION_SCHEMA + ("seconds",)
SUMMARY_SCHEMAS = {
    "01-gaussianity-mle": (
        "alternative", "alternative_label", "signal", "method", "method_label",
        "test", "calibration", "replications", "rejections", "rejection_rate",
        "mcse", "mean_seconds", "source",
    ),
    "02-kef-minimum-ksd": (
        "simulation", "dimension", "theta_index", "theta2", "theta2_base",
        "method", "display_method", "test", "calibration", "nuisance_estimator",
        "rejection_rate", "rejected", "monte_carlo_repetitions", "mcse",
        "ci_lower", "ci_upper", "mean_seconds_per_replication",
        "monte_carlo_arm", "paired_within_arm", "paired_with_legacy",
        "replications_available", "source_archive",
    ),
    "03-conditional-gaussian-score-matching": (
        "alternative", "delta", "method", "alpha", "completed_reps",
        "expected_reps", "rejections", "rejection_rate", "mcse",
        "wilson95_lower", "wilson95_upper", "mean_pvalue", "mean_seconds",
        "sd_seconds", "mean_sm_condition", "score_matching_boundary_rate",
        "brueck_boundary_refit_rate",
    ),
    "04-rbm-score-matching": (
        "alternative", "signal", "shared_null", "method", "statistic",
        "rejection_rate", "rejection_ci_lower", "rejection_ci_upper",
        "mean_pvalue", "calibration_seconds_per_dataset",
        "calibration_speed_ratio_vs_influence", "total_seconds_per_dataset",
        "total_speed_ratio_vs_influence", "bootstrap_refits_per_dataset",
        "mean_brueck_retry_rate", "observed_fit_retry_rate", "n", "B", "reps",
    ),
}
KEF_PAIRED_SUMMARY_SCHEMA = (
    "simulation", "dimension", "theta_index", "theta2", "theta2_base",
    "method", "display_method", "test", "calibration",
    "nuisance_estimator", "rejection_rate", "rejected",
    "monte_carlo_repetitions", "mcse", "ci_lower", "ci_upper",
    "mean_seconds_per_replication", "monte_carlo_arm",
    "paired_observed_datasets", "shared_bootstrap_randomness_across_all_methods",
    "paired_contrasts_reconstructible", "replications_available",
    "source_archive",
)


def _rows(
    path: Path,
    expected_schema: tuple[str, ...] | None = None,
    *,
    allow_empty: bool = False,
) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        if expected_schema is not None and tuple(reader.fieldnames) != expected_schema:
            raise ValueError(f"CSV has the wrong schema: {path}")
        rows = list(reader)
    if not rows and not allow_empty:
        raise ValueError(f"CSV has no data rows: {path}")
    return rows


def _metadata(path: Path) -> dict[str, str]:
    rows = _rows(path)
    if set(rows[0]) != {"key", "value"}:
        raise ValueError(f"metadata must have exactly key,value columns: {path}")
    keys = [row["key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"metadata contains duplicate keys: {path}")
    values = {row["key"]: row["value"] for row in rows}
    required = {"schema_version", "run_fingerprint"}
    missing = required.difference(values)
    if missing:
        raise ValueError(f"metadata is missing {sorted(missing)}: {path}")
    if any(not values[key].strip() for key in required):
        raise ValueError(f"metadata has a blank schema/fingerprint value: {path}")
    return values


def _validate_signature(path: Path) -> None:
    header = path.read_bytes()[:8]
    if path.suffix == ".pdf" and not header.startswith(b"%PDF-"):
        raise ValueError(f"not a readable PDF: {path}")
    if path.suffix == ".png" and header != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a readable PNG: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_kef_entry_point():
    name = "_reproduction_output_checker_kef_02"
    path = Path(__file__).resolve().parent / "simulation-code" / "study-scripts" / "02_kef_minimum_ksd.py"
    existing = sys.modules.get(name)
    if existing is not None and Path(existing.__file__).resolve() == path.resolve():
        return existing
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load KEF reproduction entry point: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    try:
        specification.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def _validate_saved_csv(directory: Path, simulation: str) -> tuple[int, int]:
    """Reuse the plotter's numerical checks for read-only historical CSVs."""
    scripts = {
        "01-gaussianity-mle": "01_gaussianity_mle.py",
        "03-conditional-gaussian-score-matching": "03_conditional_gaussian_score_matching.py",
        "04-rbm-score-matching": "04_rbm_score_matching.py",
    }
    if simulation not in scripts:
        raise ValueError(f"unsupported saved-csv study: {simulation}")
    path = Path(__file__).resolve().parent / "simulation-code" / "study-scripts" / scripts[simulation]
    name = "_saved_csv_checker_" + simulation.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load saved-result validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    validated = module.validate_saved_results(directory)
    summary = validated[0]
    metadata = _metadata(directory / "run_metadata.csv")
    for extension in ("pdf", "png"):
        figure = directory / f"figure.{extension}"
        _validate_signature(figure)
        if metadata.get(f"figure_{extension}_sha256") != _sha256(figure):
            raise ValueError(f"saved-result figure checksum mismatch: {figure}")
    raw_count = len(_rows(directory / "replications.csv")) if simulation != "01-gaussianity-mle" else 0
    return raw_count, len(summary)


def _validate_sphere(directory: Path) -> tuple[int, int]:
    path = Path(__file__).resolve().parent / "simulation-code" / "study-scripts" / "05_sphere_watson.py"
    name = "_reproduction_output_checker_sphere_05"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load spherical result validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    summary, metadata = module.validate_saved_results(directory)
    if {row["method"] for row in summary} != EXPECTED_METHODS["05-sphere-watson"]:
        raise ValueError("sphere results must include all three configured manuscript methods")
    for extension in ("pdf", "png"):
        figure = directory / f"figure.{extension}"
        _validate_signature(figure)
        if metadata.get(f"figure_{extension}_sha256") != _sha256(figure):
            raise ValueError(f"sphere figure checksum mismatch: {figure}")
    return len(_rows(directory / "replications.csv")), len(summary)


def _validate_score_matching_kef(directory: Path) -> tuple[int, int]:
    missing = [name for name in REQUIRED if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"02b-kef-score-matching is incomplete; missing: {', '.join(missing)}"
        )
    path = (
        Path(__file__).resolve().parent
        / "simulation-code" / "study-scripts" / "02b_kef_score_matching.py"
    )
    name = "_reproduction_output_checker_kef_02b"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load score-matching KEF result validator: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    _metadata(directory / "run_metadata.csv")
    config, metadata, summary = module.validate_complete_output(directory)
    replications = _rows(directory / "replications.csv", module.REPLICATION_FIELDS)
    raw_count = len(replications)
    if metadata.get("replications_csv_rows") != str(raw_count):
        raise ValueError("score-matching KEF replication count disagrees with metadata")
    if metadata.get("summary_csv_rows") != str(len(summary)):
        raise ValueError("score-matching KEF summary count disagrees with metadata")
    if {row["method"] for row in summary} != set(module.METHODS):
        raise ValueError("score-matching KEF summary has incomplete methods")
    if raw_count != len(module.SETUPS) * len(config.signals) * config.R:
        raise ValueError("score-matching KEF output has an incomplete replication grid")
    for saved, expected in zip(
        summary, module.summarize(replications, config, complete=True)
    ):
        if not math.isclose(
            float(saved["mean_seconds_per_dataset"]),
            float(expected["mean_seconds_per_dataset"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("score-matching KEF summary timing disagrees with raw results")
    for extension in ("pdf", "png"):
        figure = directory / f"figure.{extension}"
        if figure.stat().st_size <= 0:
            raise ValueError(f"output is empty: {figure}")
        _validate_signature(figure)
        if metadata.get(f"figure_{extension}_sha256") != _sha256(figure):
            raise ValueError(f"score-matching KEF figure checksum mismatch: {figure}")
    return raw_count, len(summary)


def validate(
    results_root: Path,
    simulations: Iterable[str] = SIMULATIONS,
) -> None:
    simulations = tuple(simulations)
    unexpected_json = sorted(results_root.rglob("*.json")) if results_root.exists() else []
    if unexpected_json:
        rendered = "\n  ".join(str(path) for path in unexpected_json)
        raise ValueError(f"JSON output is not part of this workflow:\n  {rendered}")

    report: list[tuple[str, str, int, str]] = []
    for simulation in simulations:
        directory = results_root / simulation
        manifest_path = directory / "run_metadata.csv"
        if simulation == "02b-kef-score-matching":
            raw_count, summary_count = _validate_score_matching_kef(directory)
            print(f"validated {simulation}: {raw_count} replication rows, {summary_count} summary rows")
            continue
        if simulation == "05-sphere-watson":
            raw_count, summary_count = _validate_sphere(directory)
            print(f"validated {simulation}: {raw_count} replication rows, {summary_count} summary rows")
            continue
        if manifest_path.is_file() and _metadata(manifest_path).get("schema_version") == "saved-csv-v1":
            raw_count, summary_count = _validate_saved_csv(directory, simulation)
            print(f"validated {simulation}: saved CSV results, {raw_count} raw rows, "
                  f"{summary_count} plotted summary rows")
            continue
        missing = [name for name in REQUIRED if not (directory / name).is_file()]
        if missing:
            raise FileNotFoundError(
                f"{simulation} is incomplete; missing: {', '.join(missing)}"
            )
        metadata = _metadata(directory / "run_metadata.csv")
        paired_kef = (
            simulation == "02-kef-minimum-ksd"
            and metadata.get("schema_version") == KEF_PAIRED_SCHEMA_VERSION
        )
        fresh_kef = (
            simulation == "02-kef-minimum-ksd"
            and metadata.get("schema_version") == KEF_FRESH_SCHEMA_VERSION
        )
        if simulation == "02-kef-minimum-ksd" and not (fresh_kef or paired_kef):
            if not (
                metadata.get("schema_version") == KEF_LEGACY_SCHEMA_VERSION
                and metadata.get("archive_format") == "legacy-unpaired"
            ):
                raise ValueError(
                    "02-kef-minimum-ksd has an unrecognized saved-results schema"
                )
        replication_schema = (
            KEF_FRESH_REPLICATION_SCHEMA
            if fresh_kef
            else (
                KEF_PAIRED_REPLICATION_SCHEMA
                if paired_kef else REPLICATION_SCHEMAS[simulation]
            )
        )
        summary_schema = (
            KEF_PAIRED_SUMMARY_SCHEMA
            if fresh_kef or paired_kef else SUMMARY_SCHEMAS[simulation]
        )
        replications = _rows(
            directory / "replications.csv",
            replication_schema,
            allow_empty=paired_kef,
        )
        summary = _rows(directory / "summary.csv", summary_schema)
        if fresh_kef:
            _load_kef_entry_point().validate_fresh_standardized_tables(
                metadata,
                directory / "replications.csv",
                directory / "summary.csv",
            )
        for key, name in (
            ("replications_sha256", "replications.csv"),
            ("summary_sha256", "summary.csv"),
            ("figure_pdf_sha256", "figure.pdf"),
            ("figure_png_sha256", "figure.png"),
        ):
            if metadata.get(key) != _sha256(directory / name):
                raise ValueError(f"recorded {key} does not match {simulation}/{name}")
        if "method" not in summary[0]:
            raise ValueError(f"summary has no method column: {directory / 'summary.csv'}")
        methods = {row["method"] for row in summary}
        expected_methods = EXPECTED_METHODS[simulation]
        if paired_kef:
            expected_methods = expected_methods - {"SKSD-naive-wild"}
        if methods != expected_methods:
            raise ValueError(
                f"summary methods are wrong for {simulation}: {sorted(methods)}"
            )
        if simulation == "01-gaussianity-mle":
            labels = {row.get("method_label", "") for row in summary}
            required_labels = {"SKS (Lilliefors)", "SMMD (Key et al.)", "Brück et al."}
            if not required_labels.issubset(labels):
                raise ValueError(f"Gaussianity display labels are incomplete: {sorted(labels)}")
        if simulation == "02-kef-minimum-ksd":
            labels = {row.get("display_method", "") for row in summary}
            required_labels = {"SKS", "SMMD (Key et al.)", "Brück et al."}
            if not required_labels.issubset(labels) or "SKS (Lilliefors)" in labels:
                raise ValueError(f"KEF display labels are incorrect: {sorted(labels)}")
            for row in summary:
                if row.get("display_method") != KEF_DISPLAY_METHODS.get(
                    row.get("method", "")
                ):
                    raise ValueError(
                        "KEF method/display-method mapping is inconsistent"
                    )
            if paired_kef:
                required_provenance = {
                    "archive_format": "paired-aggregate",
                    "paired_observed_datasets": "true",
                    "shared_bootstrap_randomness_across_all_methods": "false",
                    "paired_contrasts_reconstructible": "false",
                    "replication_level_pvalues_available": "false",
                    "replication_level_decisions_available": "false",
                    "replications_csv_is_header_only": "true",
                    "replications_csv_rows": "0",
                }
                for key, expected in required_provenance.items():
                    if metadata.get(key) != expected:
                        raise ValueError(
                            f"KEF paired metadata must record {key}={expected}"
                        )
                _load_kef_entry_point()._validate_paired_standardized_tables(
                    metadata,
                    directory / "replications.csv",
                    directory / "summary.csv",
                )
                if replications:
                    raise ValueError(
                        "KEF paired aggregate export must not fabricate "
                        "replication-level rows"
                    )
                if metadata.get("summary_csv_rows") != str(len(summary)):
                    raise ValueError(
                        "KEF paired summary row count disagrees with metadata"
                    )
                try:
                    expected_repetitions = int(
                        metadata["monte_carlo_replications"]
                    )
                except (KeyError, ValueError) as error:
                    raise ValueError(
                        "KEF paired metadata has an invalid replication count"
                    ) from error
                if expected_repetitions <= 0:
                    raise ValueError(
                        "KEF paired metadata replication count must be positive"
                    )
                cells: dict[tuple[int, int], set[str]] = {}
                for row in summary:
                    if (
                        row["paired_observed_datasets"] != "True"
                        or row[
                            "shared_bootstrap_randomness_across_all_methods"
                        ] != "False"
                        or row["paired_contrasts_reconstructible"] != "False"
                        or row["replications_available"] != "False"
                    ):
                        raise ValueError(
                            "KEF paired summary has inconsistent aggregate-only "
                            "provenance flags"
                        )
                    try:
                        dimension = int(row["dimension"])
                        theta_index = int(row["theta_index"])
                        repetitions = int(row["monte_carlo_repetitions"])
                        rejected = int(row["rejected"])
                        rate = float(row["rejection_rate"])
                        seconds = float(row["mean_seconds_per_replication"])
                    except ValueError as error:
                        raise ValueError(
                            "KEF paired summary contains malformed numerical values"
                        ) from error
                    if (
                        repetitions != expected_repetitions
                        or not 0 <= rejected <= repetitions
                        or abs(rate - rejected / repetitions) > 1e-12
                        or not math.isfinite(seconds)
                        or seconds <= 0
                    ):
                        raise ValueError(
                            "KEF paired summary violates its full-scale aggregate "
                            "count/timing contract"
                        )
                    methods_in_cell = cells.setdefault(
                        (dimension, theta_index), set()
                    )
                    if row["method"] in methods_in_cell:
                        raise ValueError("KEF paired summary contains a duplicate row")
                    methods_in_cell.add(row["method"])
                if any(
                    methods_in_cell != expected_methods
                    for methods_in_cell in cells.values()
                ):
                    raise ValueError(
                        "KEF paired summary must contain five methods in every cell"
                    )
                try:
                    selected_cells: set[tuple[int, int]] = set()
                    for block in metadata["selected_theta_indices"].split(";"):
                        dimension_text, indices_text = block.split(":", 1)
                        dimension = int(dimension_text.removeprefix("d"))
                        selected_cells.update(
                            (dimension, int(index))
                            for index in indices_text.split(",")
                            if index
                        )
                except (KeyError, ValueError) as error:
                    raise ValueError(
                        "KEF paired metadata has invalid selected theta indices"
                    ) from error
                if set(cells) != selected_cells:
                    raise ValueError(
                        "KEF paired summary cells disagree with metadata"
                    )
                publication_design = metadata.get("publication_design") == "true"
                if publication_design:
                    expected_cells = {
                        (dimension, theta_index)
                        for dimension in (1, 2, 3)
                        for theta_index in range(13)
                    }
                    if set(cells) != expected_cells or len(summary) != 195:
                        raise ValueError(
                            "full KEF paired output must contain all 39 "
                            "dimension/signal cells and 195 method rows"
                        )
                    if (
                        metadata.get("mode") != "full-paired-aggregate-export"
                        or expected_repetitions != 500
                    ):
                        raise ValueError(
                            "publication KEF metadata must use the full paired "
                            "mode with 500 replications"
                        )
                elif metadata.get("mode") == "full-paired-aggregate-export":
                    raise ValueError(
                        "full KEF paired mode must declare publication_design=true"
                    )
        for row in summary:
            try:
                rejection_rate = float(row["rejection_rate"])
            except (KeyError, ValueError) as error:
                raise ValueError(f"summary has an invalid rejection rate: {simulation}") from error
            if not 0.0 <= rejection_rate <= 1.0:
                raise ValueError(f"summary rejection rate is outside [0,1]: {simulation}")
        for row in replications:
            for key, value in row.items():
                if value == "":
                    continue
                if key == "p_value" or key.startswith("pvalue_"):
                    try:
                        p_value = float(value)
                    except ValueError as error:
                        raise ValueError(f"invalid p-value in {simulation}") from error
                    if not 0.0 <= p_value <= 1.0:
                        raise ValueError(f"p-value outside [0,1] in {simulation}")
                if "seconds" in key:
                    try:
                        seconds = float(value)
                    except ValueError as error:
                        raise ValueError(f"invalid runtime in {simulation}") from error
                    if seconds < 0.0:
                        raise ValueError(f"negative runtime in {simulation}")
        for name in REQUIRED:
            path = directory / name
            if path.stat().st_size <= 0:
                raise ValueError(f"output is empty: {path}")
            if path.suffix in {".pdf", ".png"}:
                _validate_signature(path)
            report.append((simulation, name, path.stat().st_size, _sha256(path)[:12]))
        print(
            f"validated {simulation}: {len(replications)} replication rows, "
            f"{len(summary)} summary rows"
        )

    print(f"\nValidated {len(simulations)} simulations in {results_root}")
    for simulation, name, size, digest in report:
        print(f"  {simulation}/{name}: {size} bytes, sha256={digest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        required=True,
        help="profile directory containing the six named simulation folders",
    )
    args = parser.parse_args()
    validate(args.results_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
