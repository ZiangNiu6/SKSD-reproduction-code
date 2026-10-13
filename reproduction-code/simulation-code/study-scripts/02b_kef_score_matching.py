#!/usr/bin/env python3
"""Reproduce the optional 1D KEF score-matching IA/NWB power comparison.

This is a companion to study 02, whose published design fits by minimum KSD.
Here the fitted spans are (phi_0), (phi_0, phi_1), and
(phi_0, phi_1, phi_2); the omitted alternative is phi_3. Rejection always
uses each method's nominal p <= alpha rule, without size correction.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Iterable

import numpy as np


SCRIPT = Path(__file__).resolve()
SIMULATION_CODE = SCRIPT.parent.parent
PACKAGE_ROOT = SIMULATION_CODE.parent
CORE_PATH = SIMULATION_CODE / "numerical-helpers" / "_kef_score_matching_core.py"
REFERENCE_CORE_PATH = SIMULATION_CODE / "numerical-helpers" / "_kef_core.py"
PLOTTER_PATH = SIMULATION_CODE / "figure-code" / "plot_02b_kef_score_matching.R"
SCHEMA_VERSION = "kef-score-matching-nominal-v1"
SLUG = "02b-kef-score-matching"
SETUPS = {"p1": (0,), "p2": (0, 1), "p3": (0, 1, 2)}
FULL_SIGNALS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
QUICK_SIGNALS = (0.0, 0.3, 0.5)
DEFAULT_SEEDS = {"p1": 20261206, "p2": 20261202, "p3": 20261204}
METHODS = ("SKSD-IA", "SKSD-NWB")
REPLICATION_FIELDS = (
    "setup", "fitted_bases", "signal", "eta", "rep", "data_seed",
    "bootstrap_seed", "theta_hat_0", "theta_hat_1", "theta_hat_2",
    "sm_condition_number", "sm_equation_residual", "bandwidth",
    "observed_v", "pvalue_ia", "pvalue_nwb", "reject_ia", "reject_nwb",
    "sample_proposals", "log_envelope", "seconds",
)
SUMMARY_FIELDS = (
    "setup", "fitted_bases", "signal", "method", "rejections",
    "replications", "rejection_rate", "mcse", "mean_seconds_per_dataset",
    "paired_gap_ia_minus_nwb", "paired_mcse",
)
OWNED_FILES = (
    "replications.csv", "summary.csv", "run_metadata.csv", "figure.pdf",
    "figure.png",
)


@dataclass(frozen=True)
class Config:
    output_dir: Path
    signals: tuple[float, ...]
    n: int
    B: int
    R: int
    alpha: float
    theta0: float
    sigma: float
    seeds: tuple[int, int, int]
    quick: bool
    plot: bool

    def seed_for(self, setup: str) -> int:
        return self.seeds[("p1", "p2", "p3").index(setup)]


def default_output_dir(quick: bool = False) -> Path:
    return PACKAGE_ROOT / "results" / (
        "simulation-analysis-check" if quick else "simulation-analysis"
    ) / SLUG


def parse_signals(value: str) -> tuple[float, ...]:
    try:
        signals = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("--signals must contain comma-separated numbers") from error
    if not signals or any(not math.isfinite(x) or x < 0 for x in signals):
        raise argparse.ArgumentTypeError("--signals must be nonnegative finite numbers")
    if signals[0] != 0 or any(a >= b for a, b in zip(signals, signals[1:])):
        raise argparse.ArgumentTypeError("--signals must start at zero and increase strictly")
    return signals


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", dest="quick", action="store_true", help="small deterministic check profile")
    parser.add_argument("--quick", dest="quick", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plot-only", action="store_true", help="validate saved CSVs and recreate the figure")
    parser.add_argument("--resume", action="store_true", help="continue a compatible interrupted run")
    parser.add_argument("--force", action="store_true", help="replace only this study's named output files")
    parser.add_argument("--no-plot", action="store_true", help="skip the R figure after simulation")
    parser.add_argument("--output-dir", type=Path,
                        help=f"result directory (default: {default_output_dir()}; --check uses simulation-analysis-check/)")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--bootstrap-draws", type=int)
    parser.add_argument("--replications", type=int)
    parser.add_argument("--signals", type=parse_signals)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--theta0", type=float)
    parser.add_argument("--sigma", type=float)
    for setup in SETUPS:
        parser.add_argument(f"--{setup}-seed", type=int)
    return parser


def config_from_args(args: argparse.Namespace) -> Config:
    if args.resume and args.force:
        raise ValueError("--resume and --force cannot be used together")
    if args.plot_only and (args.resume or args.force or args.no_plot):
        raise ValueError("--plot-only cannot be combined with --resume, --force, or --no-plot")
    if args.plot_only and any(
        getattr(args, name) is not None for name in (
            "sample_size", "bootstrap_draws", "replications", "signals", "alpha",
            "theta0", "sigma", "p1_seed", "p2_seed", "p3_seed",
        )
    ):
        raise ValueError("--plot-only reads the saved configuration; omit simulation overrides")
    quick = bool(args.quick)
    config = Config(
        output_dir=(args.output_dir or default_output_dir(quick)).expanduser().resolve(),
        signals=args.signals or (QUICK_SIGNALS if quick else FULL_SIGNALS),
        n=args.sample_size if args.sample_size is not None else (30 if quick else 100),
        B=args.bootstrap_draws if args.bootstrap_draws is not None else (19 if quick else 500),
        R=args.replications if args.replications is not None else (3 if quick else 500),
        alpha=args.alpha if args.alpha is not None else 0.05,
        theta0=args.theta0 if args.theta0 is not None else 10.0,
        sigma=args.sigma if args.sigma is not None else 3.0,
        seeds=tuple(
            getattr(args, f"{setup}_seed")
            if getattr(args, f"{setup}_seed") is not None else DEFAULT_SEEDS[setup]
            for setup in SETUPS
        ),
        quick=quick,
        plot=not args.no_plot,
    )
    if config.n < 4 or config.B < 1 or config.R < 2:
        raise ValueError("require sample size >= 4, bootstrap draws >= 1, replications >= 2")
    if not (0 < config.alpha < 1 and config.theta0 > 0 and config.sigma > 0):
        raise ValueError("require 0 < alpha < 1, theta0 > 0, sigma > 0")
    if any(seed < 0 for seed in config.seeds):
        raise ValueError("seeds must be nonnegative")
    return config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_core():
    name = "_reproduction_kef_score_matching_core"
    specification = importlib.util.spec_from_file_location(name, CORE_PATH)
    if specification is None or specification.loader is None:
        raise ImportError(f"cannot load {CORE_PATH}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def atomic_csv(path: Path, fields: tuple[str, ...], rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_csv(path: Path, fields: tuple[str, ...], *, allow_empty: bool = False) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != fields:
            raise ValueError(f"unexpected CSV schema: {path}")
        rows = list(reader)
    if not rows and not allow_empty:
        raise ValueError(f"empty CSV: {path}")
    return rows


def read_metadata(path: Path) -> dict[str, str]:
    rows = read_csv(path, ("key", "value"))
    keys = [row["key"] for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"duplicate metadata keys: {path}")
    return {row["key"]: row["value"] for row in rows}


def write_metadata(path: Path, values: dict[str, str]) -> None:
    atomic_csv(path, ("key", "value"), (
        {"key": key, "value": value} for key, value in sorted(values.items())
    ))


def identity(config: Config, phi3_sd: float) -> dict[str, str]:
    values = {
        "schema_version": SCHEMA_VERSION,
        "study": SLUG,
        "workflow": "simulate",
        "sample_size": str(config.n),
        "bootstrap_draws": str(config.B),
        "replications": str(config.R),
        "alpha": repr(config.alpha),
        "theta0": repr(config.theta0),
        "sigma": repr(config.sigma),
        "signals": ";".join(repr(signal) for signal in config.signals),
        "fitted_spans": "p1:0;p2:0,1;p3:0,1,2",
        "alternative_basis": "3",
        "signal_scale": "t=eta*SD_null(phi3)",
        "phi3_sd": repr(phi3_sd),
        "p1_seed": str(config.seed_for("p1")),
        "p2_seed": str(config.seed_for("p2")),
        "p3_seed": str(config.seed_for("p3")),
        "quick": str(config.quick).lower(),
        "rejection_rule": "nominal p<=alpha; no size correction",
        "bootstrap": "paired centered Gaussian multipliers; consume Efron counts first",
        "bandwidth": "1.5 * observed median pairwise distance",
        "estimator": "ordinary Hyvarinen score matching without ridge",
        "driver_sha256": sha256(SCRIPT),
        "core_sha256": sha256(CORE_PATH),
        "reference_core_sha256": sha256(REFERENCE_CORE_PATH),
    }
    payload = "\n".join(f"{key}={value}" for key, value in sorted(values.items()))
    values["run_fingerprint"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return values


def config_from_metadata(output_dir: Path, meta: dict[str, str]) -> Config:
    if meta.get("schema_version") != SCHEMA_VERSION or meta.get("study") != SLUG:
        raise ValueError(f"unrecognized score-matching KEF metadata in {output_dir}")
    config = Config(
        output_dir=output_dir,
        signals=tuple(float(value) for value in meta["signals"].split(";")),
        n=int(meta["sample_size"]), B=int(meta["bootstrap_draws"]),
        R=int(meta["replications"]), alpha=float(meta["alpha"]),
        theta0=float(meta["theta0"]), sigma=float(meta["sigma"]),
        seeds=tuple(int(meta[f"{setup}_seed"]) for setup in SETUPS),
        quick=meta["quick"] == "true", plot=True,
    )
    return config


def validate_rows(
    rows: list[dict[str, str]], config: Config, *, complete: bool,
    phi3_sd: float,
) -> None:
    expected = {(setup, signal, rep)
                for setup in SETUPS for signal in config.signals for rep in range(config.R)}
    seen: set[tuple[str, float, int]] = set()
    seeds: set[tuple[int, int]] = set()
    for row in rows:
        setup = row["setup"]
        if setup not in SETUPS:
            raise ValueError(f"unknown fitted span: {setup}")
        signal, rep = float(row["signal"]), int(row["rep"])
        key = (setup, signal, rep)
        if key not in expected or key in seen:
            raise ValueError(f"unexpected or duplicate replication: {key}")
        seen.add(key)
        if row["fitted_bases"] != ",".join(map(str, SETUPS[setup])):
            raise ValueError(f"incorrect basis label at {key}")
        data_seed, boot_seed = int(row["data_seed"]), int(row["bootstrap_seed"])
        signal_code = 6 if signal == 0 else int.from_bytes(
            hashlib.sha256(format(signal, ".12g").encode("ascii")).digest()[:4],
            "little",
        )
        expected_data_seed = int(np.random.SeedSequence(
            [config.seed_for(setup), 1, signal_code, rep, 0]
        ).generate_state(1)[0])
        expected_boot_seed = int(np.random.SeedSequence(
            [config.seed_for(setup), 1, signal_code, rep, 16]
        ).generate_state(1)[0])
        if (data_seed, boot_seed) != (expected_data_seed, expected_boot_seed):
            raise ValueError(f"seed pair disagrees with run configuration at {key}")
        if (data_seed, boot_seed) in seeds:
            raise ValueError(f"duplicate seed pair at {key}")
        seeds.add((data_seed, boot_seed))
        if not math.isclose(float(row["eta"]), signal / phi3_sd,
                            rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"alternative coefficient disagrees with signal at {key}")
        for index in range(3):
            value = row[f"theta_hat_{index}"]
            if index < len(SETUPS[setup]):
                if not math.isfinite(float(value)):
                    raise ValueError(f"nonfinite fitted coefficient at {key}")
            elif value != "":
                raise ValueError(f"unused fitted coefficient is nonblank at {key}")
        for method in ("ia", "nwb"):
            value = float(row[f"pvalue_{method}"])
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"invalid p-value at {key}")
            numerator = value * (config.B + 1)
            if abs(numerator - round(numerator)) > 1e-8:
                raise ValueError(f"p-value is off the B-draw grid at {key}")
            if int(row[f"reject_{method}"]) != int(value <= config.alpha):
                raise ValueError(f"decision disagrees with nominal p-value at {key}")
        if not math.isfinite(float(row["sm_condition_number"])) or float(row["sm_condition_number"]) <= 0:
            raise ValueError(f"invalid score-matching condition at {key}")
        if (not math.isfinite(float(row["sm_equation_residual"])) or
                float(row["sm_equation_residual"]) < 0 or
                float(row["sm_equation_residual"]) >= 1e-8):
            raise ValueError(f"invalid score-matching equation residual at {key}")
        if (not math.isfinite(float(row["bandwidth"])) or
                float(row["bandwidth"]) <= 0 or
                not math.isfinite(float(row["observed_v"])) or
                not math.isfinite(float(row["log_envelope"])) or
                int(row["sample_proposals"]) < config.n or
                not math.isfinite(float(row["seconds"])) or
                float(row["seconds"]) <= 0):
            raise ValueError(f"invalid numerical diagnostic at {key}")
    if complete and seen != expected:
        raise ValueError(f"incomplete replication grid: {len(seen)}/{len(expected)}")


def summarize(rows: list[dict[str, str]], config: Config, *, complete: bool) -> list[dict]:
    by_cell: dict[tuple[str, float], list[dict[str, str]]] = {}
    for row in rows:
        by_cell.setdefault((row["setup"], float(row["signal"])), []).append(row)
    output = []
    for setup, bases in SETUPS.items():
        for signal in config.signals:
            cell = by_cell.get((setup, signal), [])
            if len(cell) != config.R:
                if complete:
                    raise ValueError(f"incomplete cell: {setup}, t={signal}")
                continue
            ia = np.array([int(row["reject_ia"]) for row in cell], dtype=float)
            nwb = np.array([int(row["reject_nwb"]) for row in cell], dtype=float)
            difference = ia - nwb
            gap = float(difference.mean())
            paired_mcse = float(difference.std(ddof=1) / math.sqrt(config.R))
            mean_seconds = float(np.mean([float(row["seconds"]) for row in cell]))
            for method, indicators in zip(METHODS, (ia, nwb)):
                rate = float(indicators.mean())
                output.append({
                    "setup": setup,
                    "fitted_bases": ",".join(map(str, bases)),
                    "signal": signal,
                    "method": method,
                    "rejections": int(indicators.sum()),
                    "replications": config.R,
                    "rejection_rate": rate,
                    "mcse": math.sqrt(rate * (1 - rate) / config.R),
                    "mean_seconds_per_dataset": mean_seconds,
                    "paired_gap_ia_minus_nwb": gap,
                    "paired_mcse": paired_mcse,
                })
    return output


def validate_complete_output(output_dir: Path) -> tuple[Config, dict[str, str], list[dict[str, str]]]:
    meta = read_metadata(output_dir / "run_metadata.csv")
    if meta.get("status") != "complete" or meta.get("run_complete") != "true":
        raise ValueError(f"score-matching KEF output is incomplete: {output_dir}")
    config = config_from_metadata(output_dir, meta)
    raw_path, summary_path = output_dir / "replications.csv", output_dir / "summary.csv"
    if sha256(raw_path) != meta.get("replications_sha256"):
        raise ValueError("replications.csv checksum does not match metadata")
    if sha256(summary_path) != meta.get("summary_sha256"):
        raise ValueError("summary.csv checksum does not match metadata")
    raw = read_csv(raw_path, REPLICATION_FIELDS)
    validate_rows(raw, config, complete=True, phi3_sd=float(meta["phi3_sd"]))
    summary = read_csv(summary_path, SUMMARY_FIELDS)
    expected_summary = summarize(raw, config, complete=True)
    if len(summary) != len(expected_summary):
        raise ValueError("summary.csv has the wrong number of cells")
    for saved, expected in zip(summary, expected_summary):
        for field in ("setup", "fitted_bases", "method"):
            if saved[field] != str(expected[field]):
                raise ValueError(f"summary.csv disagrees with raw results: {field}")
        for field in ("signal", "rejections", "replications", "rejection_rate",
                      "mcse", "paired_gap_ia_minus_nwb", "paired_mcse"):
            if not math.isclose(float(saved[field]), float(expected[field]), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"summary.csv disagrees with raw results: {field}")
    return config, meta, summary


def run_plot(config: Config, meta: dict[str, str]) -> None:
    executable = shutil.which("Rscript")
    if executable is None:
        raise RuntimeError("Rscript is required to render figure.pdf and figure.png")
    if not PLOTTER_PATH.is_file():
        raise FileNotFoundError(PLOTTER_PATH)
    output = config.output_dir
    command = [executable, str(PLOTTER_PATH), str(output / "summary.csv"),
               str(output / "figure.pdf"), str(output / "figure.png"), repr(config.alpha)]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError("R plotting failed:\n" + completed.stdout + completed.stderr)
    for filename, magic in (("figure.pdf", b"%PDF-"), ("figure.png", b"\x89PNG\r\n\x1a\n")):
        if not (output / filename).read_bytes().startswith(magic):
            raise ValueError(f"invalid generated {filename}")
        meta[filename.replace(".", "_") + "_sha256"] = sha256(output / filename)
    meta["plotter_sha256"] = sha256(PLOTTER_PATH)
    write_metadata(output / "run_metadata.csv", meta)
    print(f"Wrote {output / 'figure.pdf'} and {output / 'figure.png'}", flush=True)


def execute(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    output = config.output_dir
    metadata_path = output / "run_metadata.csv"
    raw_path, summary_path = output / "replications.csv", output / "summary.csv"
    if args.plot_only:
        saved_config, meta, _ = validate_complete_output(output)
        run_plot(saved_config, meta)
        return 0

    core = load_core()
    phi3_sd = float(core.null_basis_sd(config.theta0, config.sigma, 3))
    if not math.isfinite(phi3_sd) or phi3_sd <= 0:
        raise ValueError("invalid null standard deviation of phi3")
    expected_identity = identity(config, phi3_sd)
    output.mkdir(parents=True, exist_ok=True)
    if args.force:
        for filename in OWNED_FILES:
            (output / filename).unlink(missing_ok=True)
    if metadata_path.exists() or raw_path.exists() or summary_path.exists():
        if not args.resume or not metadata_path.exists() or not raw_path.exists():
            raise FileExistsError(f"existing output in {output}; use --resume or --force")
        meta = read_metadata(metadata_path)
        for key, expected in expected_identity.items():
            if meta.get(key) != expected:
                raise ValueError(f"resume mismatch in {key}")
        rows = read_csv(raw_path, REPLICATION_FIELDS, allow_empty=True)
        validate_rows(rows, config, complete=False, phi3_sd=phi3_sd)
    else:
        meta = dict(expected_identity)
        meta.update(status="running", run_complete="false")
        write_metadata(metadata_path, meta)
        rows = []
        atomic_csv(raw_path, REPLICATION_FIELDS, rows)

    seen = {(row["setup"], float(row["signal"]), int(row["rep"])) for row in rows}
    checkpoint_count = 0
    try:
        for setup, bases in SETUPS.items():
            for signal in config.signals:
                for rep in range(config.R):
                    key = (setup, signal, rep)
                    if key in seen:
                        continue
                    row = core.run_replication(
                        bases, signal, rep, n=config.n, B=config.B,
                        alpha=config.alpha, theta0=config.theta0, sigma=config.sigma,
                        base_seed=config.seed_for(setup), phi3_sd=phi3_sd,
                    )
                    if row["setup"] != setup or float(row["signal"]) != signal or int(row["rep"]) != rep:
                        raise ValueError(f"core returned wrong replication identity at {key}")
                    if set(row) != set(REPLICATION_FIELDS):
                        raise ValueError("core returned an unexpected replication schema")
                    rows.append(row)
                    seen.add(key)
                    checkpoint_count += 1
                    if checkpoint_count >= 50:
                        atomic_csv(raw_path, REPLICATION_FIELDS, rows)
                        checkpoint_count = 0
                atomic_csv(raw_path, REPLICATION_FIELDS, rows)
                checkpoint_count = 0
                atomic_csv(summary_path, SUMMARY_FIELDS, summarize(rows, config, complete=False))
                print(f"{setup}, t={signal:g}: {config.R} paired datasets complete", flush=True)
    finally:
        if checkpoint_count:
            atomic_csv(raw_path, REPLICATION_FIELDS, rows)

    validate_rows(read_csv(raw_path, REPLICATION_FIELDS), config,
                  complete=True, phi3_sd=phi3_sd)
    atomic_csv(summary_path, SUMMARY_FIELDS, summarize(rows, config, complete=True))
    meta.update(status="complete", run_complete="true",
                replications_csv_rows=str(len(rows)),
                summary_csv_rows=str(len(SETUPS) * len(config.signals) * len(METHODS)),
                replications_sha256=sha256(raw_path), summary_sha256=sha256(summary_path))
    write_metadata(metadata_path, meta)
    if config.plot:
        run_plot(config, meta)
    print(f"Saved {len(rows)} paired datasets in {output}", flush=True)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return execute(args)
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
