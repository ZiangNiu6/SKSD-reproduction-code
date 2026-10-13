"""Reproduce the appendix Watson goodness-of-fit experiment on the sphere.

Run ``python 05_sphere_watson.py --check`` for a small end-to-end check.
Publication defaults deliberately require a separate full invocation; smoke
results are not evidence of size control or power. See ../../README.md for formulas and provenance.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from itertools import product, repeat
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

# Small dense matrices are often slower with heavily threaded BLAS. Respect
# explicit settings while avoiding oversubscription on default installations.
for _variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_variable, "1")

import numpy as np

HERE = Path(__file__).resolve().parent.parent
CORE_DIR = HERE / "numerical-helpers"
PLOT_DIR = HERE / "figure-code"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from _sphere_core import (
    adjusted_gram, fit_watson, sample_bingham, sample_crossing,
    sample_watson, stein_gram,
)
from _sphere_brueck import brueck_statistics

STUDY = "05-sphere-watson"
SCHEMA_VERSION = "sphere-watson-fresh-csv-v2"
SAVED_SCHEMA_VERSION = "sphere-watson-saved-csv-v2"
DEFAULT_OUTPUT_DIR = HERE.parent / "results" / "simulation-analysis" / STUDY
SOURCE_FILES = ("05_sphere_watson.py", "_sphere_core.py", "_sphere_brueck.py")
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
DESIGN_VERSION = "sphere-watson-bingham-v3-fixed-axis"
FIXED_AXIS = np.array([0.0, 0.0, 1.0])
PARAMETRIC_CHUNK_SIZE = 8
BRUECK_CORRECTION_VERSION = "factor-corrected-midpoint-v1"
METHODS = {"influence": "SKSD-influence", "parametric": "SKSD-parametric",
           "brueck": "Brueck-corrected"}
RAW_FIELDS = (
    "family", "dispersion_path", "sample_size", "kappa0", "signal", "rep",
    "method", "statistic_type", "statistic", "pvalue", "reject", "fitted_kappa", "axis_x",
    "axis_y", "axis_z", "scatter_gap", "influence_max_norm", "sample_seconds",
    "shared_seconds", "calibration_seconds", "total_seconds",
)
SUMMARY_FIELDS = (
    "family", "dispersion_path", "sample_size", "kappa0", "signal", "method", "statistic_type",
    "reps", "rejections", "rejection_rate", "mc_se", "ci_low", "ci_high",
    "mean_statistic", "mean_fitted_kappa", "mean_shared_seconds",
    "mean_calibration_seconds", "mean_total_seconds",
)


@dataclass(frozen=True)
class Config:
    sample_sizes: tuple[int, ...]
    kappas: tuple[float, ...]
    signals: tuple[float, ...]
    dispersion_paths: tuple[str, ...]
    crossing: bool
    crossing_signals: tuple[float, ...]
    angle_degrees: float
    reps: int
    bootstrap_draws: int
    alpha: float
    seed: int
    bandwidth: float
    methods: tuple[str, ...]
    weights: str
    workers: int = 1


@dataclass(frozen=True)
class Cell:
    family: str
    dispersion_path: str
    sample_size: int
    kappa0: float
    signal: float


def parse_grid(value, cast=float):
    parts = value.split(",")
    if not parts or any(not part.strip() for part in parts):
        raise ValueError("grids must be nonempty comma-separated values")
    result = tuple(cast(part.strip()) for part in parts)
    if len(result) != len(set(result)):
        raise ValueError("grid values must be distinct")
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-sizes", default=None)
    parser.add_argument("--kappas", default=None)
    parser.add_argument("--signals", default=None)
    parser.add_argument("--dispersion-paths", default="fixed_kappa",
                        help="fixed_kappa, matched_odi, or both comma-separated")
    parser.add_argument("--crossing", action="store_true", help="also simulate two-Watson mixtures")
    parser.add_argument("--crossing-signals", default="0,0.05,0.1,0.2,0.3,0.5")
    parser.add_argument("--angle-degrees", type=float, default=60.0)
    parser.add_argument("--reps", type=int, default=None)
    parser.add_argument("--bootstrap-draws", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--bandwidth", type=float, default=1.0)
    parser.add_argument("--methods", default="influence,parametric,brueck",
                        help="comma-separated subset of influence,parametric,brueck")
    parser.add_argument("--weights", choices=("efron", "gaussian", "rademacher"), default="gaussian",
                        help="influence-bootstrap weights; Brueck always uses Efron resampling")
    parser.add_argument("--workers", type=int, default=1, help="parallel independent replications; statistics are worker-count invariant")
    parser.add_argument("--output-dir", type=Path,
                        help=f"result directory (default: {DEFAULT_OUTPUT_DIR}; --check uses simulation-analysis-check/)")
    parser.add_argument("--resume", action="store_true", help="resume matching complete-cell checkpoints")
    parser.add_argument("--force", action="store_true", help="replace only this study's named output files")
    parser.add_argument("--check", dest="quick", action="store_true", help="n=100, kappa=4,16, r=0,.3,.75, R=3, B=19; explicit overrides are respected")
    parser.add_argument("--quick", dest="quick", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--plot-only", action="store_true", help="validate and plot existing CSVs without simulation")
    return parser


def resolve_output_dir(args):
    default = DEFAULT_OUTPUT_DIR.parent.parent / (
        "simulation-analysis-check" if args.quick else "simulation-analysis"
    ) / STUDY
    return (args.output_dir or default).expanduser().resolve()


def config_from_args(args):
    cfg = Config(
        parse_grid(args.sample_sizes or "100", int),
        parse_grid(args.kappas or "4,16"),
        parse_grid(args.signals or ("0,0.3,0.75" if args.quick else "0,0.15,0.3,0.45,0.6,0.75")),
        parse_grid(args.dispersion_paths, str), args.crossing, parse_grid(args.crossing_signals),
        args.angle_degrees, args.reps if args.reps is not None else (3 if args.quick else 1000),
        args.bootstrap_draws if args.bootstrap_draws is not None else (19 if args.quick else 499),
        args.alpha, args.seed, args.bandwidth,
        tuple(METHODS[m] for m in parse_grid(args.methods, str)), args.weights, args.workers,
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if min(cfg.sample_sizes) < 8:
        raise ValueError("sample sizes must be at least 8")
    numeric = (*cfg.kappas, *cfg.signals, *cfg.crossing_signals, cfg.angle_degrees,
               cfg.alpha, cfg.bandwidth)
    if not all(math.isfinite(x) for x in numeric):
        raise ValueError("all numeric settings must be finite")
    if min(cfg.kappas) <= 0 or cfg.bandwidth <= 0:
        raise ValueError("kappa and bandwidth must be strictly positive")
    if any(not 0 <= r < 1 for r in cfg.signals):
        raise ValueError("Bingham signals must lie in [0,1)")
    if any(not 0 <= e <= 0.5 for e in cfg.crossing_signals):
        raise ValueError("crossing fractions must lie in [0,0.5]")
    if set(cfg.dispersion_paths) - {"fixed_kappa", "matched_odi"}:
        raise ValueError("dispersion paths must be fixed_kappa or matched_odi")
    if not 0 < cfg.angle_degrees <= 90 or not 0 < cfg.alpha < 1:
        raise ValueError("require angle in (0,90] and alpha in (0,1)")
    if cfg.reps < 1 or cfg.bootstrap_draws < 1 or cfg.seed < 0:
        raise ValueError("reps and bootstrap draws must be positive; seed must be nonnegative")
    if not isinstance(cfg.workers, int) or isinstance(cfg.workers, bool) or cfg.workers < 1:
        raise ValueError("workers must be a positive integer")
    if not cfg.methods or len(set(cfg.methods)) != len(cfg.methods) or set(cfg.methods) - set(METHODS.values()):
        raise ValueError("methods must be a nonempty subset with no duplicates")


def cells(cfg):
    result = [Cell("bingham", path, n, k, r) for path, n, k, r in product(
        cfg.dispersion_paths, cfg.sample_sizes, cfg.kappas, cfg.signals)]
    if cfg.crossing:
        result += [Cell("crossing", "fixed_kappa", n, k, e) for n, k, e in product(
            cfg.sample_sizes, cfg.kappas, cfg.crossing_signals)]
    return result


def rng_for(cfg, cell, rep, stream):
    # Null datasets and bootstrap random streams coincide across optional paths.
    # Grid order, method selection and bootstrap size never change observed data.
    identity = ("null", cell.sample_size, cell.kappa0) if cell.signal == 0 else (
        cell.family, cell.dispersion_path, cell.sample_size, cell.kappa0, cell.signal,
        cfg.angle_degrees if cell.family == "crossing" else None)
    digest = hashlib.sha256(json.dumps((identity, rep, stream)).encode()).digest()
    entropy = [cfg.seed, *np.frombuffer(digest, dtype="<u4").tolist()]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def center_gram(gram):
    return gram - gram.mean(axis=0)[None, :] - gram.mean(axis=1)[:, None] + gram.mean()


def nonnegative(values):
    values = np.asarray(values)
    if not np.all(np.isfinite(values)) or np.min(values) < -1e-8:
        raise FloatingPointError("nonfinite or materially negative squared RKHS norm")
    return np.maximum(values, 0.0)


def calibrate_influence(x, fit, cfg, rng):
    gram = center_gram(adjusted_gram(x, fit, cfg.bandwidth))
    n, b = len(x), cfg.bootstrap_draws
    if cfg.weights == "efron":
        weights = rng.multinomial(n, np.full(n, 1.0 / n), size=b) - 1.0
    elif cfg.weights == "gaussian":
        weights = rng.standard_normal((b, n))
        weights -= weights.mean(axis=1, keepdims=True)
    else:
        weights = 2.0 * rng.integers(0, 2, size=(b, n)) - 1.0
    return nonnegative(np.einsum("bi,ij,bj->b", weights, gram, weights, optimize=True) / n)


def fixed_axis_statistics(samples, axis, bandwidth, chunk_size=PARAMETRIC_CHUNK_SIZE):
    """Refit scalar concentration separately in each sample, then return nV.

    This is the batched equivalent of ``fit_watson(x, fixed_axis=axis)``
    followed by ``len(x) * stein_gram(x, fit, bandwidth).mean()``. Chunking
    limits temporary Gram matrices without sharing fits between samples.
    """
    samples = np.asarray(samples, dtype=float)
    axis = np.asarray(axis, dtype=float)
    if samples.ndim != 3 or samples.shape[1] == 0 or samples.shape[2] != 3:
        raise ValueError("samples must have shape (bootstrap_draws, n, 3), with n positive")
    if not np.all(np.isfinite(samples)) or not np.allclose(
            np.sum(samples * samples, axis=2), 1.0, rtol=0, atol=1e-10):
        raise ValueError("every bootstrap observation must lie on the unit sphere")
    if axis.shape != (3,) or not np.all(np.isfinite(axis)) or not np.isclose(
            axis @ axis, 1.0, rtol=0, atol=1e-10):
        raise ValueError("axis must be a finite unit vector of length three")
    if not math.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError("bandwidth must be finite and positive")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    inverse_h2 = (1.0 / bandwidth) ** 2
    if not math.isfinite(inverse_h2):
        raise ValueError("the inverse squared bandwidth must be finite")
    result = np.empty(len(samples))
    for start in range(0, len(samples), chunk_size):
        x = samples[start:start + chunk_size]
        axial = x @ axis
        u = axial * axial
        denominator = 2.0 * np.mean(u * (1.0 - u), axis=1)
        if np.any(denominator <= 0.0) or not np.all(np.isfinite(denominator)):
            raise ValueError("the score-matching concentration denominator is not positive")
        kappa = (3.0 * u.mean(axis=1) - 1.0) / denominator
        dot = x @ x.swapaxes(1, 2)
        kernel = np.exp((dot - 1.0) * inverse_h2)
        projected_axis = axis - axial[:, :, None] * x
        b = 2.0 * kappa[:, None, None] * axial[:, :, None] * projected_axis - 2.0 * x
        b_dot_x = b @ x.swapaxes(1, 2)
        b_dot_self = np.sum(b * x, axis=2)
        score_gradient = (b_dot_self[:, :, None] + b_dot_self[:, None, :]
                          - dot * (b_dot_x + b_dot_x.swapaxes(1, 2)))
        mixed_gradient = (inverse_h2 * (1.0 + dot * dot)
                          - inverse_h2**2 * dot * (1.0 - dot * dot))
        gram = kernel * (b @ b.swapaxes(1, 2) + inverse_h2 * score_gradient + mixed_gradient)
        result[start:start + len(x)] = gram.sum(axis=(1, 2)) / samples.shape[1]
    return nonnegative(result)



def calibrate_parametric(x, fit, cfg, rng):
    if not fit.fixed_axis or not np.array_equal(fit.mu, FIXED_AXIS):
        raise ValueError("this experiment requires the prespecified axis e3")
    # The exact draws are iid. Reshaping assigns disjoint samples to B
    # bootstrap replications; every concentration is separately refitted.
    samples = sample_watson(rng, cfg.bootstrap_draws * len(x), fit.mu, fit.kappa)
    samples = samples.reshape(cfg.bootstrap_draws, len(x), 3)
    return fixed_axis_statistics(samples, FIXED_AXIS, cfg.bandwidth)


def calibrate_brueck(x, fit, cfg, rng):
    """Repository midpoint correction of the refitted empirical U-bootstrap."""
    counts = rng.multinomial(len(x), np.full(len(x), 1.0 / len(x)), size=cfg.bootstrap_draws)
    return brueck_statistics(x, fit, counts, bandwidth=cfg.bandwidth)


def run_replication(cfg, cell, rep):
    start = time.perf_counter()
    rng = rng_for(cfg, cell, rep, "data")
    if cell.signal == 0:
        x = sample_bingham(rng, cell.sample_size, cell.kappa0, 0.0)
    elif cell.family == "bingham":
        x = sample_bingham(rng, cell.sample_size, cell.kappa0, cell.signal,
                           matched_odi=cell.dispersion_path == "matched_odi")
    else:
        x = sample_crossing(rng, cell.sample_size, cell.kappa0, cell.signal, cfg.angle_degrees)
    sample_seconds = time.perf_counter() - start
    start = time.perf_counter()
    fit = fit_watson(x, fixed_axis=FIXED_AXIS)
    statistic = float(nonnegative(len(x) * stein_gram(x, fit, cfg.bandwidth).mean()))
    shared_seconds = time.perf_counter() - start
    evals = np.linalg.eigvalsh(x.T @ x / len(x))
    common = dict(asdict(cell), rep=rep, statistic=statistic, fitted_kappa=fit.kappa,
                  axis_x=fit.mu[0], axis_y=fit.mu[1], axis_z=fit.mu[2],
                  scatter_gap=evals[-1] - evals[-2],
                  influence_max_norm=np.linalg.norm(fit.influence, axis=1).max(),
                  sample_seconds=sample_seconds, shared_seconds=shared_seconds)
    rows = []
    for method in cfg.methods:
        start = time.perf_counter()
        # Recreating the same RNG pairs the two empirical methods' Efron
        # counts, while retaining the earlier two methods' random streams.
        stream = METHODS["influence"] if method == METHODS["brueck"] else method
        method_rng = rng_for(cfg, cell, rep, stream)
        observed, statistic_type = statistic, "nV"
        if method == METHODS["influence"]:
            bootstrap = calibrate_influence(x, fit, cfg, method_rng)
        elif method == METHODS["parametric"]:
            bootstrap = calibrate_parametric(x, fit, cfg, method_rng)
        elif method == METHODS["brueck"]:
            observed, bootstrap = calibrate_brueck(x, fit, cfg, method_rng)
            statistic_type = "nU"
        else:
            raise ValueError(f"Unknown method: {method}")
        pvalue = (1 + np.count_nonzero(bootstrap >= observed)) / (cfg.bootstrap_draws + 1)
        elapsed = time.perf_counter() - start
        rows.append(dict(common, method=method, statistic_type=statistic_type, statistic=float(observed),
                         pvalue=pvalue, reject=int(pvalue <= cfg.alpha),
                         calibration_seconds=elapsed, total_seconds=shared_seconds + elapsed))
    return rows


def wilson_interval(successes, total):
    p = successes / total
    z = 1.959963984540054
    den = 1 + z * z / total
    center = (p + z * z / (2 * total)) / den
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / den
    return min(p, max(0.0, center - radius)), max(p, min(1.0, center + radius))


def summarize(rows):
    groups = {}
    keys = ("family", "dispersion_path", "sample_size", "kappa0", "signal", "method")
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    summaries = []
    for key, group in groups.items():
        total = len(group)
        count = sum(int(r["reject"]) for r in group)
        rate = count / total
        low, high = wilson_interval(count, total)
        summary = dict(zip(keys, key), reps=total, rejections=count, rejection_rate=rate,
                       statistic_type=group[0]["statistic_type"],
                       mc_se=math.sqrt(rate * (1 - rate) / total), ci_low=low, ci_high=high)
        for source in ("statistic", "fitted_kappa", "shared_seconds", "calibration_seconds", "total_seconds"):
            summary["mean_" + source] = float(np.mean([float(r[source]) for r in group]))
        summaries.append(summary)
    return summaries


def atomic_csv(path, fields, rows):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def cell_key(row):
    return (row["family"], row["dispersion_path"], int(row["sample_size"]),
            float(row["kappa0"]), float(row["signal"]))


def read_checkpoint(path, cfg, planned):
    if not path.exists():
        return [], set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != RAW_FIELDS:
            raise ValueError("checkpoint CSV schema does not match")
        rows = list(reader)
    expected_cells = {cell_key(asdict(cell)) for cell in planned}
    expected_pairs = set(product(range(cfg.reps), cfg.methods))
    coverage = {}
    strings = {"family", "dispersion_path", "method", "statistic_type"}
    integers = {"sample_size", "rep", "reject"}
    for row in rows:
        for field in RAW_FIELDS:
            if field not in strings:
                row[field] = int(row[field]) if field in integers else float(row[field])
                if not math.isfinite(row[field]):
                    raise ValueError("nonfinite checkpoint value")
        if not np.array_equal([row["axis_x"], row["axis_y"], row["axis_z"]], FIXED_AXIS):
            raise ValueError("checkpoint axis must equal the prespecified e3")
        key = cell_key(row)
        if key not in expected_cells:
            raise ValueError("checkpoint contains an unexpected design cell")
        pair = (row["rep"], row["method"])
        pairs = coverage.setdefault(key, set())
        if pair in pairs:
            raise ValueError("checkpoint contains duplicate replication/method rows")
        pairs.add(pair)
        expected_statistic = "nU" if row["method"] == METHODS["brueck"] else "nV"
        if row["statistic_type"] != expected_statistic:
            raise ValueError("checkpoint statistic type does not match its method")
        if not 0 < row["pvalue"] <= 1 or row["reject"] != int(row["pvalue"] <= cfg.alpha):
            raise ValueError("checkpoint has invalid p-value or rejection indicator")
        grid_position = row["pvalue"] * (cfg.bootstrap_draws + 1)
        if not math.isclose(grid_position, round(grid_position), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("checkpoint p-value is not on the recorded bootstrap grid")
        if any(row[field] < 0 for field in RAW_FIELDS if field.endswith("_seconds")):
            raise ValueError("checkpoint contains a negative runtime")
        if not math.isclose(row["total_seconds"], row["shared_seconds"] + row["calibration_seconds"],
                            rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("checkpoint total runtime does not equal shared plus calibration time")
        if expected_statistic == "nV" and row["statistic"] < 0:
            raise ValueError("checkpoint has a negative nV statistic")
    if any(pairs != expected_pairs for pairs in coverage.values()):
        raise ValueError("checkpoint contains an incomplete cell")
    return rows, set(coverage)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_metadata(path, metadata):
    """Write generated key/value provenance without a JSON sidecar."""
    def render(value):
        if isinstance(value, bool):
            return str(value).lower()
        return str(value)
    atomic_csv(Path(path), ("key", "value"),
               ({"key": key, "value": render(value)} for key, value in metadata.items()))


def read_metadata(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != ("key", "value"):
            raise ValueError("metadata must have exactly key,value columns")
        rows = list(reader)
    result = {row["key"]: row["value"] for row in rows}
    if len(result) != len(rows) or any(not key for key in result):
        raise ValueError("metadata contains duplicate or blank keys")
    return result


def config_metadata(cfg):
    """Stable, human-readable configuration fields for a generated CSV."""
    result = {}
    for key, value in asdict(cfg).items():
        if isinstance(value, tuple):
            value = ",".join(map(str, value))
        elif isinstance(value, bool):
            value = str(value).lower()
        result["config_" + key] = str(value)
    return result


def config_from_metadata(metadata):
    tuple_fields = {"sample_sizes": int, "kappas": float, "signals": float,
                    "dispersion_paths": str, "crossing_signals": float, "methods": str}
    integer_fields = {"reps", "bootstrap_draws", "seed", "workers"}
    float_fields = {"angle_degrees", "alpha", "bandwidth"}
    values = {}
    for field in Config.__dataclass_fields__:
        value = metadata["config_" + field]
        if field in tuple_fields:
            value = parse_grid(value, tuple_fields[field])
        elif field in integer_fields:
            value = int(value)
        elif field in float_fields:
            value = float(value)
        elif field == "crossing":
            if value not in {"true", "false"}:
                raise ValueError("config_crossing must be true or false")
            value = value == "true"
        values[field] = value
    cfg = Config(**values)
    validate_config(cfg)
    return cfg


def source_hashes():
    return {
        name: sha256((HERE / "study-scripts" if name == "05_sphere_watson.py" else CORE_DIR) / name)
        for name in SOURCE_FILES
    }


def run_fingerprint(cfg, hashes):
    return hashlib.sha256(json.dumps(
        (DESIGN_VERSION, asdict(cfg), hashes), sort_keys=True).encode()).hexdigest()[:16]


def runtime_metadata():
    return {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
            **{"thread_" + name: os.environ.get(name, "unset") for name in THREAD_VARIABLES}}


def build_metadata(cfg):
    hashes = source_hashes()
    return {
        "study": STUDY, "schema_version": SCHEMA_VERSION, "workflow": "simulate",
        "design_version": DESIGN_VERSION, "status": "running", "run_complete": "false",
        "run_fingerprint": run_fingerprint(cfg, hashes), **config_metadata(cfg),
        **{"source_sha256_" + name: digest for name, digest in hashes.items()},
        "started_at": datetime.now(timezone.utc).isoformat(), **runtime_metadata(),
        "statistic": "nV for SKSD-influence/parametric; signed nU for Brueck-corrected",
        "brueck_correction_version": BRUECK_CORRECTION_VERSION,
        "brueck_correction_note": "project midpoint correction, not an author-confirmed erratum",
        "brueck_parameterization": "A=kappa*(e3*e3.T-I/3); every refit keeps the axis fixed at e3",
        "brueck_weights": "Efron counts; shared with influence when weights=efron",
        "comparison_note": "Brueck uses U rather than V; not a calibration-only comparison",
        "kernel": "exp(-||x-y||^2/(2 h^2)); h fixed for fitting and every bootstrap",
        "null": "Watson on S2 with fixed axis e3 and concentration kappa in R",
        "estimator": "fixed-axis moment ratio, equivalent to scalar spherical score matching",
        "fixed_axis": "0,0,1",
        "interpretation": "iid spherical orientations; not MRI signals or hemisphere-coded axial data",
        "parallel_workers": cfg.workers,
        "timing_scope": "standalone-test-v1",
        "timing_description": "shared observed fitting/statistic charged in full to each method plus calibration; DGP sampling, CSV and plotting excluded",
        "checkpoint_scope": "whole design cells; an interrupted unsaved cell is recomputed",
        "planned_cells": str(len(cells(cfg))), "completed_cells": "0",
    }


def validate_saved_results(output):
    """Validate complete results and all summaries before any figure mutation.

    Saved historical CSVs use a distinct schema and retain original source
    metadata. They can be plotted without loading the original producer.
    """
    output = Path(output)
    metadata = read_metadata(output / "run_metadata.csv")
    expected_workflow = {SCHEMA_VERSION: "simulate", SAVED_SCHEMA_VERSION: "saved-results"}
    if (metadata.get("schema_version") not in expected_workflow
            or metadata.get("workflow") != expected_workflow[metadata["schema_version"]]
            or metadata.get("study") != STUDY
            or metadata.get("design_version") != DESIGN_VERSION
            or metadata.get("status") != "complete" or metadata.get("run_complete") != "true"
            or not metadata.get("run_fingerprint")):
        raise ValueError("sphere metadata does not describe a complete supported result")
    cfg = config_from_metadata(metadata)
    planned = cells(cfg)
    if (int(metadata["planned_cells"]) != len(planned)
            or int(metadata["completed_cells"]) != len(planned)):
        raise ValueError("sphere metadata cell counts do not match the configured design")
    names = ("replications", "summary")
    if metadata["schema_version"] == SAVED_SCHEMA_VERSION:
        names += ("source_metadata",)
    for name in names:
        path = output / (name + ".csv")
        if metadata.get(name + "_sha256") != sha256(path):
            raise ValueError(f"saved result checksum mismatch: {path}")
    rows, completed = read_checkpoint(output / "replications.csv", cfg, planned)
    if completed != {cell_key(asdict(cell)) for cell in planned}:
        raise ValueError("sphere results do not cover every configured cell")
    with (output / "summary.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SUMMARY_FIELDS:
            raise ValueError("sphere summary has the wrong CSV schema")
        summaries = list(reader)
    calculated = summarize(rows)
    key = lambda row: (*cell_key(row), row["method"])
    stored = {key(row): row for row in summaries}
    if len(stored) != len(summaries) or set(stored) != {key(row) for row in calculated}:
        raise ValueError("sphere summary has duplicate or unexpected cells/methods")
    categorical = {"family", "dispersion_path", "method", "statistic_type"}
    for expected in calculated:
        actual = stored[key(expected)]
        for field in SUMMARY_FIELDS:
            if field in categorical:
                valid = actual[field] == expected[field]
            else:
                value = float(actual[field])
                valid = math.isfinite(value) and math.isclose(
                    value, float(expected[field]), rel_tol=1e-12, abs_tol=1e-12)
            if not valid:
                raise ValueError(f"sphere summary disagrees with replications: {field}")
    return summaries, metadata


def plot_results(output):
    """Draw only from validated tables; preserve numerical CSVs byte for byte."""
    output = Path(output)
    _, metadata = validate_saved_results(output)
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript is required to plot the saved sphere results")
    subprocess.run([rscript, str(PLOT_DIR / "plot_05_sphere_watson.R"),
                    str(output / "summary.csv"), str(output / "figure.pdf"),
                    str(output / "figure.png"), metadata["config_alpha"]], check=True)
    for suffix in ("pdf", "png"):
        path = output / ("figure." + suffix)
        header = path.read_bytes()[:8]
        if ((suffix == "pdf" and not header.startswith(b"%PDF-"))
                or (suffix == "png" and header != b"\x89PNG\r\n\x1a\n")):
            raise ValueError(f"plotting did not produce a valid {suffix.upper()}: {path}")
        metadata["figure_" + suffix + "_sha256"] = sha256(path)
    for filename in ("plot_05_sphere_watson.R", "_paper_plot.R", "_paper_plot_style.csv"):
        metadata["plot_source_sha256_" + filename] = sha256(PLOT_DIR / filename)
    metadata["plotted_at"] = datetime.now(timezone.utc).isoformat()
    write_metadata(output / "run_metadata.csv", metadata)


def simulate(cfg, output, *, resume=False, force=False):
    """Complete-cell checkpoints retain the producer's original random streams."""
    output = Path(output)
    if resume and force:
        raise ValueError("choose either --resume or --force")
    planned = cells(cfg)
    metadata = build_metadata(cfg)
    metadata_path = output / "run_metadata.csv"
    if output.exists() and any(output.iterdir()):
        if force:
            # Do not recursively delete an output folder or unrelated files.
            for name in ("replications.csv", "summary.csv", "run_metadata.csv", "figure.pdf", "figure.png", "source_metadata.csv"):
                path = output / name
                if path.is_file():
                    path.unlink()
        elif not resume:
            raise FileExistsError(f"Output exists: {output}. Use a fresh directory, --resume, or --force.")
        else:
            previous = read_metadata(metadata_path)
            if (previous.get("schema_version") != SCHEMA_VERSION
                    or previous.get("workflow") != "simulate"
                    or previous.get("run_fingerprint") != metadata["run_fingerprint"]):
                raise ValueError("resume requires identical configuration and simulation sources")
            if any(previous.get(key) != value for key, value in runtime_metadata().items()):
                raise ValueError("resume requires the recorded Python/NumPy/platform/thread environment")
            if previous.get("status") == "complete":
                validate_saved_results(output)
                return
            metadata["started_at"] = previous["started_at"]
    output.mkdir(parents=True, exist_ok=True)
    rows, completed = read_checkpoint(output / "replications.csv", cfg, planned) if resume else ([], set())
    metadata["completed_cells"] = str(len(completed))
    write_metadata(metadata_path, metadata)
    print(f"{len(planned)} cells x {cfg.reps} datasets; B={cfg.bootstrap_draws}; output={output}", flush=True)
    if cfg.alpha < 1 / (cfg.bootstrap_draws + 1):
        print("Note: this bootstrap count cannot produce a p-value below the nominal level.", flush=True)
    try:
        pool_context = ProcessPoolExecutor(max_workers=cfg.workers) if cfg.workers > 1 else nullcontext()
        with pool_context as pool:
            for index, cell in enumerate(planned, 1):
                key = cell_key(asdict(cell))
                if key in completed:
                    continue
                print(f"Cell {index}/{len(planned)}: {asdict(cell)}", flush=True)
                pending = []
                last_progress = time.monotonic()
                results = (pool.map(run_replication, repeat(cfg), repeat(cell), range(cfg.reps), chunksize=1)
                           if pool is not None else
                           (run_replication(cfg, cell, rep) for rep in range(cfg.reps)))
                for rep, result in enumerate(results):
                    pending.extend(result)
                    if time.monotonic() - last_progress >= 30:
                        print(f"  replication {rep + 1}/{cfg.reps}", flush=True)
                        last_progress = time.monotonic()
                rows.extend(pending)
                completed.add(key)
                atomic_csv(output / "replications.csv", RAW_FIELDS, rows)
                atomic_csv(output / "summary.csv", SUMMARY_FIELDS, summarize(rows))
                metadata["completed_cells"] = str(len(completed))
                write_metadata(metadata_path, metadata)
        atomic_csv(output / "summary.csv", SUMMARY_FIELDS, summarize(rows))
        metadata.update(status="complete", run_complete="true", finished_at=datetime.now(timezone.utc).isoformat(),
                        replications_sha256=sha256(output / "replications.csv"), summary_sha256=sha256(output / "summary.csv"))
        write_metadata(metadata_path, metadata)
        print("Numerical simulation complete.", flush=True)
    except BaseException as error:
        metadata["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        metadata["error"] = f"{type(error).__name__}: {error}"
        write_metadata(metadata_path, metadata)
        raise


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    output = resolve_output_dir(args)
    if args.plot_only:
        plot_results(output)
        return 0
    try:
        cfg = config_from_args(args)
    except (ValueError, KeyError) as error:
        parser.error(str(error))
    simulate(cfg, output, resume=args.resume, force=args.force)
    plot_results(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
