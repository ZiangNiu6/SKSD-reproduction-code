#!/usr/bin/env python3
"""Reproduce the Gaussianity experiment with maximum-likelihood fitting.

This is a transparent, single-experiment entry point.  By default it runs the
publication-scale Monte Carlo design from scratch.  ``--check`` selects a small
deterministic smoke design.  ``--plot-only`` validates completed simulation
tables and recreates their figures without running any simulation.

Fresh simulations compare six procedures on the same observed dataset:

* SKSD with a fitted-model parametric bootstrap;
* SKSD with the proposed influence-adjusted Gaussian multiplier calibration;
* SKSD with a naive wild Gaussian multiplier bootstrap (no adjustment);
* the project's factor-corrected midpoint empirical U-bootstrap diagnostic;
* semiparametric MMD with a fitted-model parametric bootstrap; and
* semiparametric KS (the Lilliefors setting) with a fitted-model bootstrap.

Every run writes ``replications.csv``, ``summary.csv``, and
``run_metadata.csv``.  Unless ``--no-plot`` is selected, R/ggplot2 renders
``figure.pdf`` and ``figure.png``.  No JSON configuration or output is used.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version as package_version
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Iterable

try:
    import numpy as np
    from scipy.stats import kstest
except ImportError as exc:  # pragma: no cover - exercised only in a bad env
    raise SystemExit(
        "01_gaussianity_mle.py requires NumPy and SciPy. Install the "
        "reproduction environment before running this script."
    ) from exc


SCRIPT_PATH = Path(__file__).resolve()
REPRODUCTION_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_OUTPUT_DIR = (
    REPRODUCTION_ROOT / "results" / "simulation-analysis" / "01-gaussianity-mle"
)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    default_dir = DEFAULT_OUTPUT_DIR.parent.parent / (
        "simulation-analysis-check" if args.quick else "simulation-analysis"
    ) / DEFAULT_OUTPUT_DIR.name
    return (args.output_dir or default_dir).expanduser().resolve()

PUBLICATION_N = 100
PUBLICATION_BOOTSTRAPS = 300
PUBLICATION_REPLICATIONS = 300
PUBLICATION_SEED = 20260622
PUBLICATION_ALPHA = 0.05
SCHEMA_VERSION = 1


def installed_version(distribution: str) -> str:
    """Return a dependency version without importing an optional plot module."""
    try:
        return package_version(distribution)
    except PackageNotFoundError:
        return "not installed"


PUBLICATION_GRIDS = {
    "t": tuple(i / 8.0 for i in range(9)),
    "mixture": (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75),
    "gen_chi": (1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75),
}
QUICK_GRIDS = {
    "t": (0.0, 0.5),
    "mixture": (0.0, 1.25),
    "gen_chi": (1.0, 1.5),
}
ALTERNATIVE_LABELS = {
    "t": "Student-t",
    "mixture": "Gaussian mixture",
    "gen_chi": "Generalized chi-square",
}
ALTERNATIVE_CODES = {"t": 1, "mixture": 2, "gen_chi": 3}

METHODS = (
    "SKSD-parametric",
    "SKSD-influence",
    "SKS",
    "SMMD",
    "Brueck-corrected",
    "SKSD-naive-wild",
)
SAVED_PUBLICATION_METHODS = tuple(
    method for method in METHODS if method != "SKSD-naive-wild"
)
METHOD_LABELS = {
    "SKSD-parametric": "SKSD-parametric",
    "SKSD-influence": "SKSD-influence",
    "SKS": "SKS (Lilliefors)",
    "SMMD": "SMMD (Key et al.)",
    "Brueck-corrected": "Br\u00fcck et al.",
    "SKSD-naive-wild": "SKSD-naive wild",
}
METHOD_TESTS = {
    "SKSD-parametric": "KSD",
    "SKSD-influence": "KSD",
    "SKS": "KS",
    "SMMD": "MMD",
    "Brueck-corrected": "KSD",
    "SKSD-naive-wild": "KSD",
}
METHOD_CALIBRATIONS = {
    "SKSD-parametric": "parametric bootstrap",
    "SKSD-influence": "influence-adjusted Gaussian multiplier bootstrap",
    "SKS": "parametric bootstrap",
    "SMMD": "parametric bootstrap",
    "Brueck-corrected": "factor-corrected midpoint Efron diagnostic",
    "SKSD-naive-wild": "naive wild Gaussian multiplier bootstrap (no adjustment)",
}
FRESH_TIMING_PROTOCOL = "standalone-test-v1"

REPLICATION_FIELDS = (
    "alternative",
    "signal",
    "replication",
    "method",
    "method_label",
    "test",
    "calibration",
    "p_value",
    "reject",
    "seconds",
    "source_record",
    "aggregate_rejection_rate",
)
SUMMARY_FIELDS = (
    "alternative",
    "alternative_label",
    "signal",
    "method",
    "method_label",
    "test",
    "calibration",
    "replications",
    "rejections",
    "rejection_rate",
    "mcse",
    "mean_seconds",
    "source",
)
OUTPUT_FILENAMES = (
    "replications.csv",
    "summary.csv",
    "run_metadata.csv",
    "figure.pdf",
    "figure.png",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(parts: Iterable[int]) -> int:
    return int(np.random.SeedSequence(list(parts)).generate_state(1)[0])


def parse_grid(value: str, option: str) -> tuple[float, ...]:
    try:
        grid = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"{option} must be a comma-separated numeric grid"
        ) from exc
    if not grid:
        raise argparse.ArgumentTypeError(f"{option} must not be empty")
    if len(set(grid)) != len(grid):
        raise argparse.ArgumentTypeError(f"{option} contains duplicate values")
    return grid


def parse_alternatives(value: str) -> tuple[str, ...]:
    alternatives = tuple(item.strip() for item in value.split(",") if item.strip())
    unknown = sorted(set(alternatives) - set(PUBLICATION_GRIDS))
    if not alternatives or unknown:
        allowed = ", ".join(PUBLICATION_GRIDS)
        raise argparse.ArgumentTypeError(
            f"--alternatives must use {allowed}; unknown: {', '.join(unknown)}"
        )
    if len(set(alternatives)) != len(alternatives):
        raise argparse.ArgumentTypeError("--alternatives contains duplicates")
    return alternatives


def atomic_write_csv(path: Path, fieldnames: Iterable[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def read_csv_rows_checked(
    path: Path, expected_fields: Iterable[str]
) -> list[dict[str, str]]:
    expected = tuple(expected_fields)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != expected:
            raise ValueError(
                f"invalid CSV header in {path}; expected {','.join(expected)}"
            )
        return [dict(row) for row in reader]


def write_metadata(path: Path, metadata: dict[str, object]) -> None:
    rows = [{"key": key, "value": str(value)} for key, value in sorted(metadata.items())]
    atomic_write_csv(path, ("key", "value"), rows)


def read_metadata(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    if any(set(row) != {"key", "value"} for row in rows):
        raise ValueError(f"malformed metadata file: {path}")
    metadata = {row["key"]: row["value"] for row in rows}
    if len(metadata) != len(rows):
        raise ValueError(f"duplicate metadata key in {path}")
    return metadata


def format_grid(grid: tuple[float, ...]) -> str:
    return ";".join(format(value, ".17g") for value in grid)


def configuration_fingerprint(config: dict[str, object]) -> str:
    encoded = "\n".join(f"{key}={config[key]}" for key in sorted(config))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Data-generating processes and MLE
# ---------------------------------------------------------------------------


def sample_alternative(
    rng: np.random.Generator, n: int, alternative: str, signal: float
) -> np.ndarray:
    if alternative == "t":
        if signal == 0.0:
            return rng.normal(size=n)
        if signal <= 0.0:
            raise ValueError("Student-t signals must be nonnegative")
        return rng.standard_t(df=1.0 / signal, size=n)
    if alternative == "mixture":
        if signal < 0.0:
            raise ValueError("mixture signals must be nonnegative")
        if signal == 0.0:
            return rng.normal(size=n)
        signs = rng.choice((-1.0, 1.0), size=n)
        return signs * signal + rng.normal(size=n)
    if alternative == "gen_chi":
        if signal <= 0.0:
            raise ValueError("generalized-chi signals must be positive")
        magnitude = np.sqrt(2.0 * rng.gamma(shape=signal / 2.0, size=n))
        return rng.choice((-1.0, 1.0), size=n) * magnitude
    raise ValueError(f"unknown alternative: {alternative}")


def mle(x: np.ndarray) -> tuple[float, float]:
    mu = float(np.mean(x))
    sigma = float(np.sqrt(np.mean((x - mu) ** 2)))
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise RuntimeError("Gaussian MLE has non-positive scale")
    return mu, sigma


def median_bandwidth(x: np.ndarray) -> float:
    difference = np.abs(x[:, None] - x[None, :])
    bandwidth = float(np.median(difference[np.triu_indices(x.size, k=1)]))
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        raise RuntimeError("median bandwidth is non-positive")
    return bandwidth


# ---------------------------------------------------------------------------
# Gaussian Stein kernels and empirical-bootstrap calibrations
# ---------------------------------------------------------------------------


def gaussian_stein_matrices(
    x: np.ndarray, mu: float, variance: float, bandwidth: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    difference = x[:, None] - x[None, :]
    kernel = np.exp(-0.5 * difference**2 / bandwidth**2)
    inverse_h2 = 1.0 / bandwidth**2
    score = -(x - mu) / variance
    mixed = (score[:, None] - difference * inverse_h2) * kernel
    stein = (
        score[:, None] * score[None, :] * kernel
        + score[:, None] * difference * inverse_h2 * kernel
        - score[None, :] * difference * inverse_h2 * kernel
        + (inverse_h2 - difference**2 * inverse_h2**2) * kernel
    )
    return kernel, stein, mixed


def gaussian_stein_gradient(
    x: np.ndarray, mu: float, variance: float, bandwidth: float
) -> np.ndarray:
    score = -(x - mu) / variance
    derivative = np.column_stack(
        (
            np.full(x.size, 1.0 / variance),
            (x - mu) / variance**2,
        )
    )
    difference = x[:, None] - x[None, :]
    kernel = np.exp(-0.5 * difference**2 / bandwidth**2)
    scaled_difference = difference * kernel / bandwidth**2
    common_i = score[None, :] * kernel + scaled_difference
    common_j = score[:, None] * kernel - scaled_difference
    return (
        derivative[:, None, :] * common_i[:, :, None]
        + derivative[None, :, :] * common_j[:, :, None]
    )


def weighted_u(matrix: np.ndarray, counts: np.ndarray) -> float:
    n = int(np.sum(counts))
    total = counts @ matrix @ counts - counts @ np.diag(matrix)
    return float(total / (n * (n - 1)))


def weighted_u_batch(matrices: np.ndarray, counts: np.ndarray) -> np.ndarray:
    n = counts.shape[1]
    diagonal = matrices[:, np.arange(n), np.arange(n)]
    total = (
        np.einsum("bi,bij,bj->b", counts, matrices, counts, optimize=True)
        - np.einsum("bi,bi->b", counts, diagonal, optimize=True)
    )
    return total / (n * (n - 1))


def adjusted_gram_mle(
    x: np.ndarray, mu: float, variance: float, bandwidth: float
) -> np.ndarray:
    n = x.size
    kernel, stein, mixed = gaussian_stein_matrices(x, mu, variance, bandwidth)
    centered = x - mu
    influence = np.column_stack((centered, centered**2 - variance))
    derivative = np.column_stack(
        (np.full(n, 1.0 / variance), centered / variance**2)
    )
    influence -= influence.mean(axis=0, keepdims=True)
    cross = mixed @ derivative / n
    derivative_gram = derivative.T @ kernel @ derivative / (n * n)
    adjusted = (
        stein
        + cross @ influence.T
        + influence @ cross.T
        + influence @ derivative_gram @ influence.T
    )
    row_mean = adjusted.mean(axis=1, keepdims=True)
    return adjusted - row_mean - row_mean.T + adjusted.mean()


def brueck_midpoint_statistics_mle(
    x: np.ndarray,
    mu: float,
    variance: float,
    bandwidth: float,
    counts_all: np.ndarray,
) -> np.ndarray:
    """Factor-corrected midpoint empirical U-bootstrap statistics."""

    n = x.size
    mean_b = counts_all @ x / n
    second_b = counts_all @ (x * x) / n
    variance_b = np.maximum(
        second_b - mean_b**2, np.finfo(float).tiny
    )
    theta_b = np.column_stack((mean_b, variance_b))
    score_b = -(x[None, :] - mean_b[:, None]) / variance_b[:, None]
    derivative_b = np.stack(
        (
            np.broadcast_to(1.0 / variance_b[:, None], score_b.shape),
            (x[None, :] - mean_b[:, None]) / variance_b[:, None] ** 2,
        ),
        axis=2,
    )

    difference = x[:, None] - x[None, :]
    kernel = np.exp(-0.5 * difference**2 / bandwidth**2)
    scaled_difference = difference * kernel / bandwidth**2
    trace = (
        1.0 / bandwidth**2 - difference**2 / bandwidth**4
    ) * kernel
    kernel_b = (
        score_b[:, :, None] * score_b[:, None, :] * kernel
        + score_b[:, :, None] * scaled_difference
        - score_b[:, None, :] * scaled_difference
        + trace
    )
    row_mean = kernel_b.mean(axis=2, keepdims=True)
    centered_b = (
        kernel_b
        - row_mean
        - np.swapaxes(row_mean, 1, 2)
        + kernel_b.mean(axis=(1, 2))[:, None, None]
    )
    base = weighted_u_batch(centered_b, counts_all)

    common_i = score_b[:, None, :] * kernel + scaled_difference
    common_j = score_b[:, :, None] * kernel - scaled_difference
    gradient_b = (
        derivative_b[:, :, None, :] * common_i[:, :, :, None]
        + derivative_b[:, None, :, :] * common_j[:, :, :, None]
    )
    diagonal_b = gradient_b[:, np.arange(n), np.arange(n), :]
    gradient_boot_u = (
        np.einsum(
            "bi,bija,bj->ba", counts_all, gradient_b, counts_all, optimize=True
        )
        - np.einsum("bi,bia->ba", counts_all, diagonal_b, optimize=True)
    ) / (n * (n - 1))

    gradient_hat = gaussian_stein_gradient(x, mu, variance, bandwidth)
    diagonal_hat = gradient_hat[np.arange(n), np.arange(n), :]
    gradient_original_u = np.array(
        [
            (gradient_hat[:, :, coordinate].sum()
             - np.trace(gradient_hat[:, :, coordinate]))
            / (n * (n - 1))
            for coordinate in range(2)
        ]
    )
    gradient_hat_boot_u = (
        np.einsum(
            "bi,ija,bj->ba", counts_all, gradient_hat, counts_all, optimize=True
        )
        - np.einsum("bi,ia->ba", counts_all, diagonal_hat, optimize=True)
    ) / (n * (n - 1))
    correction_gradient = (
        0.5 * (gradient_boot_u + gradient_hat_boot_u)
        - gradient_original_u
    )
    correction = np.einsum(
        "ba,ba->b",
        theta_b - np.array([mu, variance]),
        correction_gradient,
        optimize=True,
    )
    return base + correction


def empirical_pvalues_mle(
    x: np.ndarray,
    fitted: tuple[float, float],
    bandwidth: float,
    bootstrap_draws: int,
    seed: int,
) -> tuple[float, float, float, float]:
    shared_started = time.perf_counter()
    mu, sigma = fitted
    variance = sigma**2
    _, stein, _ = gaussian_stein_matrices(x, mu, variance, bandwidth)
    observed_v = float(stein.mean())
    observed_u = weighted_u(stein, np.ones(x.size))
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(
        x.size,
        np.full(x.size, 1.0 / x.size),
        size=bootstrap_draws,
    )
    weights = rng.standard_normal((bootstrap_draws, x.size))
    weights -= weights.mean(axis=1, keepdims=True)
    shared_seconds = time.perf_counter() - shared_started

    started = time.perf_counter()
    adjusted = adjusted_gram_mle(x, mu, variance, bandwidth)
    influence_statistics = np.einsum(
        "bi,ij,bj->b", weights, adjusted, weights, optimize=True
    ) / x.size**2
    influence_seconds = shared_seconds + time.perf_counter() - started

    started = time.perf_counter()
    naive_statistics = np.einsum(
        "bi,ij,bj->b", weights, stein, weights, optimize=True
    ) / x.size**2
    naive_seconds = shared_seconds + time.perf_counter() - started

    started = time.perf_counter()
    midpoint_statistics = brueck_midpoint_statistics_mle(
        x, mu, variance, bandwidth, counts
    )
    midpoint_seconds = shared_seconds + time.perf_counter() - started

    denominator = bootstrap_draws + 1
    started = time.perf_counter()
    influence_pvalue = float(
        (1 + np.sum(influence_statistics >= observed_v)) / denominator
    )
    influence_seconds += time.perf_counter() - started
    started = time.perf_counter()
    naive_pvalue = float(
        (1 + np.sum(naive_statistics >= observed_v)) / denominator
    )
    naive_seconds += time.perf_counter() - started
    started = time.perf_counter()
    midpoint_pvalue = float(
        (1 + np.sum(midpoint_statistics >= observed_u)) / denominator
    )
    midpoint_seconds += time.perf_counter() - started
    return (
        influence_pvalue,
        midpoint_pvalue,
        naive_pvalue,
        influence_seconds,
        midpoint_seconds,
        naive_seconds,
    )


# ---------------------------------------------------------------------------
# Parametric-bootstrap KSD, MMD, and KS
# ---------------------------------------------------------------------------


def ksd_u(
    x: np.ndarray, mu: float, sigma: float, bandwidth: float
) -> float:
    _, matrix, _ = gaussian_stein_matrices(x, mu, sigma**2, bandwidth)
    return weighted_u(matrix, np.ones(x.size))


def mmd_v(x: np.ndarray, y: np.ndarray, bandwidth: float) -> float:
    xx = (x[:, None] - x[None, :]) ** 2
    yy = (y[:, None] - y[None, :]) ** 2
    xy = (x[:, None] - y[None, :]) ** 2
    scale = 0.5 / bandwidth**2
    return float(
        np.exp(-xx * scale).mean()
        + np.exp(-yy * scale).mean()
        - 2.0 * np.exp(-xy * scale).mean()
    )


def ks_statistic(x: np.ndarray, mu: float, sigma: float) -> float:
    return float(kstest(x, "norm", args=(mu, sigma))[0])


def parametric_pvalues_mle(
    x: np.ndarray,
    fitted: tuple[float, float],
    bandwidth: float,
    bootstrap_draws: int,
    seed: int,
    bootstrap_mode: str,
) -> tuple[float, float, float, float, float, float]:
    rng = np.random.default_rng(seed)
    n = x.size
    mu_hat, sigma_hat = fitted
    variance_hat = sigma_hat**2

    ksd_seconds = 0.0
    mmd_seconds = 0.0
    ks_seconds = 0.0

    started = time.perf_counter()
    y_observed = rng.normal(mu_hat, sigma_hat, size=n)
    mmd_bandwidth = median_bandwidth(np.concatenate((x, y_observed)))
    mmd_seconds += time.perf_counter() - started

    started = time.perf_counter()
    observed_ksd = ksd_u(x, mu_hat, sigma_hat, bandwidth)
    ksd_seconds += time.perf_counter() - started
    started = time.perf_counter()
    observed_mmd = mmd_v(x, y_observed, mmd_bandwidth)
    mmd_seconds += time.perf_counter() - started
    started = time.perf_counter()
    observed_ks = ks_statistic(x, mu_hat, sigma_hat)
    ks_seconds += time.perf_counter() - started

    bootstrap_ksd = np.empty(bootstrap_draws)
    bootstrap_mmd = np.empty(bootstrap_draws)
    bootstrap_ks = np.empty(bootstrap_draws)
    for draw in range(bootstrap_draws):
        shared_started = time.perf_counter()
        xb = rng.normal(mu_hat, sigma_hat, size=n)
        if bootstrap_mode == "refit":
            mu_b, sigma_b = mle(xb)
        else:
            # One Newton/influence update in (mean, variance) coordinates,
            # retained as a diagnostic option from the source experiment.
            mu_b = float(mu_hat + np.mean(xb - mu_hat))
            variance_b = float(
                variance_hat + np.mean((xb - mu_hat) ** 2 - variance_hat)
            )
            sigma_b = math.sqrt(max(variance_b, np.finfo(float).tiny))
        shared_seconds = time.perf_counter() - shared_started
        ksd_seconds += shared_seconds
        mmd_seconds += shared_seconds
        ks_seconds += shared_seconds

        started = time.perf_counter()
        yb = rng.normal(mu_b, sigma_b, size=n)
        bandwidth_b = median_bandwidth(np.concatenate((xb, yb)))
        mmd_seconds += time.perf_counter() - started

        started = time.perf_counter()
        bootstrap_ksd[draw] = ksd_u(xb, mu_b, sigma_b, bandwidth)
        ksd_seconds += time.perf_counter() - started
        started = time.perf_counter()
        bootstrap_mmd[draw] = mmd_v(xb, yb, bandwidth_b)
        mmd_seconds += time.perf_counter() - started
        started = time.perf_counter()
        bootstrap_ks[draw] = ks_statistic(xb, mu_b, sigma_b)
        ks_seconds += time.perf_counter() - started

    denominator = bootstrap_draws + 1
    started = time.perf_counter()
    p_ksd = float((1 + np.sum(bootstrap_ksd >= observed_ksd)) / denominator)
    ksd_seconds += time.perf_counter() - started
    started = time.perf_counter()
    p_mmd = float((1 + np.sum(bootstrap_mmd >= observed_mmd)) / denominator)
    mmd_seconds += time.perf_counter() - started
    started = time.perf_counter()
    p_ks = float((1 + np.sum(bootstrap_ks >= observed_ks)) / denominator)
    ks_seconds += time.perf_counter() - started
    return (
        p_ksd,
        p_mmd,
        p_ks,
        ksd_seconds,
        mmd_seconds,
        ks_seconds,
    )


def replication_row(
    alternative: str,
    signal: float,
    replication: int,
    method: str,
    p_value: float | str,
    reject: int | str,
    seconds: float,
    source_record: str,
    aggregate_rejection_rate: float | str = "",
) -> dict[str, object]:
    return {
        "alternative": alternative,
        "signal": format(signal, ".17g"),
        "replication": replication,
        "method": method,
        "method_label": METHOD_LABELS[method],
        "test": METHOD_TESTS[method],
        "calibration": METHOD_CALIBRATIONS[method],
        "p_value": p_value,
        "reject": reject,
        "seconds": seconds,
        "source_record": source_record,
        "aggregate_rejection_rate": aggregate_rejection_rate,
    }


def run_one_replication(
    alternative: str,
    signal: float,
    signal_index: int,
    replication: int,
    n: int,
    bootstrap_draws: int,
    seed: int,
    alpha: float,
    bootstrap_mode: str,
) -> list[dict[str, object]]:
    alternative_code = ALTERNATIVE_CODES[alternative]
    data_rng = np.random.default_rng(
        stable_seed((seed, alternative_code, signal_index, replication, 99))
    )
    x = sample_alternative(data_rng, n, alternative, signal)

    started = time.perf_counter()
    bandwidth = median_bandwidth(x)
    observed_ksd_bandwidth_seconds = time.perf_counter() - started
    started = time.perf_counter()
    fitted = mle(x)
    observed_mle_seconds = time.perf_counter() - started

    (
        p_ksd,
        p_mmd,
        p_ks,
        seconds_ksd,
        seconds_mmd,
        seconds_ks,
    ) = parametric_pvalues_mle(
        x,
        fitted,
        bandwidth,
        bootstrap_draws,
        stable_seed((seed, alternative_code, signal_index, replication, 1)),
        bootstrap_mode,
    )
    (
        p_influence,
        p_midpoint,
        p_naive,
        seconds_influence,
        seconds_midpoint,
        seconds_naive,
    ) = empirical_pvalues_mle(
        x,
        fitted,
        bandwidth,
        bootstrap_draws,
        stable_seed((seed, alternative_code, signal_index, replication, 3)),
    )
    values = {
        "SKSD-parametric": (
            p_ksd,
            observed_mle_seconds
            + observed_ksd_bandwidth_seconds
            + seconds_ksd,
        ),
        "SKSD-influence": (
            p_influence,
            observed_mle_seconds
            + observed_ksd_bandwidth_seconds
            + seconds_influence,
        ),
        "SKS": (p_ks, observed_mle_seconds + seconds_ks),
        "SMMD": (p_mmd, observed_mle_seconds + seconds_mmd),
        "Brueck-corrected": (
            p_midpoint,
            observed_mle_seconds
            + observed_ksd_bandwidth_seconds
            + seconds_midpoint,
        ),
        "SKSD-naive-wild": (
            p_naive,
            observed_mle_seconds
            + observed_ksd_bandwidth_seconds
            + seconds_naive,
        ),
    }
    return [
        replication_row(
            alternative,
            signal,
            replication,
            method,
            p_value,
            int(p_value < alpha),
            seconds,
            "simulated replication",
        )
        for method, (p_value, seconds) in values.items()
    ]


def validate_replication_rows(rows: list[dict[str, str]], alpha: float) -> None:
    seen: set[tuple[str, str, int, str]] = set()
    for row in rows:
        if set(row) != set(REPLICATION_FIELDS):
            raise ValueError("replications.csv does not match the required schema")
        alternative = row["alternative"]
        method = row["method"]
        if alternative not in PUBLICATION_GRIDS:
            raise ValueError(f"unknown alternative in replications.csv: {alternative}")
        if method not in METHODS:
            raise ValueError(f"unknown method in replications.csv: {method}")
        if row["method_label"] != METHOD_LABELS[method]:
            raise ValueError(f"wrong method label in replications.csv: {method}")
        if row["test"] != METHOD_TESTS[method]:
            raise ValueError(f"wrong test name in replications.csv: {method}")
        if row["calibration"] != METHOD_CALIBRATIONS[method]:
            raise ValueError(f"wrong calibration in replications.csv: {method}")
        try:
            signal = float(row["signal"])
            replication = int(row["replication"])
            seconds = float(row["seconds"])
        except ValueError as error:
            raise ValueError("replications.csv contains a malformed numeric value") from error
        if not math.isfinite(signal) or replication < 0:
            raise ValueError("replications.csv contains an invalid cell key")
        if not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError("replications.csv contains an invalid timing value")
        key = (alternative, format(signal, ".17g"), replication, method)
        if key in seen:
            raise ValueError(f"duplicate row in replications.csv: {key}")
        seen.add(key)

        source = row["source_record"]
        if source in {"simulated replication", "frozen replication p-value"}:
            try:
                p_value = float(row["p_value"])
                reject = int(row["reject"])
            except ValueError as error:
                raise ValueError("replications.csv contains an invalid p-value") from error
            if not math.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
                raise ValueError("replications.csv contains a p-value outside [0,1]")
            if reject not in {0, 1} or reject != int(p_value < alpha):
                raise ValueError("replications.csv has an inconsistent rejection flag")
            if row["aggregate_rejection_rate"] != "":
                raise ValueError("replication rows cannot contain an aggregate rate")
        elif source == "frozen aggregate only":
            if row["p_value"] != "" or row["reject"] != "":
                raise ValueError("aggregate-only rows cannot contain p-values")
            try:
                rate = float(row["aggregate_rejection_rate"])
            except ValueError as error:
                raise ValueError("aggregate-only row has an invalid rejection rate") from error
            if not math.isfinite(rate) or not 0.0 <= rate <= 1.0 or replication < 1:
                raise ValueError("aggregate-only row has invalid numerical values")
        else:
            raise ValueError(f"unknown source_record in replications.csv: {source}")


def completed_replication_keys(
    rows: list[dict[str, str]],
    args: argparse.Namespace,
    grids: dict[str, tuple[float, ...]],
) -> set[tuple[str, str, int]]:
    validate_replication_rows(rows, args.alpha)
    allowed = {
        (alternative, format(signal, ".17g"), replication)
        for alternative in args.alternatives
        for signal in grids[alternative]
        for replication in range(args.replications)
    }
    grouped: dict[tuple[str, str, int], set[str]] = {}
    for row in rows:
        if row.get("source_record") != "simulated replication":
            raise ValueError("resume input contains non-simulation records")
        key = (
            row["alternative"],
            format(float(row["signal"]), ".17g"),
            int(row["replication"]),
        )
        if key not in allowed:
            raise ValueError(f"resume input contains an unexpected cell: {key}")
        methods = grouped.setdefault(key, set())
        methods.add(row["method"])
    incomplete = {key: methods for key, methods in grouped.items() if methods != set(METHODS)}
    if incomplete:
        first = next(iter(incomplete))
        raise ValueError(f"resume input contains an incomplete replication: {first}")
    return set(grouped)


def summarize_replications(
    rows: list[dict], alpha: float
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float, str], list[dict]] = {}
    for row in rows:
        key = (row["alternative"], float(row["signal"]), row["method"])
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, object]] = []
    for (alternative, signal, method), cell in grouped.items():
        replication_rows = [
            row for row in cell if row["source_record"] != "frozen aggregate only"
        ]
        aggregate_rows = [
            row for row in cell if row["source_record"] == "frozen aggregate only"
        ]
        if replication_rows and aggregate_rows:
            raise ValueError(f"mixed aggregate and replication records for {alternative}/{signal}/{method}")
        if replication_rows:
            rejects = np.array([int(row["reject"]) for row in replication_rows])
            rate = float(rejects.mean())
            replications = len(replication_rows)
            rejections = int(rejects.sum())
            seconds = float(np.mean([float(row["seconds"]) for row in replication_rows]))
            source = replication_rows[0]["source_record"]
        else:
            if len(aggregate_rows) != 1:
                raise ValueError(f"expected one frozen aggregate row for {alternative}/{signal}/{method}")
            aggregate = aggregate_rows[0]
            replications = int(aggregate["replication"])
            rate = float(aggregate["aggregate_rejection_rate"])
            rejections = int(round(rate * replications))
            seconds = float(aggregate["seconds"])
            source = aggregate["source_record"]
        mcse = math.sqrt(rate * (1.0 - rate) / replications)
        summary.append(
            {
                "alternative": alternative,
                "alternative_label": ALTERNATIVE_LABELS[alternative],
                "signal": format(signal, ".17g"),
                "method": method,
                "method_label": METHOD_LABELS[method],
                "test": METHOD_TESTS[method],
                "calibration": METHOD_CALIBRATIONS[method],
                "replications": replications,
                "rejections": rejections,
                "rejection_rate": rate,
                "mcse": mcse,
                "mean_seconds": seconds,
                "source": source,
            }
        )
    method_rank = {method: index for index, method in enumerate(METHODS)}
    alternative_rank = {
        alternative: index for index, alternative in enumerate(PUBLICATION_GRIDS)
    }
    summary.sort(
        key=lambda row: (
            alternative_rank[row["alternative"]],
            float(row["signal"]),
            method_rank[row["method"]],
        )
    )
    return summary


def validate_summary_against_replications(
    summary: list[dict[str, str]],
    replications: list[dict[str, str]],
    alpha: float,
) -> None:
    if not summary:
        raise ValueError("summary.csv is empty")
    if any(set(row) != set(SUMMARY_FIELDS) for row in summary):
        raise ValueError("summary.csv does not match the required schema")
    validate_replication_rows(replications, alpha)
    expected = summarize_replications(replications, alpha)
    key_fields = ("alternative", "signal", "method")

    def keyed(rows: Iterable[dict]) -> dict[tuple[str, str, str], dict]:
        result: dict[tuple[str, str, str], dict] = {}
        for row in rows:
            key = tuple(str(row[field]) for field in key_fields)
            if key in result:
                raise ValueError(f"summary.csv contains a duplicate cell: {key}")
            result[key] = row
        return result

    observed_by_key = keyed(summary)
    expected_by_key = keyed(expected)
    if set(observed_by_key) != set(expected_by_key):
        raise ValueError("summary.csv cells do not match replications.csv")
    numeric_fields = {
        "replications",
        "rejections",
        "rejection_rate",
        "mcse",
        "mean_seconds",
    }
    for key, expected_row in expected_by_key.items():
        observed_row = observed_by_key[key]
        for field in SUMMARY_FIELDS:
            if field in numeric_fields:
                try:
                    observed_value = float(observed_row[field])
                    expected_value = float(expected_row[field])
                except ValueError as error:
                    raise ValueError(f"summary.csv has invalid {field}: {key}") from error
                if not math.isfinite(observed_value) or not math.isclose(
                    observed_value,
                    expected_value,
                    rel_tol=1e-12,
                    abs_tol=1e-14,
                ):
                    raise ValueError(f"summary.csv disagrees on {field}: {key}")
            elif str(observed_row[field]) != str(expected_row[field]):
                raise ValueError(f"summary.csv disagrees on {field}: {key}")


def run_fresh_simulation(
    args: argparse.Namespace,
    config: dict[str, object],
    grids: dict[str, tuple[float, ...]],
    output_dir: Path,
) -> tuple[list[dict], list[dict], dict[str, object]]:
    replications_path = output_dir / "replications.csv"
    metadata_path = output_dir / "run_metadata.csv"
    script_sha256 = file_sha256(SCRIPT_PATH)
    fingerprint = configuration_fingerprint(
        {**config, "schema_version": SCHEMA_VERSION, "script_sha256": script_sha256}
    )
    metadata: dict[str, object] = {
        **config,
        "schema_version": SCHEMA_VERSION,
        "configuration_fingerprint": fingerprint,
        "run_fingerprint": fingerprint,
        "script_sha256": script_sha256,
        "rng_scheme": "SeedSequence([seed,alternative_code,signal_index,replication,stream])",
        "status": "running",
        "started_utc": utc_now(),
        "legacy_replication_detail": "not applicable; all methods simulated",
        "timing_protocol": FRESH_TIMING_PROTOCOL,
        "timing_clock": "perf_counter",
        "timing_unit": "seconds per simulated dataset",
        "timing_shared_scope": (
            "required shared costs charged in full to each applicable method"
        ),
        "timing_observed_setup_scope": (
            "observed MLE for all methods; observed KSD bandwidth for the "
            "three KSD methods"
        ),
        "timing_total_formula": (
            "required observed setup + fitted-null sampling/refitting or "
            "empirical shared preprocessing + method statistic/bootstrap/p-value"
        ),
        "timing_excludes": (
            "observed-data simulation; checkpoint and output I/O; plotting"
        ),
        "timing_method_order": (
            "joint parametric KSD/MMD/KS bootstrap (KSD,MMD,KS statistic order)"
            "|SKSD-influence|SKSD-naive-wild|Brueck-corrected"
        ),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "scipy_version": installed_version("scipy"),
        "plotting_backend": "R/ggplot2",
        "platform": platform.platform(),
        "cpu_architecture": platform.machine() or "unknown",
        "cpu_processor": platform.processor() or "unknown",
        "logical_cpu_count": os.cpu_count() or "unknown",
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "unset"),
        "OPENBLAS_NUM_THREADS": os.environ.get(
            "OPENBLAS_NUM_THREADS", "unset"
        ),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "unset"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get(
            "VECLIB_MAXIMUM_THREADS", "unset"
        ),
    }

    if args.resume:
        if not metadata_path.is_file() or not replications_path.is_file():
            raise FileNotFoundError(
                "--resume requires both run_metadata.csv and replications.csv "
                f"in {output_dir}"
            )
        existing_metadata = read_metadata(metadata_path)
        required_metadata = {
            "schema_version": str(SCHEMA_VERSION),
            "configuration_fingerprint": fingerprint,
            "run_fingerprint": fingerprint,
            "script_sha256": script_sha256,
        }
        mismatches = {
            key: (existing_metadata.get(key), expected)
            for key, expected in required_metadata.items()
            if existing_metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(
                "resume metadata does not match this configuration and code; "
                "use a different --output-dir or inspect before using --force: "
                f"{mismatches}"
            )
        if existing_metadata.get("status") not in {"running", "complete"}:
            raise ValueError("resume metadata has an invalid run status")
        rows: list[dict] = read_csv_rows_checked(
            replications_path, REPLICATION_FIELDS
        )
        completed = completed_replication_keys(rows, args, grids)
        if existing_metadata.get("status") == "complete":
            expected_datasets = sum(
                len(grids[alternative]) for alternative in args.alternatives
            ) * args.replications
            if len(completed) != expected_datasets:
                raise ValueError(
                    "completed resume metadata accompanies an incomplete checkpoint"
                )
        metadata["started_utc"] = existing_metadata.get("started_utc", utc_now())
        metadata["resumed_utc"] = utc_now()
    else:
        rows = []
        completed = set()
        write_metadata(metadata_path, metadata)
        # Create the checkpoint schema immediately so an interruption before
        # the first periodic checkpoint is still resumable by the root runner.
        atomic_write_csv(replications_path, REPLICATION_FIELDS, rows)

    total_datasets = sum(len(grids[alt]) for alt in args.alternatives) * args.replications
    completed_datasets = len(completed)
    since_checkpoint = 0
    for alternative in args.alternatives:
        for signal_index, signal in enumerate(grids[alternative]):
            for replication in range(args.replications):
                key = (alternative, format(signal, ".17g"), replication)
                if key in completed:
                    continue
                started = time.perf_counter()
                new_rows = run_one_replication(
                    alternative,
                    signal,
                    signal_index,
                    replication,
                    args.n,
                    args.bootstrap_draws,
                    args.seed,
                    args.alpha,
                    args.bootstrap_mode,
                )
                rows.extend(new_rows)
                completed.add(key)
                completed_datasets += 1
                since_checkpoint += 1
                print(
                    f"[{completed_datasets}/{total_datasets}] "
                    f"{alternative} signal={signal:g} rep={replication} "
                    f"wall={time.perf_counter() - started:.2f}s",
                    flush=True,
                )
                if since_checkpoint >= args.checkpoint_every:
                    atomic_write_csv(replications_path, REPLICATION_FIELDS, rows)
                    since_checkpoint = 0

    atomic_write_csv(replications_path, REPLICATION_FIELDS, rows)
    summary = summarize_replications(rows, args.alpha)
    atomic_write_csv(output_dir / "summary.csv", SUMMARY_FIELDS, summary)
    metadata.update(
        status="complete",
        completed_utc=utc_now(),
        completed_datasets=completed_datasets,
        replication_rows=len(rows),
        summary_rows=len(summary),
        replications_sha256=file_sha256(replications_path),
        summary_sha256=file_sha256(output_dir / "summary.csv"),
    )
    return rows, summary, metadata




# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def make_figure(
    summary: list[dict], output_dir: Path, alpha: float
) -> str:
    """Render the saved summary with the shared R/ggplot2 template."""
    alternatives = [
        alternative
        for alternative in PUBLICATION_GRIDS
        if any(row["alternative"] == alternative for row in summary)
    ]
    if not alternatives:
        raise ValueError("cannot plot an empty summary")
    timing_values = [
        float(row["mean_seconds"])
        for row in summary
        if row["alternative"] == alternatives[0] and row["method"] in METHODS
    ]
    if any(not math.isfinite(value) or value <= 0.0 for value in timing_values):
        raise ValueError("timing panels require finite, strictly positive runtimes")
    summary_path = output_dir / "summary.csv"
    plot_script = SCRIPT_PATH.parent.parent / "figure-code" / "plot_01_gaussianity_mle.R"
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
    return "created figure.pdf and figure.png"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gaussianity tests with an unknown mean and variance (MLE nuisance fit)."
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="validate completed output CSVs and recreate figures without simulation",
    )
    parser.add_argument(
        "--check",
        dest="quick",
        action="store_true",
        help="small deterministic check design; explicit numeric overrides still win",
    )
    parser.add_argument("--quick", dest="quick", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--n", type=int, default=None, help="sample size")
    parser.add_argument(
        "--bootstrap-draws", type=int, default=None, help="bootstrap draws per dataset"
    )
    parser.add_argument(
        "--replications", type=int, default=None, help="Monte Carlo replications per cell"
    )
    parser.add_argument("--seed", type=int, default=PUBLICATION_SEED)
    parser.add_argument("--alpha", type=float, default=PUBLICATION_ALPHA)
    parser.add_argument(
        "--bootstrap-mode",
        choices=("refit", "one-step"),
        default="refit",
        help="nuisance update inside the fitted-model bootstrap (default: refit)",
    )
    parser.add_argument(
        "--alternatives",
        type=parse_alternatives,
        default=tuple(PUBLICATION_GRIDS),
        help="comma-separated subset of t,mixture,gen_chi",
    )
    parser.add_argument("--t-grid", default=None, help="comma-separated Student-t signal grid")
    parser.add_argument("--mixture-grid", default=None, help="comma-separated mixture grid")
    parser.add_argument("--gen-chi-grid", default=None, help="comma-separated generalized-chi grid")
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=f"output directory (default: {DEFAULT_OUTPUT_DIR}; --check uses simulation-analysis-check/)",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--resume", action="store_true", help="resume a validated fresh run")
    action.add_argument("--force", action="store_true", help="replace this script's existing outputs")
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="checkpoint after this many completed datasets (default: 10; check: 1)",
    )
    parser.add_argument("--no-plot", action="store_true", help="do not create PDF/PNG figures")
    return parser


def resolve_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> dict[str, tuple[float, ...]]:
    if args.plot_only and args.resume:
        parser.error("--plot-only does not use --resume; pass --force to replace figures")
    if args.plot_only and args.no_plot:
        parser.error("--plot-only and --no-plot are mutually exclusive")
    args.n = args.n if args.n is not None else (60 if args.quick else PUBLICATION_N)
    args.bootstrap_draws = (
        args.bootstrap_draws
        if args.bootstrap_draws is not None
        else (39 if args.quick else PUBLICATION_BOOTSTRAPS)
    )
    args.replications = (
        args.replications
        if args.replications is not None
        else (2 if args.quick else PUBLICATION_REPLICATIONS)
    )
    args.checkpoint_every = (
        args.checkpoint_every
        if args.checkpoint_every is not None
        else (1 if args.quick else 10)
    )
    if args.n < 3:
        parser.error("--n must be at least 3")
    if args.bootstrap_draws < 1:
        parser.error("--bootstrap-draws must be positive")
    if args.replications < 1:
        parser.error("--replications must be positive")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be positive")
    if not 0.0 < args.alpha < 1.0:
        parser.error("--alpha must lie strictly between zero and one")

    base = QUICK_GRIDS if args.quick else PUBLICATION_GRIDS
    custom = {
        "t": args.t_grid,
        "mixture": args.mixture_grid,
        "gen_chi": args.gen_chi_grid,
    }
    grids = {
        alternative: (
            parse_grid(custom[alternative], f"--{alternative}-grid")
            if custom[alternative] is not None
            else base[alternative]
        )
        for alternative in PUBLICATION_GRIDS
    }
    if any(value < 0.0 for value in grids["t"]):
        parser.error("--t-grid must contain nonnegative values")
    if any(value < 0.0 for value in grids["mixture"]):
        parser.error("--mixture-grid must contain nonnegative values")
    if any(value <= 0.0 for value in grids["gen_chi"]):
        parser.error("--gen-chi-grid must contain positive values")
    return grids


def prepare_output_dir(args: argparse.Namespace) -> Path:
    output_dir = resolve_output_dir(args)
    if args.plot_only:
        # The figure writers replace their derived artifacts only after the
        # numerical inputs have passed validation, so a failed replot does not
        # first discard the previously reviewed figure.
        return output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    collisions = [output_dir / filename for filename in OUTPUT_FILENAMES if (output_dir / filename).exists()]
    if args.force:
        for path in collisions:
            path.unlink()
    elif collisions and not args.resume:
        listing = "\n".join(f"  - {path}" for path in collisions)
        raise FileExistsError(
            "Gaussianity outputs already exist; use --resume for a compatible "
            f"fresh run or --force after inspection:\n{listing}"
        )
    return output_dir


def validate_completed_design(
    metadata: dict[str, str],
    replications: list[dict[str, str]],
    summary: list[dict[str, str]],
    alpha: float,
) -> None:
    """Check complete saved tables without requiring their producing code."""
    try:
        repetition_count = int(metadata["replications"])
        alternatives = tuple(metadata["alternatives"].split(";"))
        row_counts = (int(metadata["replication_rows"]), int(metadata["summary_rows"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("run_metadata.csv lacks valid completed design counts") from error
    if repetition_count < 1 or row_counts != (len(replications), len(summary)):
        raise ValueError("completed table counts do not match run_metadata.csv")
    if (
        not alternatives
        or len(set(alternatives)) != len(alternatives)
        or not set(alternatives) <= set(PUBLICATION_GRIDS)
        or {row["alternative"] for row in summary} != set(alternatives)
    ):
        raise ValueError("completed table alternatives do not match run_metadata.csv")
    cell_methods: dict[tuple[str, str], set[str]] = {}
    for row in summary:
        if int(row["replications"]) != repetition_count:
            raise ValueError("completed table has an incomplete replication count")
        cell_methods.setdefault((row["alternative"], row["signal"]), set()).add(row["method"])
    if any(methods != set(METHODS) for methods in cell_methods.values()):
        raise ValueError("completed table has an incomplete set of methods")

    # Old standardized tables remain plottable, but are never read from NPZ
    # inputs. Fresh outputs additionally record the entire intended grid.
    if metadata.get("mode") == "simulate":
        try:
            grids = {
                alternative: tuple(float(value) for value in metadata[f"grid_{alternative}"].split(";"))
                for alternative in alternatives
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("run_metadata.csv lacks valid simulation grids") from error
        if any(
            not grid or len(set(grid)) != len(grid) or not all(math.isfinite(value) for value in grid)
            for grid in grids.values()
        ):
            raise ValueError("run_metadata.csv contains invalid simulation grids")
        design = argparse.Namespace(alpha=alpha, alternatives=alternatives, replications=repetition_count)
        completed = completed_replication_keys(replications, design, grids)
        if len(completed) != sum(len(grid) for grid in grids.values()) * repetition_count:
            raise ValueError("completed tables do not contain the full simulation design")


def validate_saved_results(output_dir: Path) -> tuple[list[dict], dict[str, str]]:
    """Validate an existing tidy summary without inventing replication records.

    This read-only compatibility path accepts the already-exported publication
    table. It never reads numerical archives or treats historical timings as
    measurements from the new joint benchmarking loop.
    """
    metadata = read_metadata(output_dir / "run_metadata.csv")
    required = {
        "schema_version": "saved-csv-v1",
        "study": "01-gaussianity-mle",
        "workflow": "saved-results",
        "run_complete": "true",
        "raw_available": "false",
        "alpha": "0.05",
        "n": "100",
        "bootstrap_draws": "300",
        "replications": "300",
        "timing_protocol": "historical-recorded",
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        raise ValueError("invalid saved Gaussianity summary provenance")
    summary_path = output_dir / "summary.csv"
    digest = file_sha256(summary_path)
    if metadata.get("summary_sha256") != digest:
        raise ValueError("saved Gaussianity summary checksum does not match metadata")
    if metadata.get("run_fingerprint") != digest:
        raise ValueError("saved Gaussianity fingerprint must identify its source table")
    fields = ("alt", "alt_label", "signal", "method", "test", "calibration",
              "nuisance", "reject", "t_per_rep")
    rows = read_csv_rows_checked(summary_path, fields)
    expected = {(alt, signal, method) for alt, grid in PUBLICATION_GRIDS.items()
                for signal in grid for method in SAVED_PUBLICATION_METHODS}
    seen = set()
    summary = []
    for row in rows:
        if row["nuisance"] not in {"MLE", "minKSD"}:
            raise ValueError("unknown nuisance estimator in saved Gaussianity table")
        if row["nuisance"] != "MLE":
            continue
        key = (row["alt"], float(row["signal"]), row["method"])
        if key not in expected or key in seen:
            raise ValueError("duplicate or unexpected saved Gaussianity summary cell")
        seen.add(key)
        rejection = float(row["reject"])
        seconds = float(row["t_per_rep"])
        if not 0.0 <= rejection <= 1.0 or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("invalid saved Gaussianity rejection rate or runtime")
        if not math.isclose(300 * rejection, round(300 * rejection), abs_tol=1e-9):
            raise ValueError("saved Gaussianity rate is inconsistent with 300 replications")
        if row["test"] != METHOD_TESTS[row["method"]]:
            raise ValueError("saved Gaussianity method/test mismatch")
        expected_calibration = {
            "SKSD-influence": "influence-adjusted Efron",
            "Brueck-corrected": "midpoint factor-corrected empirical Efron",
        }.get(row["method"], "parametric")
        if row["calibration"] != expected_calibration:
            raise ValueError("saved Gaussianity calibration mismatch")
        summary.append({"alternative": row["alt"], "signal": row["signal"],
                        "method": row["method"], "rejection_rate": row["reject"],
                        "mean_seconds": row["t_per_rep"]})
    if seen != expected:
        raise ValueError("saved Gaussianity summary is missing publication cells")
    return summary, metadata


def run_plot_only(args: argparse.Namespace, output_dir: Path) -> int:
    summary_path = output_dir / "summary.csv"
    replications_path = output_dir / "replications.csv"
    metadata_path = output_dir / "run_metadata.csv"
    if metadata_path.is_file() and read_metadata(metadata_path).get("schema_version") == "saved-csv-v1":
        summary, metadata = validate_saved_results(output_dir)
        if args.alpha != float(metadata["alpha"]):
            raise ValueError("--alpha cannot change saved rejection rates in plot-only mode")
        plot_status = make_figure(summary, output_dir, float(metadata["alpha"]))
        if not plot_status.startswith("created"):
            raise RuntimeError(f"--plot-only could not create figures: {plot_status}")
        metadata.update(plot_status=plot_status, plot_updated_utc=utc_now(),
                        figure_pdf_sha256=file_sha256(output_dir / "figure.pdf"),
                        figure_png_sha256=file_sha256(output_dir / "figure.png"))
        write_metadata(metadata_path, metadata)
        print(f"plotted saved Gaussianity summary (raw p-values unavailable): {output_dir}")
        return 0
    missing = [
        path
        for path in (replications_path, summary_path, metadata_path)
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "--plot-only requires a completed numerical run; missing: "
            + ", ".join(str(path) for path in missing)
        )
    metadata: dict[str, object] = read_metadata(metadata_path)
    if metadata.get("schema_version") != str(SCHEMA_VERSION):
        raise ValueError("run_metadata.csv has the wrong schema version")
    if metadata.get("status") != "complete":
        raise ValueError("--plot-only requires status=complete metadata")
    if metadata.get("mode") not in {"simulate", "exact-paper"}:
        raise ValueError("run_metadata.csv has an unknown saved-results mode")
    if not metadata.get("run_fingerprint") or not metadata.get("script_sha256"):
        raise ValueError("run_metadata.csv lacks source/fingerprint provenance")
    expected_hashes = {
        "replications_sha256": file_sha256(replications_path),
        "summary_sha256": file_sha256(summary_path),
    }
    for key, observed in expected_hashes.items():
        if metadata.get(key) != observed:
            raise ValueError(f"{key} does not match the completed numerical output")
    replications = read_csv_rows_checked(replications_path, REPLICATION_FIELDS)
    summary = read_csv_rows_checked(summary_path, SUMMARY_FIELDS)
    plot_alpha = float(metadata.get("alpha", args.alpha))
    if not math.isfinite(plot_alpha) or not 0.0 < plot_alpha < 1.0:
        raise ValueError("run_metadata.csv contains an invalid alpha")
    validate_summary_against_replications(summary, replications, plot_alpha)
    validate_completed_design(metadata, replications, summary, plot_alpha)
    plot_status = make_figure(summary, output_dir, plot_alpha)
    if not plot_status.startswith("created"):
        raise RuntimeError(f"--plot-only could not create figures: {plot_status}")
    metadata["plot_status"] = plot_status
    metadata["plot_updated_utc"] = utc_now()
    metadata["figure_pdf_sha256"] = file_sha256(output_dir / "figure.pdf")
    metadata["figure_png_sha256"] = file_sha256(output_dir / "figure.png")
    write_metadata(metadata_path, metadata)
    if plot_status.startswith("created"):
        print(f"wrote {output_dir / 'figure.pdf'}")
        print(f"wrote {output_dir / 'figure.png'}")
    print(f"updated {metadata_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    grids = resolve_args(parser, args)
    try:
        output_dir = prepare_output_dir(args)
        if args.plot_only:
            return run_plot_only(args, output_dir)
        selected_grids = {alternative: grids[alternative] for alternative in args.alternatives}
        config = {
            "mode": "simulate",
            "n": args.n,
            "bootstrap_draws": args.bootstrap_draws,
            "replications": args.replications,
            "seed": args.seed,
            "alpha": args.alpha,
            "bootstrap_mode": args.bootstrap_mode,
            "alternatives": ";".join(args.alternatives),
            **{
                f"grid_{alternative}": format_grid(selected_grids[alternative])
                for alternative in args.alternatives
            },
        }
        rows, summary, metadata = run_fresh_simulation(
            args, config, selected_grids, output_dir
        )

        if args.no_plot:
            plot_status = "disabled by --no-plot"
        else:
            plot_status = make_figure(summary, output_dir, args.alpha)
        metadata["plot_status"] = plot_status
        metadata["output_directory"] = output_dir
        if (output_dir / "figure.pdf").is_file():
            metadata["figure_pdf_sha256"] = file_sha256(output_dir / "figure.pdf")
        if (output_dir / "figure.png").is_file():
            metadata["figure_png_sha256"] = file_sha256(output_dir / "figure.png")
        write_metadata(output_dir / "run_metadata.csv", metadata)
        print(f"wrote {output_dir / 'replications.csv'}")
        print(f"wrote {output_dir / 'summary.csv'}")
        print(f"wrote {output_dir / 'run_metadata.csv'}")
        if not args.no_plot and plot_status.startswith("created"):
            print(f"wrote {output_dir / 'figure.pdf'}")
            print(f"wrote {output_dir / 'figure.png'}")
        return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
