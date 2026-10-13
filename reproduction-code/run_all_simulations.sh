#!/usr/bin/env bash

# Run simulations, or plot their saved CSV results without numerical reruns.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SIMULATION_CODE="$SCRIPT_DIR/simulation-code"
if [[ -f "$SCRIPT_DIR/renv/activate.R" ]]; then
  export RENV_PROJECT="$SCRIPT_DIR"
  export R_PROFILE_USER="$SCRIPT_DIR/.Rprofile"
fi

usage() {
  echo "Usage: $0 [full|check|plot-only] [--force] [--results-root DIR] [--skip-check] [--python PATH]"
  echo "full reruns completed simulations and resumes compatible checkpoints; check reuses completed results."
  echo "plot-only requires completed CSV results for all six studies; it never starts a simulation."
  echo "Default results: full uses results/simulation-analysis-full; check uses results/simulation-analysis-check; plot-only uses results/simulation-analysis."
  echo "--results-root is the directory containing the study folders directly."
}

metadata_value_is() {
  local metadata_path="$1"
  local requested_key="$2"
  local requested_value="$3"
  awk -F, -v key="$requested_key" -v expected="$requested_value" '
    $1 == key {
      sub(/\r$/, "", $2)
      if ($2 == expected) found = 1
    }
    END { exit(found ? 0 : 1) }
  ' "$metadata_path"
}

metadata_has_value() {
  local metadata_path="$1"
  local requested_key="$2"
  awk -F, -v key="$requested_key" '
    $1 == key {
      sub(/\r$/, "", $2)
      if ($2 != "") found = 1
    }
    END { exit(found ? 0 : 1) }
  ' "$metadata_path"
}

mode="full"
if (($# > 0)); then
  case "$1" in
    full|check|plot-only)
      mode="$1"
      shift
      ;;
    quick)
      mode="check"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
fi

force=0
skip_check=0
results_root=""
python_bin="${PYTHON_BIN:-python}"

while (($# > 0)); do
  case "$1" in
    --force)
      force=1
      shift
      ;;
    --skip-check)
      skip_check=1
      shift
      ;;
    --results-root)
      results_root="${2:?Missing directory after --results-root}"
      shift 2
      ;;
    --python)
      python_bin="${2:?Missing executable after --python}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$results_root" ]]; then
  case "$mode" in
    full) results_root="$SCRIPT_DIR/results/simulation-analysis-full" ;;
    check) results_root="$SCRIPT_DIR/results/simulation-analysis-check" ;;
    plot-only) results_root="$SCRIPT_DIR/results/simulation-analysis" ;;
  esac
fi

if ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable not found: $python_bin" >&2
  exit 1
fi
if ! command -v Rscript >/dev/null 2>&1; then
  echo "Rscript is required to generate the simulation figures." >&2
  exit 1
fi
if ! Rscript -e '
  required <- c("ggplot2", "cowplot")
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) stop("missing R package(s): ", paste(missing, collapse = ", "))
' >/dev/null; then
  echo "The simulation figures require the R packages ggplot2 and cowplot." >&2
  exit 1
fi

if [[ "$results_root" != /* ]]; then
  results_root="$SCRIPT_DIR/$results_root"
fi

if [[ "$mode" == "full" && "$force" -eq 0 && -d "$results_root" &&
      -d "$SCRIPT_DIR/results/simulation-analysis" &&
      "$(cd "$results_root" && pwd -P)" == "$(cd "$SCRIPT_DIR/results/simulation-analysis" && pwd -P)" ]]; then
  echo "full cannot replace the bundled simulation results. Choose a different --results-root." >&2
  exit 2
fi

scripts=(
  "study-scripts/01_gaussianity_mle.py"
  "study-scripts/02_kef_minimum_ksd.py"
  "study-scripts/02b_kef_score_matching.py"
  "study-scripts/03_conditional_gaussian_score_matching.py"
  "study-scripts/04_rbm_score_matching.py"
  "study-scripts/05_sphere_watson.py"
)
slugs=(
  "01-gaussianity-mle"
  "02-kef-minimum-ksd"
  "02b-kef-score-matching"
  "03-conditional-gaussian-score-matching"
  "04-rbm-score-matching"
  "05-sphere-watson"
)

numerical_output_is_complete() {
  local slug="$1"
  local directory="$2"
  local metadata="$directory/run_metadata.csv"
  local name
  if [[ "$slug" == "02b-kef-score-matching" ]]; then
    [[ -f "$directory/replications.csv" && -f "$directory/summary.csv" && -f "$metadata" ]] || return 1
    metadata_value_is "$metadata" schema_version kef-score-matching-nominal-v1 &&
      metadata_value_is "$metadata" study "$slug" &&
      metadata_value_is "$metadata" workflow simulate &&
      metadata_value_is "$metadata" status complete &&
      metadata_value_is "$metadata" run_complete true
    return
  fi
  if [[ -f "$metadata" ]] && metadata_value_is "$metadata" schema_version saved-csv-v1; then
    metadata_value_is "$metadata" workflow saved-results &&
      metadata_value_is "$metadata" study "$slug" &&
      metadata_value_is "$metadata" run_complete true || return 1
    [[ -f "$directory/summary.csv" ]] || return 1
    if [[ "$slug" != "01-gaussianity-mle" ]]; then
      [[ -f "$directory/replications.csv" && -f "$directory/source_metadata.csv" ]] || return 1
    fi
    return 0
  fi
  for name in replications.csv summary.csv run_metadata.csv; do
    [[ -f "$directory/$name" ]] || return 1
  done
  case "$slug" in
    01-gaussianity-mle)
      metadata_value_is "$metadata" status complete &&
        { metadata_value_is "$metadata" mode simulate ||
          metadata_value_is "$metadata" mode exact-paper; }
      ;;
    02-kef-minimum-ksd)
      if metadata_value_is "$metadata" schema_version semiparametric_ksd_kef_minimum_ksd_fresh_v1; then
        metadata_value_is "$metadata" status complete &&
          metadata_value_is "$metadata" workflow simulate
      else
        # Compatibility with already-saved tables only; no source imports.
        metadata_value_is "$metadata" schema_version semiparametric_ksd_kef_minimum_ksd_paired_aggregate_v2 ||
          metadata_value_is "$metadata" schema_version semiparametric_ksd_kef_minimum_ksd_v1
      fi
      ;;
    03-conditional-gaussian-score-matching)
      metadata_value_is "$metadata" run_complete true
      ;;
    04-rbm-score-matching)
      # RBM publishes replications.csv only after completion. Its plotter
      # validates the saved hashes and the full summary grid before plotting.
      return 0
      ;;
    05-sphere-watson)
      metadata_value_is "$metadata" study "$slug" &&
        metadata_value_is "$metadata" status complete &&
        metadata_value_is "$metadata" run_complete true || return 1
      if metadata_value_is "$metadata" schema_version sphere-watson-fresh-csv-v2; then
        metadata_value_is "$metadata" workflow simulate
      elif metadata_value_is "$metadata" schema_version sphere-watson-saved-csv-v1 ||
           metadata_value_is "$metadata" schema_version sphere-watson-saved-csv-v2; then
        metadata_value_is "$metadata" workflow saved-results &&
          [[ -f "$directory/source_metadata.csv" ]]
      else
        return 1
      fi
      ;;
    *) return 1 ;;
  esac
}

fresh_output_is_compatible() {
  local slug="$1"
  local metadata="$2"
  metadata_value_is "$metadata" schema_version saved-csv-v1 && return 1
  case "$slug" in
    01-gaussianity-mle)
      metadata_value_is "$metadata" mode simulate
      ;;
    02-kef-minimum-ksd)
      metadata_value_is "$metadata" schema_version semiparametric_ksd_kef_minimum_ksd_fresh_v1 &&
        metadata_value_is "$metadata" workflow simulate
      ;;
    02b-kef-score-matching)
      metadata_value_is "$metadata" schema_version kef-score-matching-nominal-v1 &&
        metadata_value_is "$metadata" study "$slug" &&
        metadata_value_is "$metadata" workflow simulate
      ;;
    05-sphere-watson)
      metadata_value_is "$metadata" schema_version sphere-watson-fresh-csv-v2 &&
        metadata_value_is "$metadata" workflow simulate
      ;;
    03-conditional-gaussian-score-matching)
      metadata_value_is "$metadata" schema_version 1 &&
        metadata_has_value "$metadata" run_fingerprint
      ;;
    04-rbm-score-matching)
      metadata_value_is "$metadata" schema_version 3 &&
        metadata_has_value "$metadata" run_fingerprint
      ;;
    *) return 0 ;;
  esac
}

# Check every study before dispatching work. Missing plot inputs never trigger
# a fallback simulation, even with --force or --skip-check.
for index in "${!scripts[@]}"; do
  script="$SIMULATION_CODE/${scripts[$index]}"
  output_dir="$results_root/${slugs[$index]}"
  metadata="$output_dir/run_metadata.csv"
  if [[ ! -f "$script" ]]; then
    echo "Missing simulation entry point: $script" >&2
    exit 1
  fi
  if [[ "$mode" == "plot-only" ]]; then
    if ! numerical_output_is_complete "${slugs[$index]}" "$output_dir"; then
      echo "plot-only requires completed simulation results in $output_dir (replications.csv, summary.csv, and run_metadata.csv). No simulation was started." >&2
      exit 2
    fi
  elif ((!force)) && [[ -f "$metadata" ]]; then
    if ! fresh_output_is_compatible "${slugs[$index]}" "$metadata"; then
      echo "$output_dir is not a fresh simulation checkpoint. Use plot-only for completed saved CSV results, or a new --results-root for simulation. Use --force only after reviewing outputs to replace." >&2
      exit 2
    fi
  fi
done

for index in "${!scripts[@]}"; do
  script="$SIMULATION_CODE/${scripts[$index]}"
  output_dir="$results_root/${slugs[$index]}"
  command=("$python_bin" "$script" --output-dir "$output_dir")
  if [[ "${slugs[$index]}" == "02-kef-minimum-ksd" ]]; then
    command+=(--require-plot)
  fi
  if [[ "$mode" == "plot-only" ]]; then
    command+=(--plot-only)
  elif ((!force)) && [[ "$mode" == "check" ]] && numerical_output_is_complete "${slugs[$index]}" "$output_dir"; then
    echo "Reusing completed numerical output in $output_dir"
    command+=(--plot-only)
  else
    if ((force)) || { [[ "$mode" == "full" ]] && numerical_output_is_complete "${slugs[$index]}" "$output_dir"; }; then
      command+=(--force)
    elif [[ -f "$output_dir/checkpoint.csv" ]] ||
        [[ -d "$output_dir/checkpoints" ]] ||
        { [[ -f "$output_dir/run_metadata.csv" ]] &&
          metadata_value_is "$output_dir/run_metadata.csv" workflow simulate &&
          metadata_value_is "$output_dir/run_metadata.csv" status running; } ||
        [[ "${slugs[$index]}" == "05-sphere-watson" && -f "$output_dir/run_metadata.csv" ]] ||
        { [[ -f "$output_dir/replications.csv" ]] &&
          [[ -f "$output_dir/run_metadata.csv" ]]; }; then
      command+=(--resume)
    fi
  fi
  if [[ "$mode" == "check" ]]; then
    command+=(--check)
  fi

  echo
  echo "==> ${slugs[$index]} ($mode)"
  "${command[@]}"
done

if ((!skip_check)); then
  "$python_bin" "$SCRIPT_DIR/check_outputs.py" --results-root "$results_root"
fi

echo
echo "All six simulation reproductions completed ($mode)."
