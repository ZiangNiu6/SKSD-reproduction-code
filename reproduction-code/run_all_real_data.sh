#!/usr/bin/env bash

# Reproduce or replot the TCGA-LUAD real-data application in Section 6.2.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ANALYSIS_ROOT="$SCRIPT_DIR/real-data-analysis"
PACKAGE_ROOT="$ANALYSIS_ROOT/tcga_rppa"
PLOTTER="$ANALYSIS_ROOT/plotting-code/plot_tcga_rppa.R"
if [[ -f "$SCRIPT_DIR/renv/activate.R" ]]; then
  export RENV_PROJECT="$SCRIPT_DIR"
  export R_PROFILE_USER="$SCRIPT_DIR/.Rprofile"
fi

usage() {
  echo "Usage: $0 [full|plot-only|quick] [--results-root DIR] [--python PATH]"
  echo "full verifies source data and reruns both TCGA analyses every time."
  echo "plot-only uses the two bundled model_evaluation.csv tables and never starts a numerical analysis."
  echo "Default results: full uses results/real-data-analysis-full/tcga-rppa; plot-only uses results/real-data-analysis/tcga-rppa; quick uses results/real-data-analysis-quick/tcga-rppa."
}

mode="${1:-full}"
case "$mode" in
  full|plot-only|quick)
    if (($# > 0)); then shift; fi
    ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

results_root="${TCGA_RPPA_RESULTS_DIR:-}"
python_bin="${PYTHON_BIN:-python}"

while (($# > 0)); do
  case "$1" in
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
    full) results_root="$SCRIPT_DIR/results/real-data-analysis-full/tcga-rppa" ;;
    plot-only) results_root="$SCRIPT_DIR/results/real-data-analysis/tcga-rppa" ;;
    quick) results_root="$SCRIPT_DIR/results/real-data-analysis-quick/tcga-rppa" ;;
  esac
fi

if [[ "$results_root" != /* ]]; then
  results_root="$SCRIPT_DIR/$results_root"
fi

if [[ "$mode" != "plot-only" ]] && ! command -v "$python_bin" >/dev/null 2>&1; then
  echo "Python executable not found: $python_bin" >&2
  exit 1
fi
if ! command -v Rscript >/dev/null 2>&1; then
  echo "Rscript is required for the TCGA figures." >&2
  exit 1
fi
if ! Rscript -e '
  required <- c("ggplot2", "cowplot")
  missing <- required[!vapply(required, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) stop("missing R package(s): ", paste(missing, collapse = ", "))
' >/dev/null; then
  echo "The TCGA figures require the R packages ggplot2 and cowplot." >&2
  exit 1
fi

data_path="$PACKAGE_ROOT/data/derived/luad_rppa.csv"
publication_root="$results_root/outputs"
pathway_output="$publication_root/pathway-midpoint"
complete_output="$publication_root/complete-midpoint-b4999"

plot_output() {
  local output_dir
  for output_dir in "$@"; do
    if [[ ! -f "$output_dir/model_evaluation.csv" ]]; then
      echo "Missing numerical output: $output_dir/model_evaluation.csv" >&2
      exit 1
    fi
  done
  Rscript "$PLOTTER" "$@"
}

run_graph_analysis() {
  local graph="$1"
  local bootstrap="$2"
  local output_dir="$3"
  "$python_bin" "$PACKAGE_ROOT/run_analysis.py" \
    --data "$data_path" \
    --graph "$graph" \
    --bootstrap "$bootstrap" \
    --output-dir "$output_dir"
}

case "$mode" in
  quick)
    if [[ ! -f "$data_path" ]]; then
      "$python_bin" "$PACKAGE_ROOT/prepare_data.py"
    fi
    quick_output="$results_root/quick/pathway"
    run_graph_analysis pathway 19 "$quick_output"
    ;;
  full)
    "$python_bin" "$PACKAGE_ROOT/prepare_data.py"
    run_graph_analysis pathway 999 "$pathway_output"
    run_graph_analysis complete 4999 "$complete_output"
    ;;
  plot-only)
    plot_output "$pathway_output"
    plot_output "$complete_output"
    ;;
esac

if [[ "$mode" != "quick" ]]; then
  plot_output "$pathway_output" "$complete_output"
fi

echo "TCGA-LUAD real-data reproduction complete ($mode)."
