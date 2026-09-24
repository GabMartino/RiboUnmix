#!/usr/bin/env bash
# Regenerate identical per-run plots, then the matched equal/ranked comparison.
# No training, checkpoint loading, changes to pi, or ground-truth assumptions.
# Usage: bash analyses/run_real_panel_weighting_analysis.sh [equal_root] [ranked_root] [comparison_dir]
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_DIR}/.venv/bin/python}"
EQUAL_ROOT="${1:-${PROJECT_DIR}/results/my_panels_a100_b32_20260906_114323}"
RANKED_ROOT="${2:-${PROJECT_DIR}/results/my_panels_qrank_a100_b32_20260908_103510}"
COMPARISON_DIR="${3:-${PROJECT_DIR}/analyses/artifacts/real_data/panels_equal_vs_ranked}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${COMPARISON_DIR}/.matplotlib}"
mkdir -p "${COMPARISON_DIR}" "${MPLCONFIGDIR}"
if ! command -v latex >/dev/null || ! command -v dvipng >/dev/null; then
    echo "Install LaTeX, Latin Modern (lmodern), and dvipng; plots use real TeX, not substituted fonts." >&2
    exit 2
fi
for PANEL_ROOT in "${EQUAL_ROOT}" "${RANKED_ROOT}"; do
    PANEL_NAME="$(basename "${PANEL_ROOT}")"
    PANEL_ANALYSIS="${PROJECT_DIR}/analyses/artifacts/real_data/${PANEL_NAME}/panel_convergence"
    PANEL_FIGURE="${PROJECT_DIR}/analyses/artifacts/real_data/${PANEL_NAME}/publication_figure"
    mkdir -p "${PANEL_ANALYSIS}" "${PANEL_FIGURE}"
    echo "Regenerating per-run plots: ${PANEL_ROOT}"
    "${PYTHON_BIN}" -u analyses/analyze_real_panel_convergence.py \
        --run-root "${PANEL_ROOT}" --output-dir "${PANEL_ANALYSIS}" \
        > "${PANEL_ANALYSIS}/regenerate_plots.log" 2>&1
    if "${PYTHON_BIN}" -c 'import json,sys; m=json.load(open(sys.argv[1])); sys.exit(0 if m["analysis_complete_for_planned_panels"] else 1)' \
            "${PANEL_ANALYSIS}/analysis_manifest.json"; then
        "${PYTHON_BIN}" -u analyses/create_four_panel_reproducibility_figure.py \
            --run-root "${PANEL_ROOT}" --output-dir "${PANEL_FIGURE}" \
            > "${PANEL_FIGURE}/regenerate_publication_figure.log" 2>&1
    else
        echo "Partial run: per-run available-panel plots saved; the four-panel publication figure waits for all panels."
    fi
done
echo "Creating matched equal/ranked comparison: ${COMPARISON_DIR}"
"${PYTHON_BIN}" -u analyses/compare_real_panel_weighting.py --equal-root "${EQUAL_ROOT}" \
    --ranked-root "${RANKED_ROOT}" --output-dir "${COMPARISON_DIR}" \
    --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-5000}" \
    > "${COMPARISON_DIR}/analysis.log" 2>&1
echo "Done. PDF/PNG figures, source tables, provenance and README: ${COMPARISON_DIR}"
