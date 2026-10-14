# Stage 0 — Build the canonical splits.
#
# 1. SpreadsheetBench Verified (400 tasks): 200-task evolution pool, 20-task
#    validation split (used by the Prox utility audit and gate) and 100-task
#    test split, written under results/.
# 2. OJ evaluation set: the same 100 test tasks taken from SpreadsheetBench
#    v0.1 (912 tasks, three test cases each), written under results_oj/.
#    Skip this step if you only need single-test-case evaluation.
#
# Usage:
#   bash scripts/prepare.sh

set -e

DATA_DIR="${DATA_DIR:-data/benchmarks/spreadsheetbench}"
OJ_DATA_DIR="${OJ_DATA_DIR:-data/benchmarks/spreadsheetbench_912}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
OJ_RESULTS_ROOT="${OJ_RESULTS_ROOT:-results_oj}"

python -m data.split split \
    --data-dir ${DATA_DIR} \
    --results-root ${RESULTS_ROOT} \
    --master-seed 0 --heldout-seed 42

if [ -d "${OJ_DATA_DIR}" ]; then
    python -m scripts.build_oj_split \
        --oj-data-dir ${OJ_DATA_DIR} \
        --results-root ${RESULTS_ROOT} \
        --oj-results-root ${OJ_RESULTS_ROOT} \
        --master-seed 0 --heldout-seed 42
else
    echo "No ${OJ_DATA_DIR}; skipping the OJ split."
fi
