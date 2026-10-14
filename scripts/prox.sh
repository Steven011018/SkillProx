# Stage 3 — Backward Prox on the forward skill from stage 2.
#
# First runs a frozen leave-one-out utility audit of every L2 section and L3
# reference group on the validation split. Then a Shrinker consolidates,
# demotes, or removes units with u_cell < -0.001, lowest utility first. A trial
# is committed only if it stays structurally valid, is strictly shorter, keeps
# validation hard accuracy (tol 0) and loses at most 0.02 mean cell accuracy.
# No further trial is accepted once cumulative compression reaches 0.10.
# The final skill is written to
#   results/runs/<run_id>/prox/skill_prox/xlsx/
#
# Usage:
#   bash scripts/prox.sh                   # default backbone: gpt-5.4
#   MODEL=gpt-4.1 bash scripts/prox.sh

set -e

MODEL="${MODEL:-gpt-5.4}"
RUN_ID="skillprox_${MODEL}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
RUN_DIR="${RESULTS_ROOT}/runs/${RUN_ID}"

python -m pipeline.utility \
    --skill-dir ${RUN_DIR}/train/final_skill \
    --out-dir ${RUN_DIR}/utility_audit \
    --model ${MODEL} \
    --results-root ${RESULTS_ROOT} \
    --master-seed 0 --heldout-seed 42 \
    --eps-cell 0.025 \
    --executor-concurrency 4 --grader-concurrency 1

python -m pipeline.shrinkage \
    --skill-dir ${RUN_DIR}/train/final_skill \
    --audit ${RUN_DIR}/utility_audit/utility_audit.json \
    --out-dir ${RUN_DIR}/prox \
    --model ${MODEL} \
    --results-root ${RESULTS_ROOT} \
    --master-seed 0 --heldout-seed 42 \
    --hard-tol 0 --cell-tol 0.02 \
    --min-neg-utility -0.001 --max-compression 0.10 \
    --executor-concurrency 4
