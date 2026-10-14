# Stage 2 — Closed-loop forward evolution on the failure pool from stage 1.
#
# Each iteration samples a mini-batch, executes the current skill, diagnoses
# failures and contrastive successes, updates momentum, and patches the skill.
# Every candidate patch is re-executed on the same batch. The first candidate
# with a strict hard-accuracy gain is taken; otherwise up to --cl-max-attempts
# candidates are tried and the best by (hard, cell) is kept only if neither
# metric regresses. Rejected attempts are fed back to the next diagnosis.
# The forward skill is written to
#   results/runs/<run_id>/train/final_skill/xlsx/
#
# Usage:
#   bash scripts/train.sh                  # default backbone: gpt-5.4
#   MODEL=gpt-4.1 bash scripts/train.sh

MODEL="${MODEL:-gpt-5.4}"
SKILLS_DIR="${SKILLS_DIR:-seeds}"
DATA_DIR="${DATA_DIR:-data/benchmarks/spreadsheetbench}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"

python -m pipeline.training \
    --data-dir ${DATA_DIR} \
    --skills-dir ${SKILLS_DIR} \
    --results-root ${RESULTS_ROOT} \
    --method skillprox \
    --model ${MODEL} \
    --master-seed 0 --heldout-seed 42 --training-seed 0 \
    --n-train 40 --batch-size 4 --max-turns 30 --concurrency 4 \
    --closed-loop --cl-max-attempts 3 --cl-hard-tol 0 --cl-prior-k 6
