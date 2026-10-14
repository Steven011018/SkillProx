# Stage 4 — Evaluate the final SkillProx skill on the held-out test split.
#
# Default is the OJ protocol reported in the paper: 100 test tasks from
# results_oj/ (built by scripts/prepare.sh), three test cases each. The
# executor writes one reusable solution.py on test case 1; the evaluator runs
# it unchanged on all three cases, and a task is hard-correct only if every
# case is cell-perfect. Per-task trajectories and a `summary.json` are written
# to the output directory.
#
# Usage:
#   bash scripts/eval.sh                          # default backbone: gpt-5.4
#   MODEL=gpt-4.1 bash scripts/eval.sh
#   OJ=0 bash scripts/eval.sh                     # single test case, Verified data
#   SKILL_DIR=results/runs/skillprox_gpt-5.4/train/final_skill bash scripts/eval.sh
#                                                 # forward skill only, no Prox

MODEL="${MODEL:-gpt-5.4}"
RUN_ID="skillprox_${MODEL}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
OJ_RESULTS_ROOT="${OJ_RESULTS_ROOT:-results_oj}"
OJ="${OJ:-1}"
SKILL_DIR="${SKILL_DIR:-${RESULTS_ROOT}/runs/${RUN_ID}/prox/skill_prox}"

if [ "${OJ}" = "1" ]; then
    OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/runs/${RUN_ID}/eval_oj}"
    python -m runners.stream_runner eval \
        --skill-dir ${SKILL_DIR} \
        --output-dir ${OUTPUT_DIR} \
        --model ${MODEL} \
        --results-root ${OJ_RESULTS_ROOT} \
        --master-seed 0 --heldout-seed 42 --split test_indices \
        --test-cases 3 --test-case-start 1 --oj \
        --executor-concurrency 4 --grader-concurrency 1
else
    OUTPUT_DIR="${OUTPUT_DIR:-${RESULTS_ROOT}/runs/${RUN_ID}/eval}"
    python -m runners.stream_runner eval \
        --skill-dir ${SKILL_DIR} \
        --output-dir ${OUTPUT_DIR} \
        --model ${MODEL} \
        --results-root ${RESULTS_ROOT} \
        --master-seed 0 --heldout-seed 42 \
        --executor-concurrency 4 --grader-concurrency 1
fi
