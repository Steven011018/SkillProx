"""Diagnoser prompts."""

FAILURE_DIAGNOSER_PROMPT = """\
You analyze a failed task execution to understand what went wrong and why.

An agent was given a spreadsheet task and produced an incorrect output. \
You receive the task description, a cell-by-cell comparison showing \
where the output differs from the expected answer, and file paths to \
the agent's execution trace and output files. Your job is to figure out:

1. What the output got wrong (from the cell comparison).
2. What code or approach the agent used that led to the wrong output \
(from reading the execution trace).
3. What approach would have produced the correct output.
4. Whether the agent skipped a reasoning step that would have \
prevented the failure — for example, started coding before \
inspecting the workbook, wrote code without stating the intended \
transformation, or saved the output without checking the target \
cells. If you observe such a skip, describe it. \

Write your analysis inside a <diagnosis> block. Start with a LABEL — \
a short phrase (3-6 words) naming the general type of error, not the \
specific task. Then write your analysis in plain prose.

# Example

Task: "Fill column B with the running total of column A values."
Cell comparison: B2 expected 10.0, got None; B3 expected 25.0, got None.

<diagnosis>
LABEL: Unevaluated formula output

The agent wrote Excel formulas (=SUM($A$2:A2)) into column B and \
filled them down. However, the library used to save the file \
(openpyxl) writes formula strings but does not calculate them. The \
saved file contains formula text, not computed values, so all target \
cells appear as None when read back.

The correct approach: compute the running totals in Python (e.g., \
maintain a running sum while iterating rows) and write the resulting \
numbers directly into each cell.
</diagnosis>

# MUST NOT
- Do not use task-specific values (exact column letters, cell \
references, domain terms) in the LABEL. The label should make sense \
for any similar task.
- Do not prescribe specific changes to the agent's skill document. \
Focus on what happened and what should have happened.\
"""

CONTRASTIVE_DIAGNOSER_PROMPT = """\
You compare two executions of the same task to understand what enabled \
the second one to succeed where the first one failed.

An agent attempted a spreadsheet task twice. \
The first attempt failed; the second attempt succeeded. \
You receive the task description, a cell comparison showing what was \
wrong in the first attempt, and file paths to both execution traces. \
Your job is to figure out:

1. What the agent did wrong in the first attempt (from the cell \
comparison and the failed trace).
2. What the agent did differently in the second attempt that produced \
the correct output (from comparing the two traces).
3. Whether the success seems robust or fragile — would the same \
approach work on similar tasks, or did the agent get lucky?
4. Whether the success followed a reasoning step that the failure \
skipped. For example, the success inspected the workbook structure \
before coding while the failure jumped straight to coding, or the \
success verified the saved output while the failure didn't. If you \
observe such a difference, describe it. \

Write your analysis inside a <diagnosis> block. Start with a LABEL — \
a short phrase (3-6 words) naming the general type of knowledge that \
enabled the success. Then write your analysis in plain prose.

# Example

Task: "List distinct product names from column C in column F."
Failed run cell comparison: F2 expected "Widget", got None; F3 \
expected "Gadget", got None.

<diagnosis>
LABEL: Value computation over formula

In the failed execution, the agent wrote =UNIQUE(C:C) as a formula. \
The file-saving library does not evaluate formulas, so all cells in \
column F were blank.

In the successful execution, the agent iterated column C in Python, \
collected unique values using a set, and wrote them as literal strings \
into column F. This approach does not depend on formula evaluation.

The success is robust: computing values in Python and writing them \
directly avoids the formula evaluation issue entirely, and this \
approach generalizes to any task that needs computed results in cells.
</diagnosis>

# MUST NOT
- Do not use task-specific values in the LABEL.
- Do not prescribe specific changes to the agent's skill document.\
"""
