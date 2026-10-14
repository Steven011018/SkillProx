# Datasets

The training and evaluation scripts read SpreadsheetBench Verified from
`data/benchmarks/spreadsheetbench/`. The dataset is not
bundled with this repository.

## Expected layout

```
data/benchmarks/spreadsheetbench/
├── dataset.json
└── spreadsheet/
    ├── <sample_id>/
    │   ├── 1_<sample_id>_init.xlsx       # input workbook
    │   ├── 1_<sample_id>_golden.xlsx     # reference answer
    │   └── ...                            # additional test cases (if any)
    └── ...
```

`data/load.py` accepts the three naming conventions that ship in
SpreadsheetBench releases (`*_input/_answer`, `*_init/_golden`, and the
bare `initial.xlsx/golden.xlsx` outliers), so any of the official
distributions will work.

## Fetching the data

SpreadsheetBench is released by Ma et al. (2024). Follow the upstream
instructions to download the verified subset and place the files under
`data/benchmarks/spreadsheetbench/` as shown above.

- Upstream: <https://spreadsheetbench.github.io/>
- The "verified-400" subset used in the
  [SkillProx paper](https://arxiv.org/abs/2608.07449) covers 400 tasks;
  the canonical 200/200 evolution/held-out split is produced by
  `data/split.py` from a fixed `--master-seed` / `--heldout-seed` pair.

## OJ evaluation set

The headline results in the
[paper](https://arxiv.org/abs/2608.07449) use an OJ protocol: the same
100 held-out test tasks, taken from the full SpreadsheetBench v0.1 release
(912 tasks), which ships three test cases per task. Place that release under
`data/benchmarks/spreadsheetbench_912/` with the same layout as above, then
run `bash scripts/prepare.sh`. `scripts/build_oj_split.py` writes the OJ
split to `results_oj/splits/master_0_heldout_42/split.json`.
