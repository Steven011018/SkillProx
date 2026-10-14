"""Build the OJ evaluation split used for the headline SkillProx results.

The OJ protocol scores a skill on the same 100 held-out test tasks as the
SpreadsheetBench Verified split, but takes each task from the 912-task
SpreadsheetBench v0.1 release, which ships three test cases per task. The
executor writes one reusable `solution.py` on test case 1; the evaluator
then runs that script unchanged on all three cases, and a task is
hard-correct only if every case is cell-perfect.

This script writes
  <oj-results-root>/normalized/                         (symlinks)
  <oj-results-root>/splits/master_<M>_heldout_<H>/split.json
from the Verified split at
  <results-root>/splits/master_<M>_heldout_<H>/split.json

Usage:
    python -m scripts.build_oj_split \\
        --oj-data-dir data/benchmarks/spreadsheetbench_912 \\
        --results-root results --oj-results-root results_oj \\
        --master-seed 0 --heldout-seed 42
"""

import argparse
import json
from pathlib import Path

from data.layout import normalized_dir_for, splits_dir_for
from data.load import SpreadsheetBenchDataset
from data.split import load_split


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--oj-data-dir", required=True,
                    help="Raw SpreadsheetBench v0.1 (912 tasks, 3 test cases).")
    ap.add_argument("--results-root", default="results",
                    help="Results root holding the Verified split.")
    ap.add_argument("--oj-results-root", default="results_oj")
    ap.add_argument("--master-seed", type=int, default=0)
    ap.add_argument("--heldout-seed", type=int, default=42)
    ap.add_argument("--test-cases", type=int, default=3)
    args = ap.parse_args()

    verified = load_split(
        splits_dir_for(Path(args.results_root), args.master_seed, args.heldout_seed)
    )
    test_ids = list(verified["test_ids"])

    oj_root = Path(args.oj_results_root)
    ds = SpreadsheetBenchDataset(args.oj_data_dir)
    normalized = ds.prepare_for_pipeline(normalized_dir_for(oj_root))
    dataset = json.loads((normalized / "dataset.json").read_text(encoding="utf-8"))
    id_to_idx = {str(s["id"]): i for i, s in enumerate(dataset)}

    missing = [t for t in test_ids if str(t) not in id_to_idx]
    if missing:
        raise ValueError(f"{len(missing)} Verified test ids absent from the OJ release: {missing[:10]}")
    short = [t for t in test_ids if ds.get_test_case_count(str(t)) < args.test_cases]
    if short:
        raise ValueError(
            f"{len(short)} test tasks have fewer than {args.test_cases} test cases: {short[:10]}"
        )

    split = {
        "master_seed": args.master_seed,
        "heldout_seed": args.heldout_seed,
        "n_evolution": 0,
        "n_test": len(test_ids),
        "n_val": 0,
        "source": f"912_v0.1 restricted to verified test-{len(test_ids)} "
                  f"({args.test_cases} test cases)",
        "data_dir": str(Path(args.oj_data_dir)),
        "evolution_indices": [],
        "evolution_ids": [],
        "test_indices": [id_to_idx[str(t)] for t in test_ids],
        "test_ids": test_ids,
        "val_indices": [],
        "val_ids": [],
    }
    out_dir = splits_dir_for(oj_root, args.master_seed, args.heldout_seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "split.json").write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"OJ split saved: {out_dir / 'split.json'} ({len(test_ids)} tasks)")


if __name__ == "__main__":
    main()
