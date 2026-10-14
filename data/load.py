"""
Dataset loading, naming normalization, and splitting for SpreadsheetBench.

Handles both sample_data_200 (input/answer naming, 3 test cases) and
spreadsheetbench_verified_400 (init/golden naming, 1 test case).

Split design (for paper experiments):
    Step 1: shuffle all valid samples with seed, take last n_test as fixed test
    Step 2: from remaining, take last n_val as fixed val (optional;
            SkillGrad does not use a validation set itself).
    Step 3: from remaining pool, take first n_train for training.

Usage:
    # As a library
    from data import SpreadsheetBenchDataset
    ds = SpreadsheetBenchDataset("data/spreadsheetbench/spreadsheetbench_verified_400")
    split = ds.split(n_train=40, n_test=100, n_val=20, seed=0)
    normalized_dir = ds.prepare_for_pipeline(Path("results/split_seed0/normalized"))

    # As a CLI (create split + normalize)
    python data.py \\
      --data-dir data/spreadsheetbench/spreadsheetbench_verified_400 \\
      --n-train 40 --n-test 100 --n-val 20 --seed 0 \\
      --output-dir results/split_seed0
"""

import argparse
import json
import os
import random
from pathlib import Path


class SpreadsheetBenchDataset:
    """Load and normalize any SpreadsheetBench dataset variant."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.dataset_path = self.data_dir / "dataset.json"
        with open(self.dataset_path, encoding="utf-8") as f:
            self.dataset = json.load(f)
        self.spreadsheet_dir = self.data_dir / "spreadsheet"

        # Build path index: detect naming convention per sample
        self._path_index: dict[str, dict] = {}
        for sample in self.dataset:
            sid = str(sample["id"])
            sample_dir = self.spreadsheet_dir / sid
            if not sample_dir.exists():
                continue
            files = os.listdir(sample_dir)
            self._path_index[sid] = self._detect_naming(sid, files)

    def _detect_naming(self, sid: str, files: list[str]) -> dict:
        """Detect the naming convention for a sample's files."""
        info = {"dir": self.spreadsheet_dir / sid, "test_cases": 0, "naming": "unknown"}

        # Pattern 1: {tc}_{id}_input.xlsx / {tc}_{id}_answer.xlsx (sample_data_200)
        input_files = sorted(f for f in files if f.endswith("_input.xlsx"))
        if input_files:
            info["naming"] = "input_answer"
            info["test_cases"] = len(input_files)
            return info

        # Pattern 2: {tc}_{id}_init.xlsx / {tc}_{id}_golden.xlsx (verified_400 main)
        init_files = sorted(f for f in files if f.endswith("_init.xlsx"))
        if init_files:
            info["naming"] = "init_golden"
            info["test_cases"] = len(init_files)
            return info

        # Pattern 3: initial.xlsx / golden.xlsx (verified_400 outliers)
        if "initial.xlsx" in files and "golden.xlsx" in files:
            info["naming"] = "bare"
            info["test_cases"] = 1
            return info

        return info

    def get_input_path(self, sample_id: str, test_case: int = 1) -> Path | None:
        """Resolve input file path regardless of naming convention."""
        info = self._path_index.get(str(sample_id))
        if not info:
            return None
        d = info["dir"]
        if info["naming"] == "input_answer":
            return d / f"{test_case}_{sample_id}_input.xlsx"
        elif info["naming"] == "init_golden":
            return d / f"{test_case}_{sample_id}_init.xlsx"
        elif info["naming"] == "bare":
            return d / "initial.xlsx"
        return None

    def get_answer_path(self, sample_id: str, test_case: int = 1) -> Path | None:
        """Resolve answer/golden file path."""
        info = self._path_index.get(str(sample_id))
        if not info:
            return None
        d = info["dir"]
        if info["naming"] == "input_answer":
            return d / f"{test_case}_{sample_id}_answer.xlsx"
        elif info["naming"] == "init_golden":
            expected = d / f"{test_case}_{sample_id}_golden.xlsx"
            if expected.exists():
                return expected
            # Fallback: find any golden file for this test case (handles mismatched IDs)
            for f in d.iterdir():
                if f.name.startswith(f"{test_case}_") and f.name.endswith("_golden.xlsx"):
                    return f
            return expected
        elif info["naming"] == "bare":
            return d / "golden.xlsx"
        return None

    def get_test_case_count(self, sample_id: str) -> int:
        info = self._path_index.get(str(sample_id))
        return info["test_cases"] if info else 0

    def __len__(self) -> int:
        return len(self.dataset)

    def split(
        self, n_train: int, n_test: int, n_val: int = 0, seed: int = 0,
    ) -> dict:
        """Split dataset into train / val / test returning INDICES (not IDs).

        Split procedure (deterministic given seed):
          1. Shuffle all valid indices with seed
          2. Last n_test → fixed test set
          3. Next-to-last n_val → fixed validation set (optional; unused by SkillGrad)
          4. From the remaining pool, first n_train → training set

        This ensures:
          - Test set is always the same for a given seed (independent of n_train)
          - Val set is always the same for a given seed (independent of n_train)
          - Training size can vary for ablation without affecting test/val
          - All methods reading this split see the same train seeds

        Returns dict with train_indices, test_indices, val_indices, pool_indices.
        """
        valid_indices = [
            i for i, s in enumerate(self.dataset)
            if str(s["id"]) in self._path_index
            and self._path_index[str(s["id"])]["test_cases"] > 0
        ]

        total_needed = n_train + n_val + n_test
        if total_needed > len(valid_indices):
            raise ValueError(
                f"Requested {n_train}+{n_val}+{n_test}={total_needed} but only "
                f"{len(valid_indices)} valid samples available"
            )

        rng = random.Random(seed)
        shuffled = list(valid_indices)
        rng.shuffle(shuffled)

        # Split from the END so test/val are stable across n_train changes
        test_indices = sorted(shuffled[-n_test:])
        val_indices = sorted(shuffled[-(n_test + n_val):-n_test]) if n_val > 0 else []
        pool = shuffled[:-(n_test + n_val)] if n_val > 0 else shuffled[:-n_test]
        train_indices = sorted(pool[:n_train])
        remaining_pool = sorted(pool[n_train:])

        return {
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
            "pool_indices": remaining_pool,  # unused training pool
        }

    def prepare_for_pipeline(self, output_dir: Path) -> Path:
        """Create a normalized directory with input/answer naming for evolve.py.

        Creates symlinks so evolve.py's prepare_seed_data() works unchanged:
          {output_dir}/dataset.json
          {output_dir}/spreadsheet/{id}/1_{id}_input.xlsx  -> original init file
          {output_dir}/spreadsheet/{id}/1_{id}_answer.xlsx -> original golden file

        Returns output_dir (to pass as data_dir to evolve.py).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy dataset.json
        out_dataset = output_dir / "dataset.json"
        if not out_dataset.exists():
            with open(out_dataset, "w", encoding="utf-8") as f:
                json.dump(self.dataset, f, indent=2, ensure_ascii=False)

        # Create normalized symlinks for each sample
        out_spreadsheet = output_dir / "spreadsheet"
        out_spreadsheet.mkdir(exist_ok=True)

        for sample in self.dataset:
            sid = str(sample["id"])
            if sid not in self._path_index:
                continue

            info = self._path_index[sid]
            sample_out = out_spreadsheet / sid
            sample_out.mkdir(exist_ok=True)

            for tc in range(1, info["test_cases"] + 1):
                input_src = self.get_input_path(sid, tc)
                answer_src = self.get_answer_path(sid, tc)

                if input_src and input_src.exists():
                    link = sample_out / f"{tc}_{sid}_input.xlsx"
                    if not link.exists():
                        link.symlink_to(input_src.resolve())

                if answer_src and answer_src.exists():
                    link = sample_out / f"{tc}_{sid}_answer.xlsx"
                    if not link.exists():
                        link.symlink_to(answer_src.resolve())

        return output_dir


def main():
    parser = argparse.ArgumentParser(description="Prepare SpreadsheetBench data split")
    parser.add_argument("--data-dir", required=True, help="Path to dataset")
    parser.add_argument("--n-train", type=int, default=40,
                        help="Training set size (default: 40, 10%% of 400)")
    parser.add_argument("--n-test", type=int, default=100,
                        help="Test set size (fixed, default: 100)")
    parser.add_argument("--n-val", type=int, default=20,
                        help="Validation set size (fixed, default: 20; unused by SkillGrad)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True, help="Output directory")
    args = parser.parse_args()

    ds = SpreadsheetBenchDataset(args.data_dir)
    print(f"Loaded {len(ds)} samples from {args.data_dir}")
    print(f"Valid samples with files: {len(ds._path_index)}")

    split = ds.split(args.n_train, args.n_test, args.n_val, args.seed)
    print(f"Train: {len(split['train_indices'])} | "
          f"Val: {len(split['val_indices'])} | "
          f"Test: {len(split['test_indices'])} | "
          f"Unused pool: {len(split['pool_indices'])}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save split
    split_info = {
        "seed": args.seed,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "n_val": args.n_val,
        "train_indices": split["train_indices"],
        "val_indices": split["val_indices"],
        "test_indices": split["test_indices"],
        "pool_indices": split["pool_indices"],
        "train_ids": [str(ds.dataset[i]["id"]) for i in split["train_indices"]],
        "val_ids": [str(ds.dataset[i]["id"]) for i in split["val_indices"]],
        "test_ids": [str(ds.dataset[i]["id"]) for i in split["test_indices"]],
    }
    with open(output_dir / "split.json", "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Split saved to {output_dir / 'split.json'}")

    # Normalize dataset
    normalized_dir = ds.prepare_for_pipeline(output_dir / "normalized")
    print(f"Normalized dataset at {normalized_dir}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Seed:    {args.seed}")
    print(f"  Train:   {len(split['train_indices'])} seeds ({args.n_train/4:.0f}%)")
    print(f"  Val:     {len(split['val_indices'])} seeds (fixed, optional)")
    print(f"  Test:    {len(split['test_indices'])} seeds (fixed, held-out)")
    print(f"  Pool:    {len(split['pool_indices'])} seeds (unused, for ablation)")
    print(f"{'='*60}")
    print(f"\nFor 5%  training: rerun with --n-train 20  (test/val unchanged)")
    print(f"For 20% training: rerun with --n-train 80  (test/val unchanged)")
    print(f"\nTrain IDs (first 10): {split_info['train_ids'][:10]}")
    print(f"Val IDs:              {split_info['val_ids'][:10]}")
    print(f"Test IDs (first 10):  {split_info['test_ids'][:10]}")


if __name__ == "__main__":
    main()
