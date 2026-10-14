"""
Data split and training set construction.

Split design:
    - 200 held-out + 200 evolution pool (master_seed)
    - Inside the 200 held-out: 20 val + 180 test_pool (heldout_seed)
    - Run base skill on the 200 evolution pool once per executor model
    - From failures (~50), select 40 as training set (training_seed, varies)

The 200/200 split and base trajectories are master-seed invariant; the
val/test sub-shuffle is heldout-seed invariant. Only the training set
selection varies with training_seed, which is the variance source for
mean±std reporting.

Outputs land under the canonical layout (see ``data/layout.py``):

  results/normalized/                                 — symlinks to data
  results/splits/master_<M>_heldout_<H>/split.json    — the canonical split

Usage:
    # Create / refresh the canonical split for (master=0, heldout=42)
    python -m data.split split \\
        --data-dir data/benchmarks/spreadsheetbench \\
        --master-seed 0 --heldout-seed 42

    # Inspect a training set selection (does not write to disk; training.py
    # writes the per-run train_set.json inside the run directory)
    python -m data.split training-set \\
        --master-seed 0 --heldout-seed 42 --model gpt-5.4 \\
        --training-seed 0 --n-train 40 --batch-size 4
"""

import argparse
import json
import random
from pathlib import Path

from data.load import SpreadsheetBenchDataset
from data.layout import (
    base_trajectories_dir_for,
    normalized_dir_for,
    splits_dir_for,
)


def create_split(
    data_dir: str | Path,
    output_dir: str | Path,
    master_seed: int = 0,
    heldout_seed: int = 42,
    n_evolution: int = 200,
    n_test: int = 180,
    n_val: int = 20,
    normalized_dir: str | Path | None = None,
) -> dict:
    """Create a fixed evolution / val / test split.

    Two-stage split:
      Stage 1 (master_seed): 200 evolution + 200 held-out.
      Stage 2 (heldout_seed): inside the 200 held-out,
        val      = first n_val (FIXED at 20).
        test_pool = remaining 180 (FIXED order).
        test     = test_pool[:n_test] (prefix-stable).

    Prefix stability: enlarging n_test only ADDS tasks to the test_ids
    list — the existing test_ids are preserved. So evaluation results
    on a smaller n_test can be reused when n_test grows.

    Args:
        data_dir: Path to SpreadsheetBench dataset.
        output_dir: Where to save split.json.
        master_seed: Controls the 200/200 split.
        heldout_seed: Controls the val/test_pool shuffle inside the
            200 held-out tasks. Independent of master_seed.
        n_evolution: Size of evolution pool.
        n_test: Size of held-out test set (selected as a prefix of
            test_pool, length 180).
        n_val: Size of held-out validation set (FIXED at first n_val
            after the heldout_seed shuffle).

    Returns:
        Split info dict.
    """
    ds = SpreadsheetBenchDataset(data_dir)
    valid_indices = [
        i for i, s in enumerate(ds.dataset)
        if str(s["id"]) in ds._path_index
        and ds._path_index[str(s["id"])]["test_cases"] > 0
    ]
    print(f"Loaded {len(ds)} samples, {len(valid_indices)} valid")

    if n_evolution + 200 > len(valid_indices):
        raise ValueError(
            f"Requested {n_evolution} evolution + 200 held-out = "
            f"{n_evolution + 200} but only {len(valid_indices)} valid samples"
        )
    if n_test + n_val > 200:
        raise ValueError(
            f"n_test ({n_test}) + n_val ({n_val}) = {n_test + n_val} "
            f"> 200 (held-out pool size)"
        )

    # Stage 1: 200 evolution + 200 held-out
    rng = random.Random(master_seed)
    shuffled = list(valid_indices)
    rng.shuffle(shuffled)

    heldout_pool = shuffled[:200]
    evolution_indices = sorted(shuffled[200:200 + n_evolution])

    # Stage 2: inside the 200 held-out, val = first n_val (FIXED),
    # test = prefix of the remaining test_pool (FIXED order).
    rng2 = random.Random(heldout_seed)
    rng2.shuffle(heldout_pool)

    val_pool_indices = heldout_pool[:n_val]            # FIXED 20, in shuffle order
    test_pool_indices = heldout_pool[n_val:]           # FIXED 180, in shuffle order
    test_selection = test_pool_indices[:n_test]        # prefix-stable

    val_indices = sorted(val_pool_indices)
    test_indices = sorted(test_selection)

    split_info = {
        "master_seed": master_seed,
        "heldout_seed": heldout_seed,
        "n_evolution": len(evolution_indices),
        "n_test": len(test_indices),
        "n_val": len(val_indices),
        "evolution_indices": evolution_indices,
        "test_indices": test_indices,
        "val_indices": val_indices,
        "evolution_ids": [str(ds.dataset[i]["id"]) for i in evolution_indices],
        "test_ids": [str(ds.dataset[i]["id"]) for i in test_indices],
        "val_ids": [str(ds.dataset[i]["id"]) for i in val_indices],
        # Prefix-stable references: regenerating with a larger n_test gives
        # test_ids = test_pool_ids[:new_n_test], a superset of the current
        # test_ids. Use these to verify or extend without reshuffling.
        "test_pool_indices": test_pool_indices,
        "test_pool_ids": [str(ds.dataset[i]["id"]) for i in test_pool_indices],
        "data_dir": str(Path(data_dir).resolve()),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_path = output_dir / "split.json"
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Split saved: {split_path}")
    print(f"  Evolution pool: {len(evolution_indices)} tasks")
    print(f"  Held-out test:  {len(test_indices)} tasks")
    print(f"  Held-out val:   {len(val_indices)} tasks")
    print(f"  Evolution pool: {len(evolution_indices)} tasks")
    print(f"  Held-out test:  {len(test_indices)} tasks")

    # Prepare normalized symlinks. Default puts them next to the split for
    # backward compat; callers passing an explicit ``normalized_dir`` (e.g.
    # ``results/normalized``) get the master-seed-invariant shared location.
    if normalized_dir is None:
        normalized_dir = output_dir / "normalized"
    prepared = ds.prepare_for_pipeline(Path(normalized_dir))
    print(f"  Normalized dir: {prepared}")

    return split_info


def load_split(split_dir: str | Path) -> dict:
    """Load an existing split.json."""
    with open(Path(split_dir) / "split.json") as f:
        return json.load(f)


def identify_failures(
    base_trajectories_dir: str | Path,
    evolution_ids: list[str],
    threshold: float = 1.0,
) -> tuple[list[str], list[str]]:
    """Identify failed and succeeded task IDs from base trajectory assessments.

    Args:
        base_trajectories_dir: Directory with {task_id}/assessment.json files.
        evolution_ids: All task IDs in the evolution pool.
        threshold: Accuracy below this is considered failure. Default 1.0
                   (anything less than perfect is a failure).

    Returns:
        (failure_ids, success_ids)
    """
    base_dir = Path(base_trajectories_dir)
    failure_ids = []
    success_ids = []

    for task_id in evolution_ids:
        assessment_path = base_dir / task_id / "assessment.json"
        if not assessment_path.exists():
            print(f"  Warning: no assessment for {task_id}, treating as failure")
            failure_ids.append(task_id)
            continue

        with open(assessment_path) as f:
            assessment = json.load(f)

        accuracy = assessment.get("cell_accuracy", 0.0)
        if accuracy < threshold:
            failure_ids.append(task_id)
        else:
            success_ids.append(task_id)

    return failure_ids, success_ids


def build_training_set(
    failure_ids: list[str],
    training_seed: int,
    n_train: int = 40,
    batch_size: int = 4,
) -> dict:
    """Select training tasks from failures and generate batch order.

    Args:
        failure_ids: All failed task IDs from base skill run.
        training_seed: Controls which failures are selected and batch order.
        n_train: Number of training tasks to select.
        batch_size: Mini-batch size.

    Returns:
        Dict with train_ids, batches, and metadata.
    """
    if n_train > len(failure_ids):
        print(f"  Warning: requested {n_train} training tasks but only "
              f"{len(failure_ids)} failures. Using all {len(failure_ids)}.")
        n_train = len(failure_ids)

    rng = random.Random(training_seed)

    # Select training tasks
    sampled = list(failure_ids)
    rng.shuffle(sampled)
    train_ids = sampled[:n_train]

    # Shuffle for batch order (1 epoch)
    rng.shuffle(train_ids)

    # Create batches
    batches = []
    for i in range(0, len(train_ids), batch_size):
        batch = train_ids[i:i + batch_size]
        if batch:
            batches.append(batch)

    return {
        "training_seed": training_seed,
        "n_train": len(train_ids),
        "n_failures_available": len(failure_ids),
        "batch_size": batch_size,
        "n_iterations": len(batches),
        "train_ids": train_ids,
        "batches": batches,
    }


def build_fixed_update_training_set(
    failure_ids: list[str],
    training_seed: int,
    n_train: int = 40,
    batch_size: int = 4,
    n_iterations: int = 10,
    batch_seed: int | None = None,
) -> dict:
    """Select a fixed train pool, then emit a fixed number of update batches.

    This schedule supports optimization-style ablations where the update budget
    is fixed but batch size varies. The 40-task pool is selected exactly as in
    ``build_training_set`` for the same ``training_seed``. Batches are then
    produced from that pool with DataLoader-like epoch cycling:

      - first epoch uses the canonical ``train_ids`` order, so bsz=4,
        n_iterations=10 matches the existing seed-0 one-epoch schedule;
      - when the pool is exhausted, the same pool is reshuffled and sampling
        continues until ``n_iterations`` full batches are produced;
      - no task repeats within a batch when ``batch_size <= n_train``.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if n_iterations <= 0:
        raise ValueError(f"n_iterations must be positive, got {n_iterations}")

    # Reuse the exact training-pool selection and canonical order from the
    # default one-epoch schedule. Using batch_size=n_train avoids introducing a
    # third shuffle path while keeping the selected train_ids identical.
    base = build_training_set(
        failure_ids=failure_ids,
        training_seed=training_seed,
        n_train=n_train,
        batch_size=max(1, n_train),
    )
    train_ids = list(base["train_ids"])
    if batch_size > len(train_ids):
        raise ValueError(
            f"batch_size ({batch_size}) cannot exceed selected n_train "
            f"({len(train_ids)}) for fixed-update batches"
        )

    rng = random.Random(training_seed if batch_seed is None else batch_seed)
    order = list(train_ids)
    cursor = 0
    batches: list[list[str]] = []

    while len(batches) < n_iterations:
        batch: list[str] = []
        while len(batch) < batch_size:
            if cursor >= len(order):
                order = list(train_ids)
                rng.shuffle(order)
                # If a batch crosses an epoch boundary, avoid within-batch
                # duplicates by consuming unseen items first.
                if batch:
                    seen = set(batch)
                    order = [x for x in order if x not in seen] + [
                        x for x in order if x in seen
                    ]
                cursor = 0
            take = min(batch_size - len(batch), len(order) - cursor)
            batch.extend(order[cursor:cursor + take])
            cursor += take
        batches.append(batch)

    task_counts = {task_id: 0 for task_id in train_ids}
    for batch in batches:
        for task_id in batch:
            task_counts[task_id] = task_counts.get(task_id, 0) + 1

    return {
        "training_seed": training_seed,
        "n_train": len(train_ids),
        "n_failures_available": base["n_failures_available"],
        "batch_size": batch_size,
        "n_iterations": len(batches),
        "batch_schedule": "fixed_updates",
        "batch_seed": training_seed if batch_seed is None else batch_seed,
        "train_ids": train_ids,
        "batches": batches,
        "n_total_batch_slots": len(batches) * batch_size,
        "n_unique_in_batches": sum(1 for c in task_counts.values() if c > 0),
        "task_sample_counts": task_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="SkillGrad data split and training set")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- split command ---
    sp_split = subparsers.add_parser("split", help="Create canonical 200-200 split")
    sp_split.add_argument("--data-dir", required=True,
                          help="Path to SpreadsheetBench dataset.")
    sp_split.add_argument("--results-root", default="results",
                          help="Root results directory. Splits land at "
                               "<results-root>/splits/master_<M>_heldout_<H>/, "
                               "normalized data at <results-root>/normalized/.")
    sp_split.add_argument("--master-seed", type=int, default=0,
                          help="Seed for 200/200 evolution-vs-heldout split.")
    sp_split.add_argument("--heldout-seed", type=int, default=42,
                          help="Seed for val/test_pool shuffle inside the "
                               "200 held-out tasks. Independent of master-seed.")
    sp_split.add_argument("--n-test", type=int, default=100,
                          help="Test set size (prefix of the 180-task test_pool).")
    sp_split.add_argument("--n-val", type=int, default=20,
                          help="Validation set size (fixed prefix of held-out).")

    # --- training-set command (inspect-only; training.py writes per-run files) ---
    sp_train = subparsers.add_parser(
        "training-set",
        help="Inspect a training set selection. Does not write a per-run file — "
             "training.py owns runs/<run_id>/train/train_set.json.",
    )
    sp_train.add_argument("--results-root", default="results")
    sp_train.add_argument("--master-seed", type=int, default=0)
    sp_train.add_argument("--heldout-seed", type=int, default=42)
    sp_train.add_argument("--model", required=True,
                          help="Executor model — locates the right "
                               "base_trajectories/<model>/ for failure ids.")
    sp_train.add_argument("--training-seed", type=int, default=0)
    sp_train.add_argument("--n-train", type=int, default=40)
    sp_train.add_argument("--batch-size", type=int, default=4)
    sp_train.add_argument("--batch-schedule",
                          choices=["epoch", "fixed-updates"],
                          default="epoch")
    sp_train.add_argument("--n-iterations", type=int, default=None)
    sp_train.add_argument("--batch-seed", type=int, default=None)

    args = parser.parse_args()

    results_root = Path(args.results_root)

    if args.command == "split":
        split_dir = splits_dir_for(results_root, args.master_seed, args.heldout_seed)
        normalized_dir = normalized_dir_for(results_root)
        create_split(
            args.data_dir,
            split_dir,
            master_seed=args.master_seed,
            heldout_seed=args.heldout_seed,
            n_test=args.n_test,
            n_val=args.n_val,
            normalized_dir=normalized_dir,
        )

    elif args.command == "training-set":
        split_dir = splits_dir_for(results_root, args.master_seed, args.heldout_seed)
        split = load_split(split_dir)
        print(f"Loaded split: {split['n_evolution']} evolution, {split['n_test']} test")

        base_traj_dir = base_trajectories_dir_for(
            results_root, args.master_seed, args.heldout_seed, args.model,
        )
        failure_ids, success_ids = identify_failures(
            base_traj_dir, split["evolution_ids"],
        )
        print(f"Base skill results: {len(failure_ids)} failures, "
              f"{len(success_ids)} successes")

        if args.batch_schedule == "fixed-updates":
            if args.n_iterations is None:
                parser.error("--n-iterations is required for fixed-updates")
            training = build_fixed_update_training_set(
                failure_ids=failure_ids,
                training_seed=args.training_seed,
                n_train=args.n_train,
                batch_size=args.batch_size,
                n_iterations=args.n_iterations,
                batch_seed=args.batch_seed,
            )
        else:
            if args.n_iterations is not None:
                parser.error("--n-iterations requires --batch-schedule fixed-updates")
            training = build_training_set(
                failure_ids, args.training_seed, args.n_train, args.batch_size,
            )
        print(f"\nTraining set (training_seed={args.training_seed}):")
        print(f"  Tasks:      {training['n_train']} / {training['n_failures_available']} failures")
        print(f"  Batch size: {training['batch_size']}")
        print(f"  Iterations: {training['n_iterations']}")
        print(f"  Schedule:   {training.get('batch_schedule', 'epoch')}")
        if "n_unique_in_batches" in training:
            print(f"  Unique used: {training['n_unique_in_batches']} / {training['n_train']}")
        for i, batch in enumerate(training["batches"]):
            print(f"  Batch {i+1}: {batch}")


if __name__ == "__main__":
    main()
