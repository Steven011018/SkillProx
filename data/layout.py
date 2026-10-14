"""Canonical results-folder layout helpers.

The results layout factors run inputs from run outputs:

  results/
  ├── normalized/                              # master-seed invariant
  ├── splits/
  │   └── master_<M>_heldout_<H>/split.json    # one canonical split per seed pair
  ├── base_trajectories/
  │   └── master_<M>_heldout_<H>/<model>/      # one set per executor model
  ├── runs/
  │   └── <run_id>/
  │       ├── config.json
  │       ├── train/
  │       └── eval/
  └── manifest.json                            # cross-run index

Run id format:
    <method>_<model>[_<config_tag>]

Examples:
    skillgrad_gpt-5.4
    skillgrad_gpt-4.1
    skillgrad_gpt-5.4_seed1            (config_tag set by the user)
"""

from pathlib import Path
from typing import Iterable, Optional


def splits_dir_for(results_root: Path, master_seed: int, heldout_seed: int) -> Path:
    """Path to the canonical split for a (master, heldout) seed pair."""
    return Path(results_root) / "splits" / f"master_{master_seed}_heldout_{heldout_seed}"


def normalized_dir_for(results_root: Path) -> Path:
    """Path to the master-seed-invariant normalized dataset symlinks."""
    return Path(results_root) / "normalized"


def base_trajectories_dir_for(
    results_root: Path,
    master_seed: int,
    heldout_seed: int,
    model: str,
) -> Path:
    """Path to base trajectories for one (master, heldout, model) triple."""
    return (
        Path(results_root)
        / "base_trajectories"
        / f"master_{master_seed}_heldout_{heldout_seed}"
        / model
    )


def runs_dir_for(results_root: Path) -> Path:
    """Path to the runs root."""
    return Path(results_root) / "runs"


def manifest_path_for(results_root: Path) -> Path:
    """Path to the cross-run manifest."""
    return Path(results_root) / "manifest.json"


def run_id_for(
    method: str,
    model: str,
    config_tag: Optional[str] = None,
) -> str:
    """Construct the canonical run_id string.

    Format: ``<method>_<model>[_<config_tag>]``. The optional
    ``config_tag`` is a user-supplied suffix to keep multiple runs with
    the same method+model from colliding.
    """
    parts: list[str] = [method, model]
    if config_tag:
        parts.append(config_tag)
    return "_".join(parts)


def run_dir_for(results_root: Path, run_id: str) -> Path:
    """Path to one run's leaf directory."""
    return runs_dir_for(results_root) / run_id


def all_run_dirs(results_root: Path) -> Iterable[Path]:
    """Iterate existing run directories under ``results_root``."""
    rdir = runs_dir_for(results_root)
    if not rdir.exists():
        return iter(())
    return (p for p in sorted(rdir.iterdir()) if p.is_dir())
