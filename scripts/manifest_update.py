"""Upsert a run record into ``results/manifest.json``.

The manifest is the cross-run index. One record per run, keyed by
``run_id``. Re-running an existing run overwrites the record (does not
duplicate).

Called by training and eval at completion. Can also be invoked
directly:

    python -m scripts.manifest_update \\
        --results-root results \\
        --run-dir results/runs/skillgrad_gpt-5.4
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1


def _load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        return {"schema_version": SCHEMA_VERSION, "runs": [], "historical": []}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data.setdefault("schema_version", SCHEMA_VERSION)
    data.setdefault("runs", [])
    data.setdefault("historical", [])
    return data


def _save_manifest(manifest_path: Path, data: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.rename(manifest_path)


def _record_from_run_dir(run_dir: Path) -> dict[str, Any]:
    """Read config.json from the run dir, project a manifest record."""
    config_path = run_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"{config_path} not found. The run must write config.json before "
            f"calling manifest_update."
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))

    seeds = config.get("seeds", {})
    metrics = config.get("metrics", {})
    record: dict[str, Any] = {
        "run_id": config["run_id"],
        "path": str(run_dir),
        "method": config.get("method"),
        "model": config.get("model"),
        "version": config.get("version"),
        "master_seed": seeds.get("master_seed"),
        "heldout_seed": seeds.get("heldout_seed"),
        "training_seed": seeds.get("training_seed"),
        "config_tag": config.get("config_tag"),
        "test_hard": metrics.get("test_hard"),
        "test_cell_acc": metrics.get("test_cell_acc"),
        "elapsed_s": config.get("elapsed_s"),
        "cost": (config.get("cost") or {}).get("cost") if isinstance(config.get("cost"), dict) else config.get("cost"),
        "status": config.get("status"),
        "completed": config.get("completed"),
    }
    return record


def upsert(results_root: Path, run_dir: Path) -> dict:
    """Read run_dir/config.json, upsert a record into manifest.json.

    Returns the updated manifest dict.
    """
    manifest_path = Path(results_root) / "manifest.json"
    data = _load_manifest(manifest_path)
    record = _record_from_run_dir(Path(run_dir))

    runs = data["runs"]
    for i, existing in enumerate(runs):
        if existing.get("run_id") == record["run_id"]:
            runs[i] = record
            break
    else:
        runs.append(record)

    data["runs"] = runs
    data["last_updated"] = datetime.now(timezone.utc).isoformat()
    _save_manifest(manifest_path, data)
    print(
        f"  [manifest] upsert {record['run_id']} → {manifest_path} "
        f"(status={record.get('status')}, test_hard={record.get('test_hard')})"
    )
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Upsert a run record into manifest.json")
    parser.add_argument("--results-root", default="results")
    parser.add_argument("--run-dir", required=True,
                        help="Path to the run directory containing config.json.")
    args = parser.parse_args()
    upsert(Path(args.results_root), Path(args.run_dir))


if __name__ == "__main__":
    main()
