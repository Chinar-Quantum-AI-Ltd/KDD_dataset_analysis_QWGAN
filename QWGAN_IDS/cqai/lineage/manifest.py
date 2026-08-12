"""FR-8 Machine-Readable Manifest and Lineage Generator."""
from __future__ import annotations

import datetime
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def _get_git_commit_sha() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "uncommitted_local_build"


def compute_file_sha256(filepath: str | Path) -> str:
    path = Path(filepath)
    if not path.exists():
        return "file_not_found"
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def build_lineage_manifest(
    run_id: str,
    run_type: str,
    dataset_name: str,
    metrics: dict[str, Any],
    *,
    config: dict[str, Any] | None = None,
    artifact_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Generate machine-readable FR-8 lineage manifest.

    Parameters
    ----------
    run_id : str
        Unique run identifier.
    run_type : str
        Requirement/Experiment type (e.g. 'fr6_ablation', 'fr7_serving').
    dataset_name : str
        Name/version of dataset.
    metrics : dict
        Calculated evaluation metrics.
    config : dict, optional
        Resolved experiment configuration.
    artifact_paths : list[str], optional
        List of generated artifact filepaths to hash.

    Returns
    -------
    dict[str, Any]
        FR-8 compliant lineage manifest.
    """
    timestamp_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    git_sha = _get_git_commit_sha()

    artifact_hashes = {}
    if artifact_paths:
        for p in artifact_paths:
            artifact_hashes[p] = compute_file_sha256(p)

    manifest = {
        "run_id": run_id,
        "run_type": run_type,
        "timestamp_utc": timestamp_utc,
        "git_commit_sha": git_sha,
        "dataset_name": dataset_name,
        "config": config or {},
        "metrics": metrics,
        "artifact_hashes": artifact_hashes,
        "schema_version": "1.0",
    }

    return manifest
