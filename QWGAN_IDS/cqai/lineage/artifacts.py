"""Trusted artifact integrity and compatibility controls.

Pickle-backed formats must never be opened until their identity has been
checked against provenance supplied by a trusted registry or a Git-tracked
manifest.  Hashes establish identity, not authenticity: callers remain
responsible for protecting the manifest/registry itself.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Any, Mapping

import joblib


ARTIFACT_FORMAT = "cqai.joblib.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PACKAGE_NAMES = {
    "joblib": "joblib",
    "numpy": "numpy",
    "scikit_learn": "scikit-learn",
}


class ArtifactIntegrityError(ValueError):
    """Artifact bytes do not match their trusted registration record."""


class ArtifactCompatibilityError(RuntimeError):
    """Artifact fitting environment is incompatible with this runtime."""


def sha256_file(path: str | Path, *, chunk_size: int = 1 << 20) -> str:
    """Stream a file and return a lowercase SHA-256 digest."""

    artifact = Path(path)
    if not artifact.is_file():
        raise FileNotFoundError(f"Artifact does not exist: {artifact}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with artifact.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file_sha256(path: str | Path, expected_sha256: str) -> str:
    """Verify identity before any deserializer is invoked."""

    if not isinstance(expected_sha256, str) or not _SHA256_RE.fullmatch(expected_sha256):
        raise ArtifactIntegrityError("A trusted lowercase SHA-256 digest is required")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ArtifactIntegrityError(
            f"Artifact SHA-256 mismatch for {Path(path)}: "
            f"expected={expected_sha256}, actual={actual}"
        )
    return actual


def runtime_versions() -> dict[str, str]:
    """Return versions that affect persisted sklearn/joblib estimators."""

    return {
        name: importlib.metadata.version(package)
        for name, package in _PACKAGE_NAMES.items()
    }


def require_compatible_versions(expected: Mapping[str, str] | None) -> None:
    """Reject declared fitting/runtime version mismatches before unpickling."""

    if not expected:
        raise ArtifactCompatibilityError(
            "Artifact fitting versions are missing; refusing unsafe compatibility guess"
        )
    current = runtime_versions()
    mismatches = {
        name: {"fitted": str(version), "runtime": current.get(name)}
        for name, version in expected.items()
        if name in current and str(version) != current[name]
    }
    if mismatches:
        raise ArtifactCompatibilityError(
            "Persisted estimator version mismatch; rebuild under the supported runtime: "
            f"{mismatches}"
        )


def artifact_envelope(payload: Any, *, kind: str) -> dict[str, Any]:
    """Wrap a new artifact with fitting-version metadata."""

    return {
        "artifact_format": ARTIFACT_FORMAT,
        "kind": kind,
        "fitting_versions": runtime_versions(),
        "payload": payload,
    }


def dump_joblib_artifact(payload: Any, path: str | Path, *, kind: str) -> Path:
    """Persist a versioned artifact envelope."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact_envelope(payload, kind=kind), destination)
    return destination


def verified_joblib_load(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_type: type | tuple[type, ...] | None = None,
    expected_kind: str | None = None,
    fitting_versions: Mapping[str, str] | None = None,
    require_envelope: bool = True,
) -> Any:
    """Verify identity and declared compatibility before deserializing."""

    verify_file_sha256(path, expected_sha256)
    if fitting_versions is None and require_envelope:
        raise ArtifactCompatibilityError(
            "Trusted external fitting-version metadata is required before deserialization"
        )
    if fitting_versions is not None:
        require_compatible_versions(fitting_versions)

    loaded = joblib.load(path)
    if isinstance(loaded, dict) and loaded.get("artifact_format") == ARTIFACT_FORMAT:
        require_compatible_versions(loaded.get("fitting_versions"))
        if expected_kind is not None and loaded.get("kind") != expected_kind:
            raise TypeError(
                f"Artifact kind mismatch: expected={expected_kind!r}, "
                f"actual={loaded.get('kind')!r}"
            )
        payload = loaded.get("payload")
    elif require_envelope:
        raise ArtifactCompatibilityError(
            "Legacy pickle has no versioned CQAI artifact envelope; rebuild it before use"
        )
    else:
        payload = loaded

    if expected_type is not None and not isinstance(payload, expected_type):
        raise TypeError(
            f"Artifact type mismatch: expected={expected_type}, actual={type(payload)}"
        )
    return payload


def verified_pandas_read_pickle(
    path: str | Path, *, expected_sha256: str
) -> Any:
    """Hash-gate a legacy pandas pickle before its executable loader runs."""

    verify_file_sha256(path, expected_sha256)
    import pandas as pd

    return pd.read_pickle(path)


def load_artifact_manifest(path: str | Path) -> dict[str, Any]:
    """Load a non-executable JSON registry and validate its minimal structure."""

    manifest_path = Path(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"Cannot read artifact manifest {manifest_path}: {exc}") from exc
    if manifest.get("manifest_version") != "1.0":
        raise ArtifactIntegrityError("Unsupported or missing artifact manifest version")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ArtifactIntegrityError("Artifact manifest has no registered artifacts")
    for name, record in artifacts.items():
        digest = record.get("sha256") if isinstance(record, dict) else None
        if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
            raise ArtifactIntegrityError(f"Invalid SHA-256 record for {name}")
    return manifest


def registered_artifact(
    manifest: Mapping[str, Any], name: str
) -> tuple[str, Mapping[str, str] | None]:
    """Resolve a digest and fitting versions from a loaded trusted manifest."""

    try:
        record = manifest["artifacts"][name]
    except (KeyError, TypeError) as exc:
        raise ArtifactIntegrityError(f"Artifact {name!r} is not registered") from exc
    return str(record["sha256"]), record.get("fitting_versions")
