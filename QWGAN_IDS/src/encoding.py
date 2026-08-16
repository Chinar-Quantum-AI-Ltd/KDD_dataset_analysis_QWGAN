"""Feature encoding (FR-1 pipeline stage 3).

    Categoricals (protocol_type/service/flag) -> OneHotEncoder
    Numerics (duration, src_bytes, dst_bytes, ...) -> log1p -> RobustScaler

Persists:
    data/feature_matrix.pkl      (encoded feature matrix as DataFrame)
    artifacts/encoder.pkl        (fitted OneHotEncoder)
    artifacts/robust_scaler.pkl  (fitted RobustScaler)
    artifacts/feature_names.json (final column order)
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from cqai.lineage import load_artifact_manifest, registered_artifact, verified_joblib_load
from .preprocessing import BINARY_COLS, CATEGORICAL_COLS, CONTINUOUS_COLS

DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts")


# --------------------------------------------------------------------------- #
# Numeric transforms
# --------------------------------------------------------------------------- #
def log1p_transform(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Apply ``log1p`` to the given numeric columns (in place-safe copy)."""
    out = df.copy()
    out[cols] = np.log1p(out[cols].astype(np.float64))
    return out


def inverse_log1p(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Inverse of ``log1p``: ``x = expm1(log1p(x))``."""
    out = df.copy()
    out[cols] = np.expm1(out[cols].astype(np.float64))
    return out


# --------------------------------------------------------------------------- #
# Main encoding pipeline
# --------------------------------------------------------------------------- #
def build_feature_matrix(
    df: pd.DataFrame,
    categorical_cols: list[str] | None = None,
    numeric_cols: list[str] | None = None,
    fit_encoder: bool = True,
    artifact_dir: str | Path = ARTIFACT_DIR,
    data_dir: str | Path = DATA_DIR,
) -> tuple[pd.DataFrame, OneHotEncoder, RobustScaler]:
    """Encode a cleaned NSL-KDD dataframe into a numeric feature matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe (43-column NSL-KDD schema).
    categorical_cols :
        Defaults to ``CATEGORICAL_COLS`` (protocol_type, service, flag).
    numeric_cols :
        Defaults to ``CONTINUOUS_COLS + BINARY_COLS`` (skip label/difficulty).
    fit_encoder :
        Whether to fit the OneHotEncoder. Set False when transforming new data
        using a previously fitted encoder (pass it in ``df`` already-encoded).
    artifact_dir :
        Where to persist the fitted encoders.
    data_dir :
        Where to persist ``feature_matrix.pkl`` (created if missing).

    Returns
    -------
    (feature_matrix, encoder, robust_scaler)
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if categorical_cols is None:
        categorical_cols = list(CATEGORICAL_COLS)
    if numeric_cols is None:
        numeric_cols = list(CONTINUOUS_COLS) + list(BINARY_COLS)

    # 1) Categorical -> OneHot
    if fit_encoder:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        encoded = encoder.fit_transform(df[categorical_cols])
        cat_names = encoder.get_feature_names_out(categorical_cols)
    else:
        # Transforming new rows with an already-fitted encoder.
        encoder_path = artifact_dir / "encoder.pkl"
        if not encoder_path.exists():
            raise FileNotFoundError(f"{encoder_path} not found. Fit first.")
        registry = load_artifact_manifest(artifact_dir / "artifact_manifest.json")
        digest, fitting_versions = registered_artifact(registry, encoder_path.name)
        encoder = verified_joblib_load(
            encoder_path,
            expected_sha256=digest,
            fitting_versions=fitting_versions,
            require_envelope=False,
        )
        encoded = encoder.transform(df[categorical_cols])
        cat_names = encoder.get_feature_names_out(categorical_cols)

    cat_df = pd.DataFrame(encoded, columns=cat_names, index=df.index)

    # 2) Numeric -> log1p -> RobustScaler
    log_df = log1p_transform(df[numeric_cols], numeric_cols)
    scaler = RobustScaler()
    scaled = scaler.fit_transform(log_df)
    num_df = pd.DataFrame(scaled, columns=numeric_cols, index=df.index)

    # 3) Concatenate into a single feature matrix
    feature_matrix = pd.concat([num_df, cat_df], axis=1)

    # 4) Persist
    joblib.dump(encoder, artifact_dir / "encoder.pkl")
    joblib.dump(scaler, artifact_dir / "robust_scaler.pkl")
    with open(artifact_dir / "feature_names.json", "w") as fh:
        json.dump(list(feature_matrix.columns), fh, indent=2)

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    feature_matrix.to_pickle(data_dir / "feature_matrix.pkl")

    print(f"[encode] feature_matrix: {feature_matrix.shape} "
          f"({len(numeric_cols)} numeric + {len(cat_names)} one-hot)")
    print(f"[save] {data_dir / 'feature_matrix.pkl'}")
    print(f"[save] artifacts/encoder.pkl, robust_scaler.pkl, feature_names.json")
    return feature_matrix, encoder, scaler


# --------------------------------------------------------------------------- #
# Helpers to decode back toward the original feature space
# --------------------------------------------------------------------------- #
def numeric_cols_of(feature_names: list[str]) -> list[str]:
    """Return the subset of ``feature_names`` that are numeric (continuous +
    binary), i.e. not one-hot-encoded categorical columns."""
    return [c for c in feature_names if c in CONTINUOUS_COLS + BINARY_COLS]


def categorical_cols_of(feature_names: list[str]) -> list[str]:
    """Return the original categorical column names that appear (as one-hot
    columns) in ``feature_names``."""
    matched = set()
    for f in feature_names:
        for cat in CATEGORICAL_COLS:
            if f.startswith(f"{cat}_"):
                matched.add(cat)
    return [c for c in CATEGORICAL_COLS if c in matched]


def inverse_numeric(
    scaled_df: pd.DataFrame,
    scaler: RobustScaler,
    numeric_cols: list[str],
) -> pd.DataFrame:
    """RobustScaler inverse + expm1 (inverse log1p).

    The scaler is fitted on the *full* numeric column set (continuous +
    binary). When ``numeric_cols`` is a subset of it (e.g. only the numeric
    features among the top-k selected for PCA), the inverse transform is
    applied column-wise via ``center_`` / ``scale_`` so a partial matrix can
    be decoded without reconstructing the full fitted input.
    """
    if hasattr(scaler, "feature_names_in_"):
        fitted = list(scaler.feature_names_in_)
        missing = [c for c in numeric_cols if c not in fitted]
        if missing:
            raise ValueError(f"Columns not fitted by RobustScaler: {missing}")
        idx = [fitted.index(c) for c in numeric_cols]
        raw_scaled = (
            scaled_df[numeric_cols].to_numpy(dtype=np.float64) * scaler.scale_[idx]
            + scaler.center_[idx]
        )
    else:
        raw_scaled = scaler.inverse_transform(scaled_df[numeric_cols])
    log_df = pd.DataFrame(raw_scaled, columns=numeric_cols, index=scaled_df.index)
    return inverse_log1p(log_df, numeric_cols)


def one_hot_to_categorical(
    encoded_df: pd.DataFrame,
    encoder: OneHotEncoder,
    categorical_cols: list[str],
) -> pd.DataFrame:
    """Convert one-hot columns back to the original categorical columns."""
    raw = encoder.inverse_transform(encoded_df.values)
    return pd.DataFrame(raw, columns=categorical_cols, index=encoded_df.index)


def join_categories(
    encoded_df: pd.DataFrame,
    encoder: OneHotEncoder,
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> pd.DataFrame:
    """Return a single dataframe with numeric + original categorical columns."""
    cat_df = one_hot_to_categorical(
        encoded_df[encoder.get_feature_names_out(categorical_cols)],
        encoder,
        categorical_cols,
    )
    return pd.concat(
        [encoded_df[numeric_cols].reset_index(drop=True),
         cat_df.reset_index(drop=True)],
        axis=1,
    )

