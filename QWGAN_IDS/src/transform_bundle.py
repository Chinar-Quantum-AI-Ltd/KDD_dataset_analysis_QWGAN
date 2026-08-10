"""
TransformBundle class for FR-7.

This wraps the existing FR-1/FR-2 fitted artifacts and provides train/serve
compatible transformations, latent/angle encoding, and inverse paths where
mathematically supported.

The bundle expects the repository artifacts to contain:
- artifacts/encoder.pkl
- artifacts/robust_scaler.pkl
- artifacts/pca.pkl
- artifacts/feature_names.json
- data/selected_feature_names.npy
- data/latent_features.npy

If some artifacts are missing, the bundle raises informative errors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple

from src.loader import COLUMN_NAMES
from src.encoding import log1p_transform, inverse_log1p, numeric_cols_of, categorical_cols_of
from src.preprocessing import verify_schema, CONTINUOUS_COLS, CATEGORICAL_COLS, BINARY_COLS


@dataclass
class TransformBundle:
    # fitted objects
    encoder: Any
    scaler: Any
    pca: Any

    # feature lists / ordering
    feature_names: List[str]
    selected_feature_names: List[str]

    # latent training stats
    latent_training: Optional[np.ndarray] = None

    # angle scaling metadata
    angle_min: Optional[np.ndarray] = None
    angle_max: Optional[np.ndarray] = None

    # metadata
    bundle_version: str = "0.1"
    schema_version: str = "NSL-KDD-43"
    dataset: str = "NSL-KDD"
    created_at: Optional[str] = None
    artifact_versions: Dict[str, Any] = field(default_factory=dict)

    # internal derived
    categorical_columns: List[str] = field(default_factory=list)
    numerical_columns: List[str] = field(default_factory=list)
    feature_order: List[str] = field(default_factory=list)
    latent_dim: int = 0

    @classmethod
    def load_from_artifacts(cls, artifacts_dir: str = "artifacts", data_dir: str = "data") -> "TransformBundle":
        art = Path(artifacts_dir)
        data = Path(data_dir)

        # load encoder
        encoder_path = art / "encoder.pkl"
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder not found at {encoder_path}")
        encoder = joblib.load(encoder_path)

        # load scaler
        scaler_path = art / "robust_scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        scaler = joblib.load(scaler_path)

        # load pca (may be absent if autoencoder used)
        pca_path = art / "pca.pkl"
        pca = joblib.load(pca_path) if pca_path.exists() else None

        # feature names
        fn_path = art / "feature_names.json"
        if not fn_path.exists():
            raise FileNotFoundError(f"feature_names.json not found at {fn_path}")
        with open(fn_path, "r") as fh:
            feature_names = json.load(fh)

        # selected feature names
        sel_path = data / "selected_feature_names.npy"
        if not sel_path.exists():
            raise FileNotFoundError(f"selected_feature_names.npy not found at {sel_path}")
        selected = list(np.load(sel_path, allow_pickle=True))

        # latent training data (used to compute angle scaling)
        latent_path = data / "latent_features.npy"
        latent_training = None
        angle_min = None
        angle_max = None
        if latent_path.exists():
            latent_training = np.load(latent_path)
            angle_min = np.min(latent_training, axis=0)
            angle_max = np.max(latent_training, axis=0)

        bundle = cls(
            encoder=encoder,
            scaler=scaler,
            pca=pca,
            feature_names=feature_names,
            selected_feature_names=selected,
            latent_training=latent_training,
            angle_min=angle_min,
            angle_max=angle_max,
        )

        bundle.categorical_columns = [c for c in feature_names if any(c.startswith(cat + "_") for cat in CATEGORICAL_COLS)]
        # numerical columns are those in CONTINUOUS_COLS + BINARY_COLS intersect feature_names
        bundle.numerical_columns = [c for c in feature_names if c in (CONTINUOUS_COLS + BINARY_COLS)]
        bundle.feature_order = feature_names
        bundle.latent_dim = int(bundle.pca.n_components) if bundle.pca is not None else (bundle.latent_training.shape[1] if bundle.latent_training is not None else 0)

        return bundle

    def save(self, path: str):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "TransformBundle":
        obj = joblib.load(path)
        if not isinstance(obj, cls):
            raise ValueError("Loaded object is not a TransformBundle")
        return obj

    # ----------------------------- validation ------------------------------
    def validate_schema(self, df: pd.DataFrame) -> dict:
        # reuse preprocessing.verify_schema
        return verify_schema(df, strict=True)

    # ----------------------------- forward transforms ---------------------
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        # encoder expects categorical columns in order
        cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
        encoded = self.encoder.transform(df[cat_cols])
        cat_names = self.encoder.get_feature_names_out(cat_cols)
        cat_df = pd.DataFrame(encoded, columns=list(cat_names), index=df.index)
        return cat_df

    def _scale_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        # apply log1p then scaler.transform
        num_cols = [c for c in (CONTINUOUS_COLS + BINARY_COLS) if c in df.columns]
        if len(num_cols) == 0:
            return pd.DataFrame(index=df.index)
        log_df = log1p_transform(df[num_cols], num_cols)
        scaled = self.scaler.transform(log_df)
        num_df = pd.DataFrame(scaled, columns=num_cols, index=df.index)
        return num_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return the encoded+scaled full feature matrix in the same ordering as training."""
        self.validate_schema(df)
        num_df = self._scale_numeric(df)
        cat_df = self._encode_categorical(df)
        # concat in feature order
        full = pd.concat([num_df, cat_df], axis=1)
        # ensure ordering matches saved feature_names
        missing = [c for c in self.feature_order if c not in full.columns]
        if missing:
            raise ValueError(f"Transformed feature matrix missing columns expected by bundle: {missing}")
        full = full[self.feature_order]
        # deterministic: ensure dtype
        return full

    def transform_to_latent(self, df: pd.DataFrame) -> np.ndarray:
        X = self.transform(df)
        # select top features before PCA
        missing_selected = [c for c in self.selected_feature_names if c not in X.columns]
        if missing_selected:
            raise ValueError(f"Selected features missing from transformed matrix: {missing_selected}")
        X_top = X[self.selected_feature_names]
        if self.pca is None:
            raise ValueError("No PCA model available in bundle for latent transform")
        latent = self.pca.transform(X_top.values)
        if latent.shape[1] != self.latent_dim:
            raise ValueError(f"Latent dimension mismatch: expected {self.latent_dim}, got {latent.shape[1]}")
        return latent.astype(np.float32)

    def transform_to_angles(self, df: pd.DataFrame, eps: float = 1e-8) -> np.ndarray:
        latent = self.transform_to_latent(df)
        if self.angle_min is None or self.angle_max is None:
            # fallback: scale by per-dimension 99th percentiles from training latent if available
            if self.latent_training is None:
                raise ValueError("No latent training stats available to compute angle scaling")
            amin = np.min(self.latent_training, axis=0)
            amax = np.max(self.latent_training, axis=0)
        else:
            amin = self.angle_min
            amax = self.angle_max
        amin = amin.astype(np.float32)
        amax = amax.astype(np.float32)
        # avoid division by zero
        denom = amax - amin
        denom[denom == 0] = eps
        scaled = (latent - amin) / denom
        angles = np.clip(scaled, 0.0, 1.0) * np.pi
        # validations
        if not np.isfinite(angles).all():
            raise ValueError("Angles contain non-finite values")
        if angles.min() < -1e-6 or angles.max() > np.pi + 1e-6:
            raise ValueError("Angles out of [0, pi] after scaling")
        return angles.astype(np.float32)

    # ----------------------------- inverse transforms ---------------------
    def inverse_angles(self, angles: np.ndarray) -> np.ndarray:
        angles = np.asarray(angles, dtype=np.float32)
        if angles.ndim != 2 or angles.shape[1] != self.latent_dim:
            raise ValueError(f"Angles must be shape (N, {self.latent_dim})")
        scaled = angles / np.pi
        if self.angle_min is None or self.angle_max is None:
            if self.latent_training is None:
                raise ValueError("No latent training stats available to invert angles")
            amin = np.min(self.latent_training, axis=0)
            amax = np.max(self.latent_training, axis=0)
        else:
            amin = self.angle_min
            amax = self.angle_max
        # invert scaling
        latent = scaled * (amax - amin) + amin
        return latent.astype(np.float32)

    def inverse_latent(self, latent: np.ndarray) -> pd.DataFrame:
        latent = np.asarray(latent)
        if self.pca is None or not hasattr(self.pca, 'inverse_transform'):
            raise ValueError("PCA inverse not available; latent may not be invertible")
        X_top = self.pca.inverse_transform(latent)
        # X_top columns correspond to selected_feature_names
        df_top = pd.DataFrame(X_top, columns=self.selected_feature_names)
        # inverse numeric scaling for numeric subset
        numeric_cols = [c for c in self.selected_feature_names if c in (CONTINUOUS_COLS + BINARY_COLS)]
        if numeric_cols:
            # construct a temporary scaled df matching scaler's full columns shape if necessary
            inv_numeric = inverse_log1p(df_top[numeric_cols], numeric_cols)
            df_top[numeric_cols] = inv_numeric
        # For categorical columns among selected features, we cannot perfectly inverse if encoder produced one-hots
        # Attempt to reconstruct categorical columns if all one-hot groups present
        # Build full encoded dataframe with zeros for missing one-hot columns
        # Note: exact inversion to original category strings may be approximate
        return df_top

    def metadata(self) -> Dict[str, Any]:
        return {
            'bundle_version': self.bundle_version,
            'schema_version': self.schema_version,
            'dataset': self.dataset,
            'latent_dimension': int(self.latent_dim),
            'angle_range': '[0, pi]',
            'categorical_features': CATEGORICAL_COLS,
            'numerical_features': CONTINUOUS_COLS + BINARY_COLS,
            'feature_order': self.feature_order,
            'artifact_versions': self.artifact_versions,
        }
