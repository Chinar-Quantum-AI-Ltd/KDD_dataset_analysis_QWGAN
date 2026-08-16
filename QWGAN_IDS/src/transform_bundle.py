"""
TransformBundle class for FR-7: Registered Transform Bundle for Live Inference.

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

No preprocessing objects are fitted during transform() or any live inference path.
All parameters come from the registered bundle created at training time.
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
from cqai.lineage import (
    dump_joblib_artifact,
    load_artifact_manifest,
    registered_artifact,
    sha256_file,
    verified_joblib_load,
)


@dataclass
class TransformBundle:
    """
    Serializable bundle containing fitted preprocessing, encoding, and dimensionality
    reduction objects for live inference.
    
    The bundle provides a unified transform(df_raw_flow) interface that converts
    raw network-flow features into the exact normalized representation expected by
    registered classifiers.
    
    Attributes
    ----------
    encoder : object
        Fitted OneHotEncoder for categorical features.
    scaler : object
        Fitted RobustScaler for numeric features (applied after log1p).
    pca : object
        Fitted PCA for dimensionality reduction to latent space.
    feature_names : List[str]
        Exact column order after encoding (numeric + one-hot categorical).
    selected_feature_names : List[str]
        Top-k features selected by mutual information before PCA.
    latent_training : Optional[np.ndarray]
        Training latent features (used to compute angle scaling bounds).
    angle_min : Optional[np.ndarray]
        Per-dimension minimum angle scaling factors.
    angle_max : Optional[np.ndarray]
        Per-dimension maximum angle scaling factors.
    bundle_version : str
        Version of this bundle class.
    schema_version : str
        Version of the NSL-KDD schema (43 columns).
    dataset : str
        Dataset identifier ("NSL-KDD").
    created_at : Optional[str]
        Timestamp when the bundle was created.
    artifact_versions : Dict[str, Any]
        Metadata about artifact versions/hashes.
    """
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
    bundle_version: str = "1.0"
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
    def load_from_artifacts(
        cls,
        artifacts_dir: str = "artifacts",
        data_dir: str = "data",
        manifest_path: str | None = None,
    ) -> "TransformBundle":
        """
        Load a TransformBundle from fitted artifacts on disk.
        
        Parameters
        ----------
        artifacts_dir : str
            Directory containing encoder.pkl, robust_scaler.pkl, pca.pkl, feature_names.json
        data_dir : str
            Directory containing selected_feature_names.npy, latent_features.npy
            
        Returns
        -------
        TransformBundle
            Fully initialized bundle with all artifacts loaded.
            
        Raises
        ------
        FileNotFoundError
            If any required artifact is missing.
        """
        art = Path(artifacts_dir)
        data = Path(data_dir)
        registry = load_artifact_manifest(
            manifest_path or art / "artifact_manifest.json"
        )

        def load_registered_joblib(path: Path) -> Any:
            digest, fitting_versions = registered_artifact(registry, path.name)
            return verified_joblib_load(
                path,
                expected_sha256=digest,
                fitting_versions=fitting_versions,
                require_envelope=False,
            )

        # load encoder
        encoder_path = art / "encoder.pkl"
        if not encoder_path.exists():
            raise FileNotFoundError(f"Encoder not found at {encoder_path}")
        encoder = load_registered_joblib(encoder_path)

        # load scaler
        scaler_path = art / "robust_scaler.pkl"
        if not scaler_path.exists():
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        scaler = load_registered_joblib(scaler_path)

        # load pca (may be absent if autoencoder used)
        pca_path = art / "pca.pkl"
        pca = load_registered_joblib(pca_path) if pca_path.exists() else None

        # feature names
        fn_path = art / "feature_names.json"
        if not fn_path.exists():
            raise FileNotFoundError(f"feature_names.json not found at {fn_path}")
        feature_names_hash, _ = registered_artifact(registry, fn_path.name)
        if sha256_file(fn_path) != feature_names_hash:
            raise ValueError("feature_names.json failed registered SHA-256 verification")
        with open(fn_path, "r") as fh:
            feature_names = json.load(fh)

        # selected feature names
        sel_path = data / "selected_feature_names.npy"
        if not sel_path.exists():
            raise FileNotFoundError(f"selected_feature_names.npy not found at {sel_path}")
        selected_hash, _ = registered_artifact(registry, sel_path.name)
        if sha256_file(sel_path) != selected_hash:
            raise ValueError("selected feature names failed registered SHA-256 verification")
        selected = list(np.load(sel_path, allow_pickle=False))

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

        # Derive internal state
        bundle.categorical_columns = [
            c for c in feature_names
            if any(c.startswith(cat + "_") for cat in CATEGORICAL_COLS)
        ]
        # numerical columns are those in CONTINUOUS_COLS + BINARY_COLS intersect feature_names
        bundle.numerical_columns = [
            c for c in feature_names
            if c in (CONTINUOUS_COLS + BINARY_COLS)
        ]
        bundle.feature_order = feature_names
        bundle.latent_dim = (
            int(bundle.pca.n_components)
            if bundle.pca is not None
            else (bundle.latent_training.shape[1] if bundle.latent_training is not None else 0)
        )

        return bundle

    def save(self, path: str) -> None:
        """
        Serialize the bundle to disk using joblib.
        
        Parameters
        ----------
        path : str
            Destination file path.
        """
        dump_joblib_artifact(self, path, kind="transform_bundle")

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_sha256: str,
        fitting_versions: Dict[str, str],
    ) -> "TransformBundle":
        """
        Deserialize a bundle from disk.
        
        Parameters
        ----------
        path : str
            Path to the .joblib file.
            
        Returns
        -------
        TransformBundle
            Loaded bundle object.
            
        Raises
        ------
        ValueError
            If the loaded object is not a TransformBundle.
        """
        return verified_joblib_load(
            path,
            expected_sha256=expected_sha256,
            expected_type=cls,
            expected_kind="transform_bundle",
            fitting_versions=fitting_versions,
        )

    # ============================== VALIDATION ============================== #

    def validate_schema(self, df: pd.DataFrame) -> dict:
        """
        Validate that the input dataframe matches the 43-column NSL-KDD schema.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe to validate.
            
        Returns
        -------
        dict
            Validation results dictionary.
            
        Raises
        ------
        ValueError
            If schema validation fails.
        """
        return verify_schema(df, strict=True)

    def _validate_raw_input(self, df: pd.DataFrame) -> dict:
        """
        Deep validation of raw input dataframe before transformation.
        
        Detects and reports:
        - missing columns
        - unexpected columns
        - invalid datatypes
        - NaN / Inf values
        - invalid categorical values
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw input dataframe.
            
        Returns
        -------
        dict
            Validation report.
            
        Raises
        ------
        ValueError
            If critical validation issues are detected.
        """
        report = {
            "schema_valid": False,
            "missing_columns": [],
            "unexpected_columns": [],
            "invalid_dtypes": [],
            "nan_issues": [],
            "inf_issues": [],
            "invalid_categories": [],
        }

        if not isinstance(df, pd.DataFrame):
            raise TypeError("Raw flow input must be a pandas DataFrame")

        # Check schema
        try:
            self.validate_schema(df)
            report["schema_valid"] = True
        except ValueError as e:
            raise ValueError(f"Schema validation failed: {e}")

        # Check for NaN in any column
        nan_cols = df.columns[df.isna().any()].tolist()
        if nan_cols:
            report["nan_issues"] = nan_cols
            raise ValueError(f"Found NaN in columns: {nan_cols}")

        # Check for Inf in numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if np.isinf(df[col]).any():
                report["inf_issues"].append(col)
        if report["inf_issues"]:
            raise ValueError(f"Found Inf in columns: {report['inf_issues']}")

        # Check categorical values
        for cat_col in CATEGORICAL_COLS:
            if cat_col in df.columns:
                # Get unique values seen during encoding
                try:
                    fitted_categories = self.encoder.categories_
                    cat_idx = CATEGORICAL_COLS.index(cat_col)
                    allowed = set(fitted_categories[cat_idx])
                    actual = set(df[cat_col].unique())
                    invalid = actual - allowed
                    if invalid:
                        report["invalid_categories"].append({
                            "column": cat_col,
                            "invalid_values": list(invalid),
                        })
                except (AttributeError, IndexError) as exc:
                    raise ValueError(
                        f"Cannot validate fitted categories for {cat_col}: {exc}"
                    ) from exc

        if report["invalid_categories"]:
            raise ValueError(
                "Unknown categorical values are out-of-distribution and cannot be "
                f"silently encoded as zero blocks: {report['invalid_categories']}"
            )

        return report

    # ========================== FORWARD TRANSFORMS ========================== #

    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the fitted OneHotEncoder to categorical columns.
        
        No fitting happens here; only transform.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with categorical columns.
            
        Returns
        -------
        pd.DataFrame
            One-hot encoded categorical columns.
        """
        cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
        encoded = self.encoder.transform(df[cat_cols])
        cat_names = self.encoder.get_feature_names_out(cat_cols)
        cat_df = pd.DataFrame(encoded, columns=list(cat_names), index=df.index)
        return cat_df

    def _scale_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply log1p and the fitted RobustScaler to numeric columns.
        
        No fitting happens here; only transform.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe with numeric columns.
            
        Returns
        -------
        pd.DataFrame
            Scaled numeric columns.
        """
        num_cols = [c for c in (CONTINUOUS_COLS + BINARY_COLS) if c in df.columns]
        if len(num_cols) == 0:
            return pd.DataFrame(index=df.index)
        log_df = log1p_transform(df[num_cols], num_cols)
        scaled = self.scaler.transform(log_df)
        num_df = pd.DataFrame(scaled, columns=num_cols, index=df.index)
        return num_df

    def transform(self, df_raw_flow: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw network-flow features into the encoded+scaled representation.
        
        This is the primary entry point for live inference. It applies:
        1. Schema validation
        2. Numeric scaling (log1p + RobustScaler)
        3. Categorical encoding (OneHotEncoder)
        4. Feature reordering to match training
        
        Parameters
        ----------
        df_raw_flow : pd.DataFrame
            Raw input dataframe with 43-column NSL-KDD schema.
            
        Returns
        -------
        pd.DataFrame
            Encoded and scaled feature matrix in training feature order.
            
        Raises
        ------
        ValueError
            If schema validation fails or transformed output has incorrect shape.
        """
        # Validate raw input
        self._validate_raw_input(df_raw_flow)

        # Transform numerics and categoricals
        num_df = self._scale_numeric(df_raw_flow)
        cat_df = self._encode_categorical(df_raw_flow)

        # Concatenate
        full = pd.concat([num_df, cat_df], axis=1)

        # Verify all expected features are present
        missing = [c for c in self.feature_order if c not in full.columns]
        if missing:
            raise ValueError(
                f"Transformed feature matrix missing columns expected by bundle: {missing}"
            )

        # Reorder to match training
        full = full[self.feature_order]

        # Ensure deterministic dtype
        return full.astype(np.float64)

    def transform_to_latent(self, df_raw_flow: pd.DataFrame) -> np.ndarray:
        """
        Transform raw features to the 10-dimensional latent space via PCA.
        
        Parameters
        ----------
        df_raw_flow : pd.DataFrame
            Raw input dataframe.
            
        Returns
        -------
        np.ndarray
            Shape (n_samples, latent_dim) with dtype float32.
            
        Raises
        ------
        ValueError
            If PCA is not available or latent dimension is incorrect.
        """
        X = self.transform(df_raw_flow)

        # Select top features before PCA
        missing_selected = [
            c for c in self.selected_feature_names if c not in X.columns
        ]
        if missing_selected:
            raise ValueError(
                f"Selected features missing from transformed matrix: {missing_selected}"
            )

        X_top = X[self.selected_feature_names]

        if self.pca is None:
            raise ValueError("No PCA model available in bundle for latent transform")

        latent = self.pca.transform(X_top.values)

        if latent.shape[1] != self.latent_dim:
            raise ValueError(
                f"Latent dimension mismatch: expected {self.latent_dim}, got {latent.shape[1]}"
            )

        return latent.astype(np.float32)

    def transform_to_angles(
        self, df_raw_flow: pd.DataFrame, eps: float = 1e-8
    ) -> np.ndarray:
        """
        Transform raw features to quantum angles in [0, π].
        
        Applies: latent -> MinMax scaling -> [0, 1] -> multiply by π -> [0, π].
        
        Parameters
        ----------
        df_raw_flow : pd.DataFrame
            Raw input dataframe.
        eps : float
            Small epsilon to avoid division by zero.
            
        Returns
        -------
        np.ndarray
            Shape (n_samples, latent_dim) with angles in [0, π], dtype float32.
            
        Raises
        ------
        ValueError
            If angle scaling bounds or latent vectors are invalid.
        """
        latent = self.transform_to_latent(df_raw_flow)

        # Get angle scaling bounds
        if self.angle_min is None or self.angle_max is None:
            if self.latent_training is None:
                raise ValueError(
                    "No latent training stats available to compute angle scaling"
                )
            amin = np.min(self.latent_training, axis=0)
            amax = np.max(self.latent_training, axis=0)
        else:
            amin = self.angle_min
            amax = self.angle_max

        amin = amin.astype(np.float32)
        amax = amax.astype(np.float32)

        # Avoid division by zero
        denom = amax - amin
        denom[denom == 0] = eps
        scaled = (latent - amin) / denom
        angles = np.clip(scaled, 0.0, 1.0) * np.pi

        # Validate angles
        if not np.isfinite(angles).all():
            raise ValueError("Angles contain non-finite values")
        if angles.min() < -1e-6 or angles.max() > np.pi + 1e-6:
            raise ValueError("Angles out of [0, π] after scaling")

        return angles.astype(np.float32)

    # ========================== INVERSE TRANSFORMS ========================== #

    def inverse_angles(self, angles: np.ndarray) -> np.ndarray:
        """
        Invert angles back to latent space.
        
        Parameters
        ----------
        angles : np.ndarray
            Shape (n_samples, latent_dim), values in [0, π].
            
        Returns
        -------
        np.ndarray
            Shape (n_samples, latent_dim), latent space values.
        """
        angles = np.asarray(angles, dtype=np.float32)
        if angles.ndim != 2 or angles.shape[1] != self.latent_dim:
            raise ValueError(f"Angles must be shape (N, {self.latent_dim})")

        scaled = angles / np.pi

        if self.angle_min is None or self.angle_max is None:
            if self.latent_training is None:
                raise ValueError(
                    "No latent training stats available to invert angles"
                )
            amin = np.min(self.latent_training, axis=0)
            amax = np.max(self.latent_training, axis=0)
        else:
            amin = self.angle_min
            amax = self.angle_max

        # Invert scaling
        latent = scaled * (amax - amin) + amin
        return latent.astype(np.float32)

    def inverse_latent(self, latent: np.ndarray) -> pd.DataFrame:
        """
        Invert latent vectors back to feature space (lossy).
        
        Note: This is approximate due to PCA dimensionality reduction.
        Categorical features are kept as inverse-PCA values.
        
        Parameters
        ----------
        latent : np.ndarray
            Shape (n_samples, latent_dim).
            
        Returns
        -------
        pd.DataFrame
            Decoded feature space (selected features only).
        """
        latent = np.asarray(latent)

        if self.pca is None or not hasattr(self.pca, "inverse_transform"):
            raise ValueError("PCA inverse not available; latent may not be invertible")

        X_top = self.pca.inverse_transform(latent)
        df_top = pd.DataFrame(X_top, columns=self.selected_feature_names)

        # Inverse numeric scaling for numeric subset
        numeric_cols = [
            c for c in self.selected_feature_names
            if c in (CONTINUOUS_COLS + BINARY_COLS)
        ]
        if numeric_cols:
            inv_numeric = inverse_log1p(df_top[numeric_cols], numeric_cols)
            df_top[numeric_cols] = inv_numeric

        return df_top

    # ============================== METADATA ============================== #

    def metadata(self) -> Dict[str, Any]:
        """
        Return bundle metadata as a dictionary.
        
        Returns
        -------
        dict
            Metadata including versions, feature order, and configuration.
        """
        return {
            "bundle_version": self.bundle_version,
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "latent_dimension": int(self.latent_dim),
            "angle_range": "[0, π]",
            "categorical_features": CATEGORICAL_COLS,
            "numerical_features": CONTINUOUS_COLS + BINARY_COLS,
            "feature_order": self.feature_order,
            "selected_features": self.selected_feature_names,
            "artifact_versions": self.artifact_versions,
            "created_at": self.created_at,
        }
