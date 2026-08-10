from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any

from src.transform_bundle import TransformBundle


class OnlineFeatureTransformer:
    """Load a saved TransformBundle and apply the exact training transforms to
    inbound flow data in a fail-closed manner.

    Usage:
        t = OnlineFeatureTransformer("artifacts/transform_bundle.joblib")
        features = t.transform_flow(flow_df)  # returns latent ndarray (N, latent_dim)
    """

    def __init__(self, bundle_path: str | Path):
        p = Path(bundle_path)
        if not p.exists():
            raise FileNotFoundError(f"Transform bundle not found at {p}")
        obj = joblib.load(p)
        if not isinstance(obj, TransformBundle):
            raise ValueError("Loaded object is not a TransformBundle")
        self.bundle: TransformBundle = obj

    def transform_flow(self, flow: pd.DataFrame) -> np.ndarray:
        """Validate and transform a flow dataframe into the classifier feature
        vector. This implementation returns the latent representation produced
        by the training PCA (shape (N, latent_dim)).

        The transformer never fits or mutates the bundle; it only uses fitted
        artifacts. On any schema mismatch, NaN/Inf, or other problem it raises
        an informative exception (fail-closed).
        """
        if not isinstance(flow, pd.DataFrame):
            raise TypeError("flow must be a pandas DataFrame")

        # Validate schema (this will raise if schema doesn't match)
        self.bundle.validate_schema(flow)

        # Transform to latent using the pre-fitted objects
        latent = self.bundle.transform_to_latent(flow)

        # Validate numeric stability
        if not np.isfinite(latent).all():
            raise ValueError("Latent representation contains non-finite values")
        if latent.shape[1] != self.bundle.latent_dim:
            raise ValueError(
                f"Latent dimension mismatch: expected {self.bundle.latent_dim}, got {latent.shape[1]}"
            )
        return latent
