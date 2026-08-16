"""Feature selection & dimensionality reduction (FR-1 pipeline stage 4).

    feature_matrix.pkl
        |
        v
    Mutual Information (vs 'label')
        |
        v
    Top-20 features
        |
        v
    PCA -> 10 latent dimensions
        |
        v
    latent_features.npy

Optional alternative: Keras autoencoder producing a 10-dim latent vector
(guarded import so sklearn-only installs still work).
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from cqai.lineage import (
    load_artifact_manifest,
    registered_artifact,
    verified_pandas_read_pickle,
)
from sklearn.preprocessing import LabelEncoder

DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts")

TOP_K = 20
LATENT_DIM = 10


# --------------------------------------------------------------------------- #
# Mutual information based feature selection
# --------------------------------------------------------------------------- #
def compute_mutual_information(
    X: pd.DataFrame, y: pd.Series, random_state: int = 42
) -> pd.DataFrame:
    """Rank features by mutual information with the target."""
    y_enc = LabelEncoder().fit_transform(y.astype(str))
    mi = mutual_info_classif(
        X.astype(np.float64), y_enc, random_state=random_state
    )
    ranking = pd.DataFrame({"feature": X.columns, "mutual_information": mi})
    ranking = ranking.sort_values("mutual_information", ascending=False)
    ranking = ranking.reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking


def select_top_k(ranking: pd.DataFrame, k: int = TOP_K) -> list[str]:
    """Return the names of the top-``k`` features by MI."""
    top = ranking.head(k)["feature"].tolist()
    print(f"[select] MI top-{k} -> {top}")
    return top


# --------------------------------------------------------------------------- #
# PCA dimensionality reduction
# --------------------------------------------------------------------------- #
def apply_pca(
    X: pd.DataFrame,
    n_components: int = LATENT_DIM,
    random_state: int = 42,
    artifact_dir: str | Path = ARTIFACT_DIR,
) -> tuple[np.ndarray, PCA, dict]:
    """Fit PCA and reduce ``X`` to ``n_components`` latent dimensions."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    pca = PCA(n_components=n_components, random_state=random_state)
    latent = pca.fit_transform(X)
    joblib.dump(pca, artifact_dir / "pca.pkl")

    stats = {
        "n_components": n_components,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_explained_variance": float(
            np.cumsum(pca.explained_variance_ratio_)[-1]
        ),
        "input_features": int(X.shape[1]),
        "samples": int(X.shape[0]),
    }
    print(f"[pca] latent: {latent.shape}, "
          f"cum. explained variance: {stats['cumulative_explained_variance']:.4f}")
    print(f"[save] artifacts/pca.pkl")
    return latent, pca, stats


# --------------------------------------------------------------------------- #
# Autoencoder alternative (optional)
# --------------------------------------------------------------------------- #
def apply_autoencoder(
    X: pd.DataFrame,
    latent_dim: int = LATENT_DIM,
    epochs: int = 30,
    batch_size: int = 256,
    random_state: int = 42,
) -> tuple[np.ndarray, object, dict]:
    """Train a Keras autoencoder and return (latent, autoencoder_model, history).

    Requires ``tensorflow`` to be installed. Falls back to a PCA path if the
    import fails.
    """
    try:
        import keras  # noqa: F401
        from keras import layers, models
    except ImportError as exc:  # pragma: no cover
        print(f"[!] tensorflow/keras not installed ({exc}). Using PCA instead.")
        return apply_pca(X, n_components=latent_dim, random_state=random_state)

    np.random.seed(random_state)
    data = X.astype(np.float32).values
    input_dim = data.shape[1]

    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(32, activation="relu")(x)
    latent = layers.Dense(latent_dim, activation="linear", name="latent")(x)
    # Decoder
    x = layers.Dense(32, activation="relu")(latent)
    x = layers.Dense(64, activation="relu")(x)
    outputs = layers.Dense(input_dim, activation="linear")(x)

    autoencoder = models.Model(inputs, outputs)
    encoder_model = models.Model(inputs, latent)
    autoencoder.compile(optimizer="adam", loss="mse")

    history = autoencoder.fit(
        data, data,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=0,
    ).history

    latent_vec = encoder_model.predict(data, verbose=0)
    print(f"[autoencoder] latent: {latent_vec.shape}, final loss: "
          f"{history['loss'][-1]:.6f}")
    return latent_vec, autoencoder, history


# --------------------------------------------------------------------------- #
# End-to-end helper
# --------------------------------------------------------------------------- #
def run_feature_selection(
    feature_matrix_path: str | Path = DATA_DIR / "feature_matrix.pkl",
    label_col: str = "label",
    method: str = "pca",          # "pca" or "autoencoder"
    top_k: int = TOP_K,
    latent_dim: int = LATENT_DIM,
    random_state: int = 42,
    artifact_dir: str | Path = ARTIFACT_DIR,
) -> dict:
    """MI -> top-k -> {PCA | autoencoder} -> 10-dim latent representation.

    Expects ``feature_matrix.pkl`` plus ``kdd_clean.csv`` (for labels).
    """
    artifact_dir = Path(artifact_dir)
    manifest = load_artifact_manifest(artifact_dir / "artifact_manifest.json")
    digest, _ = registered_artifact(manifest, Path(feature_matrix_path).name)
    feature_matrix = verified_pandas_read_pickle(
        feature_matrix_path, expected_sha256=digest
    )
    clean = pd.read_csv(DATA_DIR / "kdd_clean.csv")
    y = clean[label_col]

    ranking = compute_mutual_information(feature_matrix, y, random_state)
    top_feats = select_top_k(ranking, top_k)
    X_top = feature_matrix[top_feats]

    if method == "autoencoder":
        latent, model, history = apply_autoencoder(
            X_top, latent_dim=latent_dim, random_state=random_state
        )
    else:
        latent, model, stats = apply_pca(
            X_top, n_components=latent_dim, random_state=random_state,
            artifact_dir=artifact_dir,
        )

    np.save(DATA_DIR / "latent_features.npy", latent)
    ranking.to_csv(DATA_DIR / "mi_ranking.csv", index=False)
    np.save(DATA_DIR / "selected_feature_names.npy", np.array(top_feats))
    print(f"[save] data/latent_features.npy ({latent.shape})")
    print(f"[save] data/mi_ranking.csv")
    return {
        "latent": latent,
        "method": method,
        "top_features": top_feats,
        "ranking": ranking,
        "pca": model if method == "pca" else None,
        "autoencoder": model if method == "autoencoder" else None,
    }

