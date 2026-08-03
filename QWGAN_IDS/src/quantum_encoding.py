"""Quantum-ready angle encoding (FR-2).

Design (per Section 7 / Section 8 of the spec):

    latent_features -> MinMaxScaler -> [0, 1] -> angles = x * pi -> [0, pi]

Invertible decoding:

    angles -> angles / pi -> inverse MinMax -> inverse PCA
           -> inverse RobustScaler -> inverse log1p -> decoded features

Persists:
    data/angles.npy              (angle-encoded latent features)
    artifacts/minmax_scaler.pkl  (fitted MinMaxScaler)
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from .encoding import inverse_numeric, numeric_cols_of
from .feature_selection import LATENT_DIM

DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts")

PI = float(np.pi)


# --------------------------------------------------------------------------- #
# Angle encoding / decoding
# --------------------------------------------------------------------------- #
def fit_minmax(latent: np.ndarray,
               artifact_dir: str | Path = ARTIFACT_DIR) -> MinMaxScaler:
    """Fit a MinMaxScaler on the latent features and persist it."""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    scaler = MinMaxScaler().fit(latent)
    joblib.dump(scaler, artifact_dir / "minmax_scaler.pkl")
    print(f"[angle] fitted MinMaxScaler on {latent.shape} latent vectors")
    return scaler


def angle_encode(latent: np.ndarray,
                 scaler: MinMaxScaler | None = None,
                 artifact_dir: str | Path = ARTIFACT_DIR) -> np.ndarray:
    """Map latent features to angles in ``[0, pi]``.

    ``angles = MinMax(latent) * pi``
    """
    if scaler is None:
        scaler = MinMaxScaler().fit(latent)
    normalized = scaler.transform(latent)
    angles = normalized * PI
    angles = np.clip(angles, 0.0, PI)  # numerical safety
    print(f"[angle] angles: shape={angles.shape}, range=({angles.min():.4f}, "
          f"{angles.max():.4f})")
    return angles


def angle_decode(angles: np.ndarray,
                 scaler: MinMaxScaler,
                 pca,
                 robust_scaler,
                 numeric_cols: list[str],
                 top_feature_names: list[str]) -> np.ndarray:
    """Invert the full latent pipeline back to (approx.) feature values.

    ``angles -> angles/pi -> inverse MinMax -> inverse PCA -> inverse
    RobustScaler -> inverse log1p``

    Numeric columns are fully inverted (RobustScaler + log1p). Non-numeric
    (one-hot) columns among ``top_feature_names`` are kept as the inverse-PCA
    values so the returned array always has one column per selected feature.

    Returns a 2-D array whose columns correspond to ``top_feature_names``
    (the features selected before PCA).
    """
    angles = np.asarray(angles, dtype=np.float64)
    normalized = angles / PI
    latent = scaler.inverse_transform(normalized)
    feature_space = pca.inverse_transform(latent)
    feature_df = pd.DataFrame(feature_space, columns=top_feature_names)

    # Numeric columns: inverse RobustScaler + inverse log1p.
    decoded_numeric = inverse_numeric(feature_df, robust_scaler, numeric_cols)
    # Non-numeric (one-hot) columns: keep the inverse-PCA values so the
    # returned matrix has exactly one column per ``top_feature_names`` entry.
    non_numeric = [c for c in top_feature_names if c not in numeric_cols]
    decoded = pd.concat(
        [decoded_numeric.reset_index(drop=True),
         feature_df[non_numeric].reset_index(drop=True)],
        axis=1,
    )[top_feature_names]
    return decoded.values


# --------------------------------------------------------------------------- #
# PennyLane quantum verification (FR-2 / Notebook 6)
# --------------------------------------------------------------------------- #
def verify_quantum_circuit(
    angles: np.ndarray,
    n_samples: int = 8,
    n_layers: int = 2,
    random_state: int = 42,
) -> dict:
    """Build a 10-qubit AngleEmbedding circuit with StronglyEntanglingLayers
    and measure Pauli-Z expectations.

    Verifies:
      * angles are valid (finite, in [0, pi])
      * PennyLane accepts them
      * the circuit output is differentiable

    Returns a stats dict. ``requires_torch`` controls the autodiff backend.
    """
    try:
        import pennylane as qml
        import torch
    except ImportError as exc:
        raise ImportError(
            "PennyLane is required for quantum verification. "
            f"Install with: pip install pennylane (got: {exc})"
        ) from exc

    np.random.seed(random_state)
    torch.manual_seed(random_state)
    angles = np.asarray(angles, dtype=np.float64)

    # ---- validation ------------------------------------------------------- #
    finite = bool(np.isfinite(angles).all())
    in_range = bool((angles >= 0).all() and (angles <= PI).all())
    n_qubits = int(angles.shape[1])
    if not finite:
        raise ValueError("Angles contain NaN/Inf - encoding invalid.")
    if not in_range:
        raise ValueError(f"Angles out of [0, pi]: "
                         f"min={angles.min():.4f}, max={angles.max():.4f}")

    # ---- circuit ---------------------------------------------------------- #
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(x):
        qml.AngleEmbedding(features=x, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(
            weights=torch.randn(n_layers, n_qubits, 3, requires_grad=True),
            wires=range(n_qubits),
        )
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    # ---- forward pass ----------------------------------------------------- #
    sample = torch.tensor(angles[:n_samples], dtype=torch.float64,
                          requires_grad=True)
    # PennyLane returns a list of per-qubit expectation tensors; stack them
    # into (n_qubits, batch) and transpose to (batch, n_qubits).
    expectations = torch.stack(circuit(sample)).detach().numpy().T

    # ---- differentiability (gradient w.r.t. input angles) ----------------- #
    # Use torch autograd: qml.grad on a list-returning torch QNode is not
    # supported by PennyLane >= 0.34.
    torch.stack(circuit(sample)).sum().backward()
    grad_np = sample.grad.detach().numpy()
    nonzero_grad = float(np.abs(grad_np).sum())

    stats = {
        "n_qubits": n_qubits,
        "n_angles": int(angles.shape[0]),
        "angles_finite": finite,
        "angles_in_range": in_range,
        "n_samples_forward": int(sample.shape[0]),
        "expectation_shape": expectations.shape,
        "expectation_range": [float(expectations.min()),
                              float(expectations.max())],
        "gradient_abs_sum": nonzero_grad,
        "differentiable": bool(np.isfinite(grad_np).all())
        and nonzero_grad > 0.0,
        "interface": "torch",
        "diff_method": "backprop",
    }
    print(f"[quantum] {stats}")
    return stats


# --------------------------------------------------------------------------- #
# End-to-end helper
# --------------------------------------------------------------------------- #
def run_angle_encoding(
    latent_path: str | Path = DATA_DIR / "latent_features.npy",
    artifact_dir: str | Path = ARTIFACT_DIR,
    data_dir: str | Path = DATA_DIR,
) -> dict:
    """Load latent features, fit MinMax, compute angles, persist both."""
    data_dir = Path(data_dir)
    latent = np.load(latent_path)
    scaler = fit_minmax(latent, artifact_dir)
    angles = angle_encode(latent, scaler, artifact_dir)
    np.save(data_dir / "angles.npy", angles)
    print(f"[save] data/angles.npy ({angles.shape})")
    return {"latent": latent, "angles": angles, "minmax_scaler": scaler}


def run_inverse_verification(
    angles_path: str | Path = DATA_DIR / "angles.npy",
    artifact_dir: str | Path = ARTIFACT_DIR,
    data_dir: str | Path = DATA_DIR,
) -> np.ndarray:
    """Reconstruct feature values from angles and return the decoded matrix."""
    angles = np.load(angles_path)
    minmax = joblib.load(artifact_dir / "minmax_scaler.pkl")
    pca = joblib.load(artifact_dir / "pca.pkl")
    robust = joblib.load(artifact_dir / "robust_scaler.pkl")

    top_names = np.load(data_dir / "selected_feature_names.npy",
                        allow_pickle=True).tolist()
    numeric_cols = numeric_cols_of(top_names)

    decoded = angle_decode(angles, minmax, pca, robust, numeric_cols, top_names)
    print(f"[decode] decoded feature matrix: {decoded.shape}")
    return decoded

