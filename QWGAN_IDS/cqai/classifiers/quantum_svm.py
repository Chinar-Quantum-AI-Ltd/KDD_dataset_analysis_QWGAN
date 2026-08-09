"""Quantum-Kernel SVM Classifier for FR-5.

Calculates a quantum state overlap kernel matrix using PennyLane's AngleEmbedding
and fits a precomputed-kernel Support Vector Machine (sklearn.svm.SVC).

Includes sub-sampling bounds to handle quadratic O(N^2) kernel scaling on large datasets.
"""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from .base import BaseClassifier

PI = float(np.pi)


def compute_quantum_kernel_matrix(X1: np.ndarray, X2: np.ndarray, n_qubits: int = 10) -> np.ndarray:
    """Compute quantum state overlap kernel matrix between X1 (N1, d) and X2 (N2, d).

    Angles are mapped to [0, pi] and evaluated via PennyLane's AngleEmbedding.
    Kernel value K(x_i, x_j) = |<psi(x_i)|psi(x_j)>|^2.
    """
    try:
        import pennylane as qml
        import torch
    except ImportError as exc:
        raise ImportError("PennyLane is required for QuantumKernelSVM.") from exc

    N1, d1 = X1.shape
    N2, d2 = X2.shape
    qubits = min(d1, n_qubits)

    dev = qml.device("default.qubit", wires=qubits)

    @qml.qnode(dev, interface="torch")
    def kernel_circuit(x1, x2):
        qml.AngleEmbedding(features=x1[:qubits], wires=range(qubits), rotation="Y")
        qml.adjoint(qml.AngleEmbedding)(features=x2[:qubits], wires=range(qubits), rotation="Y")
        return qml.probs(wires=range(qubits))

    X1_t = torch.tensor(np.clip(X1[:, :qubits], 0.0, PI), dtype=torch.float64)
    X2_t = torch.tensor(np.clip(X2[:, :qubits], 0.0, PI), dtype=torch.float64)

    K = np.zeros((N1, N2), dtype=np.float64)
    for i in range(N1):
        for j in range(N2):
            probs = kernel_circuit(X1_t[i], X2_t[j]).detach().numpy()
            K[i, j] = probs[0]  # Probability of ground state |0...0>

    return K


class QuantumKernelSVM(BaseClassifier):
    """Quantum-Kernel SVM Classifier for tabular network flow data."""

    def __init__(
        self,
        n_qubits: int = 10,
        max_samples: int = 2000,
        C: float = 1.0,
        random_state: int = 42,
    ) -> None:
        super().__init__(name="quantum_kernel_svm", random_state=random_state)
        self.n_qubits = n_qubits
        self.max_samples = max_samples
        self.C = C
        self.label_encoder = LabelEncoder()
        self.svc = SVC(kernel="precomputed", C=self.C, probability=True, random_state=self.random_state)
        self.X_train_sub: np.ndarray | None = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> QuantumKernelSVM:
        X_arr = (X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)).astype(np.float64)
        y_arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)

        # Sub-sample if dataset exceeds max_samples (O(N^2) kernel protection)
        N = len(X_arr)
        if N > self.max_samples:
            rng = np.random.default_rng(self.random_state)
            indices = rng.choice(N, size=self.max_samples, replace=False)
            indices.sort()
            X_arr = X_arr[indices]
            y_arr = y_arr[indices]

        self.X_train_sub = X_arr
        y_encoded = self.label_encoder.fit_transform(y_arr)
        self.classes_ = self.label_encoder.classes_

        # Compute Gram matrix K(X_train, X_train)
        K_train = compute_quantum_kernel_matrix(self.X_train_sub, self.X_train_sub, n_qubits=self.n_qubits)
        self.svc.fit(K_train, y_encoded)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.X_train_sub is None:
            raise RuntimeError("Model is not fitted.")
        X_arr = (X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)).astype(np.float64)

        K_test = compute_quantum_kernel_matrix(X_arr, self.X_train_sub, n_qubits=self.n_qubits)
        return self.svc.predict_proba(K_test)

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.X_train_sub is None:
            raise RuntimeError("Model is not fitted.")
        X_arr = (X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)).astype(np.float64)

        K_test = compute_quantum_kernel_matrix(X_arr, self.X_train_sub, n_qubits=self.n_qubits)
        preds_int = self.svc.predict(K_test)
        return self.label_encoder.inverse_transform(preds_int)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> QuantumKernelSVM:
        obj = joblib.load(path)
        if not isinstance(obj, QuantumKernelSVM):
            raise TypeError(f"Loaded object is not QuantumKernelSVM: {type(obj)}")
        return obj
