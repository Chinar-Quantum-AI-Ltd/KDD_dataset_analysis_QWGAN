"""PyTorch Fully Connected Deep Neural Network Classifier for FR-5.

Provides a multi-layer PyTorch MLP adhering to ``BaseClassifier``.
"""
from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .base import BaseClassifier


class _FCMLP(nn.Module):
    """PyTorch FC Multi-Layer Perceptron."""

    def __init__(self, in_features: int, out_features: int, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PyTorchDNNClassifier(BaseClassifier):
    """PyTorch Fully Connected DNN Classifier for tabular network flow data."""

    def __init__(
        self,
        hidden_dim: int = 128,
        epochs: int = 20,
        batch_size: int = 256,
        lr: float = 1e-3,
        dropout: float = 0.2,
        random_state: int = 42,
    ) -> None:
        super().__init__(name="pytorch_dnn", random_state=random_state)
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout = dropout
        self.label_encoder = LabelEncoder()
        self.net: _FCMLP | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> PyTorchDNNClassifier:
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_arr = (X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)).astype(np.float32)
        y_arr = y.to_numpy() if isinstance(y, pd.Series) else np.asarray(y)

        y_encoded = self.label_encoder.fit_transform(y_arr)
        self.classes_ = self.label_encoder.classes_
        n_classes = len(self.classes_)
        in_features = X_arr.shape[1]

        self.net = _FCMLP(in_features, n_classes, hidden_dim=self.hidden_dim, dropout=self.dropout).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        dataset = TensorDataset(torch.tensor(X_arr), torch.tensor(y_encoded, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=min(self.batch_size, len(dataset)), shuffle=True)

        self.net.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                out = self.net(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        if not self.is_fitted or self.net is None:
            raise RuntimeError("Model is not fitted.")
        X_arr = (X.to_numpy() if isinstance(X, pd.DataFrame) else np.asarray(X)).astype(np.float32)
        
        self.net.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X_arr).to(self.device)
            logits = self.net(tensor_x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        preds_int = np.argmax(probs, axis=1)
        return self.label_encoder.inverse_transform(preds_int)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> PyTorchDNNClassifier:
        obj = joblib.load(path)
        if not isinstance(obj, PyTorchDNNClassifier):
            raise TypeError(f"Loaded object is not PyTorchDNNClassifier: {type(obj)}")
        return obj
