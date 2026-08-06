"""Leakage-safe, versioned NSL-KDD train-only contract for FR-3.

Why this exists
---------------
The checked-in ``data/angles.npy`` was produced by merging ``KDDTrain+.txt``
and ``KDDTest+.txt`` and fitting every transform on the merge. That is
evaluation leakage, so ``TrainingAngles`` rejects it and FR-3 must not train on
it. This module rebuilds the same FR-2 transform chain

    one-hot(categoricals) + log1p -> RobustScaler(numerics)
        -> mutual-information top-k -> PCA(n_qubits) -> MinMax -> * pi

with one difference that matters: **every transform is fitted on the training
partition alone**, and ``KDDTest+.txt`` is never opened. The held-out real test
set therefore stays untouched and immutable for FR-5/FR-6.

The teammate-owned ``src/`` helpers are imported read-only for the role tables
and the decode path. Nothing here writes into ``data/`` or ``artifacts/``.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    RobustScaler,
)

from src.encoding import log1p_transform, numeric_cols_of
from src.loader import COLUMN_NAMES
from src.preprocessing import BINARY_COLS, CATEGORICAL_COLS, CONTINUOUS_COLS
from src.quantum_encoding import angle_decode

from ..qwgan.config import QWGANConfig
from ..qwgan.data_contract import TrainingAngles

CONTRACT_ID = "nslkdd-train-only"
CONTRACT_VERSION = "1.0.0"
SCHEMA_VERSION = "1.0.0"
PI = float(np.pi)

#: NSL-KDD raw label -> canonical attack family. Sources: the KDD Cup '99
#: task description plus the NSL-KDD additions present in KDDTest+.
ATTACK_FAMILIES: dict[str, str] = {
    "normal": "normal",
    # Denial of service
    "back": "dos",
    "land": "dos",
    "neptune": "dos",
    "pod": "dos",
    "smurf": "dos",
    "teardrop": "dos",
    "apache2": "dos",
    "udpstorm": "dos",
    "processtable": "dos",
    "mailbomb": "dos",
    # Probe
    "satan": "probe",
    "ipsweep": "probe",
    "nmap": "probe",
    "portsweep": "probe",
    "mscan": "probe",
    "saint": "probe",
    # Remote to local
    "guess_passwd": "r2l",
    "ftp_write": "r2l",
    "imap": "r2l",
    "phf": "r2l",
    "multihop": "r2l",
    "warezmaster": "r2l",
    "warezclient": "r2l",
    "spy": "r2l",
    "xlock": "r2l",
    "xsnoop": "r2l",
    "snmpguess": "r2l",
    "snmpgetattack": "r2l",
    "httptunnel": "r2l",
    "sendmail": "r2l",
    "named": "r2l",
    # User to root
    "buffer_overflow": "u2r",
    "loadmodule": "u2r",
    "rootkit": "u2r",
    "perl": "u2r",
    "sqlattack": "u2r",
    "xterm": "u2r",
    "ps": "u2r",
}

#: Never fed to a transform: ``difficulty`` is NSL-KDD grading metadata, not a
#: network-flow feature, and using it would be identifier leakage.
EXCLUDED_COLUMNS = ("label", "difficulty")

_PARTITIONS = ("train", "val")


@dataclass(frozen=True, slots=True)
class ContractSpec:
    """Everything that determines the contract's content, and nothing else."""

    n_qubits: int = 10
    top_k: int = 20
    val_fraction: float = 0.2
    split_seed: int = 42
    transform_seed: int = 42
    mi_sample_cap: int = 20_000

    def __post_init__(self) -> None:
        if not 8 <= self.n_qubits <= 12:
            raise ValueError("n_qubits must be between 8 and 12")
        if self.top_k < self.n_qubits:
            raise ValueError("top_k must be at least n_qubits")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must be strictly between 0 and 1")
        if self.mi_sample_cap < 1:
            raise ValueError("mi_sample_cap must be positive")


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of a file, streamed so large artefacts stay cheap."""

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _count_lines(path: str | Path) -> int:
    with open(path, "rb") as handle:
        return sum(1 for _ in handle)


class TrainContract:
    """A built contract on disk, plus the accessors FR-3/FR-4 consume."""

    def __init__(self, root: str | Path, manifest: dict[str, Any]) -> None:
        self.root = Path(root)
        self.manifest = manifest

    # -- identity ---------------------------------------------------------- #
    @property
    def spec(self) -> ContractSpec:
        return ContractSpec(**self.manifest["spec"])

    @property
    def latent_columns(self) -> tuple[str, ...]:
        return tuple(self.manifest["latent_columns"])

    @property
    def angle_range(self) -> tuple[float, float]:
        low, high = self.manifest["angle_range"]
        return float(low), float(high)

    @property
    def source_sha256(self) -> str:
        return str(self.manifest["source_sha256"])

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self.manifest["metadata"])

    @property
    def attack_classes(self) -> tuple[str, ...]:
        return tuple(self.manifest["attack_families"])

    # -- partitions -------------------------------------------------------- #
    def _load(self, name: str) -> np.ndarray:
        return np.load(self.root / name, allow_pickle=False)

    def angles(self, partition: str) -> np.ndarray:
        self._require_partition(partition)
        return self._load(f"angles_{partition}.npy")

    def row_index(self, partition: str) -> np.ndarray:
        """Row positions in the *source file* that fed this partition."""

        self._require_partition(partition)
        return self._load(f"row_index_{partition}.npy")

    def families(self, partition: str) -> np.ndarray:
        self._require_partition(partition)
        return np.load(
            self.root / f"families_{partition}.npy", allow_pickle=True
        )

    def raw_labels(self, partition: str) -> np.ndarray:
        self._require_partition(partition)
        return np.load(
            self.root / f"raw_labels_{partition}.npy", allow_pickle=True
        )

    def class_counts(self, partition: str) -> dict[str, int]:
        self._require_partition(partition)
        return {
            name: int(count)
            for name, count in self.manifest["partitions"][partition][
                "class_counts"
            ].items()
        }

    @staticmethod
    def _require_partition(partition: str) -> None:
        if partition not in _PARTITIONS:
            raise KeyError(
                f"unknown partition {partition!r}; the contract carries "
                f"{_PARTITIONS} only — the real test set is never built here"
            )

    def _class_rows(self, partition: str, attack_class: str) -> np.ndarray:
        if attack_class not in self.attack_classes:
            raise KeyError(
                f"{attack_class!r} is not in this contract; available classes: "
                f"{self.attack_classes}"
            )
        families = self.families(partition)
        return self.angles(partition)[families == attack_class]

    # -- FR-3 / FR-4 consumption ------------------------------------------- #
    def training_angles(
        self, attack_class: str, *, config: QWGANConfig
    ) -> TrainingAngles:
        """Return the validated train-only batch for one attack class."""

        rows = self._class_rows("train", attack_class)
        if rows.shape[0] == 0:
            raise ValueError(
                f"no training rows for class {attack_class!r}; refusing to "
                "fabricate a per-class generator"
            )
        return TrainingAngles.from_array(
            rows,
            config=config,
            partition="train",
            attack_class=attack_class,
            latent_columns=self.latent_columns,
        )

    def validation_angles(self, attack_class: str) -> np.ndarray:
        """Held-out real rows for this class.

        These exist for FR-4 fidelity gating. They are deliberately returned as
        a plain array: ``TrainingAngles`` refuses anything but ``train``, so
        this cannot be fed to a trainer by accident.
        """

        return self._class_rows("val", attack_class)

    # -- decode path (FR-2 inverse, documented as lossy) -------------------- #
    def decode_angles(self, angles: np.ndarray) -> pd.DataFrame:
        """Invert angles back toward the selected feature space.

        The chain is ``angles/pi -> inverse MinMax -> inverse PCA -> inverse
        RobustScaler -> expm1``. It is **approximate**: PCA and the top-k
        selection both discard information, so this recovers the selected
        features only, not the full 41-column NSL-KDD record.
        """

        selected = list(self.manifest["selected_features"])
        decoded = angle_decode(
            np.asarray(angles, dtype=np.float64),
            joblib.load(self.root / "minmax_scaler.joblib"),
            joblib.load(self.root / "pca.joblib"),
            joblib.load(self.root / "robust_scaler.joblib"),
            numeric_cols_of(selected),
            selected,
        )
        return pd.DataFrame(decoded, columns=selected)


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def _read_and_clean(source: Path) -> pd.DataFrame:
    frame = pd.read_csv(source, header=None, names=COLUMN_NAMES)
    frame["_source_row"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.drop_duplicates(subset=COLUMN_NAMES, keep="first")
    frame = frame.dropna(subset=COLUMN_NAMES, how="any")
    return frame.reset_index(drop=True)


def _map_families(labels: pd.Series) -> pd.Series:
    unknown = sorted(set(labels) - set(ATTACK_FAMILIES))
    if unknown:
        raise ValueError(
            "unmapped NSL-KDD labels; add them to ATTACK_FAMILIES rather than "
            f"letting them fall into a silent bucket: {unknown}"
        )
    return labels.map(ATTACK_FAMILIES)


def _stratified_split(
    labels: pd.Series, *, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split positions per raw label so rare attacks appear in both parts.

    A label with a single row stays entirely in train: a held-out row that
    leaves the generator with nothing to learn from helps nobody.
    """

    train_positions: list[int] = []
    val_positions: list[int] = []
    for label in sorted(labels.unique()):
        positions = np.flatnonzero(labels.to_numpy() == label)
        # Seed per label so adding a class cannot reshuffle the others.
        digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
        shuffled = rng.permutation(positions)
        n_val = int(np.floor(val_fraction * len(shuffled)))
        if len(shuffled) >= 2:
            n_val = max(1, min(n_val, len(shuffled) - 1))
        else:
            n_val = 0
        val_positions.extend(shuffled[:n_val].tolist())
        train_positions.extend(shuffled[n_val:].tolist())
    return np.sort(np.array(train_positions, dtype=np.int64)), np.sort(
        np.array(val_positions, dtype=np.int64)
    )


def _encode_features(
    frame: pd.DataFrame,
    *,
    encoder: OneHotEncoder,
    scaler: RobustScaler,
    fit: bool,
) -> pd.DataFrame:
    numeric_cols = list(CONTINUOUS_COLS) + list(BINARY_COLS)
    categorical = frame[list(CATEGORICAL_COLS)]
    logged = log1p_transform(frame[numeric_cols], numeric_cols)

    if fit:
        encoded = encoder.fit_transform(categorical)
        scaled = scaler.fit_transform(logged)
    else:
        encoded = encoder.transform(categorical)
        scaled = scaler.transform(logged)

    categorical_names = encoder.get_feature_names_out(list(CATEGORICAL_COLS))
    return pd.concat(
        [
            pd.DataFrame(scaled, columns=numeric_cols, index=frame.index),
            pd.DataFrame(encoded, columns=categorical_names, index=frame.index),
        ],
        axis=1,
    )


def _select_features(
    matrix: pd.DataFrame, labels: pd.Series, spec: ContractSpec
) -> tuple[list[str], pd.DataFrame]:
    """Rank features by mutual information on the training partition only."""

    if len(matrix) > spec.mi_sample_cap:
        rng = np.random.default_rng(spec.transform_seed)
        sample = rng.choice(len(matrix), size=spec.mi_sample_cap, replace=False)
        sample.sort()
        ranking_matrix = matrix.iloc[sample]
        ranking_labels = labels.iloc[sample]
    else:
        ranking_matrix = matrix
        ranking_labels = labels

    encoded_labels = LabelEncoder().fit_transform(ranking_labels.astype(str))
    scores = mutual_info_classif(
        ranking_matrix.astype(np.float64),
        encoded_labels,
        random_state=spec.transform_seed,
    )
    ranking = pd.DataFrame(
        {"feature": list(matrix.columns), "mutual_information": scores}
    ).sort_values(
        ["mutual_information", "feature"], ascending=[False, True]
    ).reset_index(drop=True)
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    return ranking.head(spec.top_k)["feature"].tolist(), ranking


def build_train_contract(
    source: str | Path,
    output_dir: str | Path,
    *,
    spec: ContractSpec | None = None,
) -> TrainContract:
    """Build a leakage-safe train-only angle contract from ``KDDTrain+.txt``.

    ``source`` must be the NSL-KDD **training** file. There is deliberately no
    parameter for a test file: this builder cannot see one, so it cannot fit on
    one.
    """

    spec = spec or ContractSpec()
    source = Path(source)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    frame = _read_and_clean(source)
    families = _map_families(frame["label"])
    train_positions, val_positions = _stratified_split(
        frame["label"], val_fraction=spec.val_fraction, seed=spec.split_seed
    )

    train_frame = frame.iloc[train_positions].reset_index(drop=True)
    val_frame = frame.iloc[val_positions].reset_index(drop=True)

    # ---- fit on train only -------------------------------------------------
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    robust = RobustScaler()
    train_matrix = _encode_features(
        train_frame, encoder=encoder, scaler=robust, fit=True
    )
    val_matrix = _encode_features(
        val_frame, encoder=encoder, scaler=robust, fit=False
    )

    selected, ranking = _select_features(
        train_matrix, train_frame["label"], spec
    )

    pca = PCA(n_components=spec.n_qubits, random_state=spec.transform_seed)
    train_latent = pca.fit_transform(train_matrix[selected])
    val_latent = pca.transform(val_matrix[selected])

    minmax = MinMaxScaler().fit(train_latent)
    train_angles = np.clip(minmax.transform(train_latent) * PI, 0.0, PI)
    # Validation latents can fall outside the training range; clipping keeps
    # the [0, pi] consumer invariant and the amount clipped is reported.
    raw_val_angles = minmax.transform(val_latent) * PI
    val_angles = np.clip(raw_val_angles, 0.0, PI)
    val_clipped = int(np.count_nonzero(raw_val_angles != val_angles))

    # ---- persist -----------------------------------------------------------
    partitions = {
        "train": (train_frame, train_angles, train_positions),
        "val": (val_frame, val_angles, val_positions),
    }
    for name, (part_frame, angles, _positions) in partitions.items():
        np.save(root / f"angles_{name}.npy", angles.astype(np.float64))
        np.save(
            root / f"row_index_{name}.npy",
            part_frame["_source_row"].to_numpy(dtype=np.int64),
        )
        np.save(
            root / f"families_{name}.npy",
            families.iloc[_positions].to_numpy().astype(object),
        )
        np.save(
            root / f"raw_labels_{name}.npy",
            part_frame["label"].to_numpy().astype(object),
        )

    joblib.dump(encoder, root / "encoder.joblib")
    joblib.dump(robust, root / "robust_scaler.joblib")
    joblib.dump(pca, root / "pca.joblib")
    joblib.dump(minmax, root / "minmax_scaler.joblib")
    ranking.to_csv(root / "mi_ranking.csv", index=False)

    latent_columns = tuple(f"z{i}" for i in range(spec.n_qubits))
    manifest: dict[str, Any] = {
        "contract_id": CONTRACT_ID,
        "contract_version": CONTRACT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_id": "nsl-kdd",
        "source_file": source.name,
        "source_sha256": sha256_file(source),
        "spec": asdict(spec),
        "latent_columns": list(latent_columns),
        "angle_range": [0.0, PI],
        "dtype": "float64",
        "excluded_columns": list(EXCLUDED_COLUMNS),
        "selected_features": list(selected),
        "explained_variance_ratio": [
            float(value) for value in pca.explained_variance_ratio_
        ],
        "cumulative_explained_variance": float(
            np.sum(pca.explained_variance_ratio_)
        ),
        "attack_families": sorted(set(families)),
        "fitted_on": "train",
        "partitions": {
            name: {
                "rows": int(angles.shape[0]),
                "class_counts": {
                    str(cls): int(count)
                    for cls, count in families.iloc[_positions]
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "label_counts": {
                    str(label): int(count)
                    for label, count in part_frame["label"]
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "angles_sha256": sha256_file(root / f"angles_{name}.npy"),
            }
            for name, (part_frame, angles, _positions) in partitions.items()
        },
        "metadata": {
            "source_rows": _count_lines(source),
            "cleaned_rows": int(len(frame)),
            "validation_angles_clipped": val_clipped,
        },
    }
    manifest["artifact_sha256"] = {
        path.name: sha256_file(path)
        for path in sorted(root.iterdir())
        if path.is_file() and path.name != "contract.json"
    }

    (root / "contract.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return TrainContract(root, manifest)


def load_train_contract(root: str | Path) -> TrainContract:
    """Load a previously built contract and re-verify its artefact hashes."""

    root = Path(root)
    manifest = json.loads((root / "contract.json").read_text(encoding="utf-8"))

    for name, expected in manifest["artifact_sha256"].items():
        path = root / name
        if not path.exists():
            raise FileNotFoundError(f"contract artefact missing: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"contract artefact {name} changed after it was built "
                f"(expected {expected}, found {actual})"
            )
    return TrainContract(root, manifest)
