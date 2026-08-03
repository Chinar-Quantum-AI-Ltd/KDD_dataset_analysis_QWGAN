"""End-to-end QWGAN_IDS pipeline runner.

Runs the full FR-1 / FR-2 chain:

    load -> clean -> encode -> feature-select -> angle-encode -> verify

Usage
-----
    python run_pipeline.py                 # run every stage
    python run_pipeline.py --stages load   # run a single stage
    python run_pipeline.py --stages load clean encode
    python run_pipeline.py --skip-download # assume datasets/ already exists
    python run_pipeline.py --method autoencoder   # use autoencoder instead of PCA
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import loader, preprocessing
from src.encoding import build_feature_matrix
from src.feature_selection import run_feature_selection
from src.quantum_encoding import (
    run_angle_encoding,
    run_inverse_verification,
    verify_quantum_circuit,
)

DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts")
DATASETS_DIR = Path("datasets")


def stage_load(skip_download: bool = False) -> pd.DataFrame:
    if skip_download and (DATASETS_DIR / "KDDTrain+.txt").exists():
        df = loader.load_all(DATASETS_DIR, merge=True, download_if_missing=False)
    else:
        df = loader.load_all(DATASETS_DIR, merge=True)
    loader.inspect_dataframe(df)
    loader.save_raw(df, DATA_DIR / "kdd_raw.csv")
    return df


def stage_clean() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "kdd_raw.csv")
    # Raw (merged) frame may contain duplicates — tolerate them pre-clean.
    checks_raw = preprocessing.verify_schema(df, strict=False)
    print(f"[clean] raw duplicates: {checks_raw['duplicates']}")
    df = preprocessing.clean_dataset(df)
    # After cleaning, duplicates and missing values must both be zero.
    checks_clean = preprocessing.verify_schema(df, strict=True)
    roles = preprocessing.identify_roles(df)
    print(f"[roles] {roles}")
    preprocessing.save_clean(df, DATA_DIR / "kdd_clean.csv")
    return df


def stage_encode() -> None:
    df = pd.read_csv(DATA_DIR / "kdd_clean.csv")
    build_feature_matrix(df, artifact_dir=ARTIFACT_DIR)


def stage_select(method: str) -> None:
    run_feature_selection(
        feature_matrix_path=DATA_DIR / "feature_matrix.pkl",
        method=method,
        artifact_dir=ARTIFACT_DIR,
    )


def stage_angles() -> None:
    run_angle_encoding(DATA_DIR / "latent_features.npy",
                       artifact_dir=ARTIFACT_DIR,
                       data_dir=DATA_DIR)


def stage_verify() -> None:
    angles = np.load(DATA_DIR / "angles.npy")
    stats = verify_quantum_circuit(angles)
    print(f"[verify] differentiability ok: {stats['differentiable']}")
    decoded = run_inverse_verification(DATA_DIR / "angles.npy",
                                       artifact_dir=ARTIFACT_DIR,
                                       data_dir=DATA_DIR)
    np.save(DATA_DIR / "decoded_features.npy", decoded)


STAGES = {
    "load": lambda args: stage_load(args.skip_download),
    "clean": lambda args: stage_clean(),
    "encode": lambda args: stage_encode(),
    "select": lambda args: stage_select(args.method),
    "angles": lambda args: stage_angles(),
    "verify": lambda args: stage_verify(),
}


def main() -> None:
    parser = argparse.ArgumentParser(description="QWGAN_IDS NSL-KDD pipeline")
    parser.add_argument("--stages", nargs="+",
                        default=list(STAGES),
                        choices=list(STAGES),
                        help="Stages to run (default: all).")
    parser.add_argument("--skip-download", action="store_true",
                        help="Do not download datasets if already present.")
    parser.add_argument("--method", choices=["pca", "autoencoder"],
                        default="pca",
                        help="Dimensionality reduction method.")
    args = parser.parse_args()

    for name in args.stages:
        print(f"\n{'=' * 60}\nSTAGE: {name}\n{'=' * 60}")
        STAGES[name](args)


if __name__ == "__main__":
    main()

