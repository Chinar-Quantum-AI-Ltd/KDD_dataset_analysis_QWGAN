"""Data cleaning and validation (FR-1 pipeline stage 2).

Transforms ``data/kdd_raw.csv`` -> ``data/kdd_clean.csv`` by

* removing duplicate rows
* removing rows with missing values
* verifying the 43-column NSL-KDD schema
* identifying continuous / categorical / binary / label feature roles
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .loader import COLUMN_NAMES

# Feature-role tables (KDD Cup '99 / NSL-KDD conventions).
BINARY_COLS = [
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
]
CATEGORICAL_COLS = ["protocol_type", "service", "flag"]
LABEL_COLS = ["label", "difficulty"]
CONTINUOUS_COLS = [
    c for c in COLUMN_NAMES if c not in CATEGORICAL_COLS + BINARY_COLS + LABEL_COLS
]

DATA_DIR = Path("data")
ARTIFACT_DIR = Path("artifacts")


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    out = df.drop_duplicates().reset_index(drop=True)
    print(f"[clean] removed {before - len(out)} duplicate rows "
          f"({len(out)} remain)")
    return out


def drop_missing(df: pd.DataFrame, how: str = "any") -> pd.DataFrame:
    before = len(df)
    out = df.dropna(how=how).reset_index(drop=True)
    print(f"[clean] removed {before - len(out)} rows with missing values "
          f"({len(out)} remain)")
    return out


def verify_schema(df: pd.DataFrame, strict: bool = True) -> dict:
    """Verify the dataframe matches the 43-column NSL-KDD schema.

    Parameters
    ----------
    strict : bool
        If ``True`` (default), the dataframe must also have zero duplicate
        rows and zero missing values. If ``False``, duplicate rows are
        reported but tolerated — use this on the **raw** merged dataframe
        *before* cleaning, where duplicates are expected.

    Raises
    ------
    ValueError
        If any check fails (or, in non-strict mode, if column order, missing
        values, or empty strings are detected).
    """
    checks = {
        "column_count": int(df.shape[1]),
        "expected_columns": len(COLUMN_NAMES),
        "columns_match": list(df.columns) == COLUMN_NAMES,
        "missing_values": int(df.isna().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "empty_strings": int((df == "").sum().sum()),
        "strict": bool(strict),
    }
    base_ok = bool(
        checks["columns_match"]
        and checks["missing_values"] == 0
        and checks["empty_strings"] == 0
    )
    checks["passed"] = bool(base_ok and (not strict or checks["duplicates"] == 0))
    print(f"[schema] {checks}")
    if not checks["passed"]:
        raise ValueError("Schema verification failed. See `checks` dict.")
    return checks


def identify_roles(df: pd.DataFrame | None = None) -> dict:
    """Return the feature-role split used downstream.

    When ``df`` is provided, also validates that the expected columns exist
    and reports per-role dtype compatibility (categoricals / label should be
    ``object`` strings).
    """
    roles = {
        "continuous": list(CONTINUOUS_COLS),
        "categorical": list(CATEGORICAL_COLS),
        "binary": list(BINARY_COLS),
        "label": ["label"],
        "metadata": ["difficulty"],
    }
    if df is not None:
        missing = [c for c in COLUMN_NAMES if c not in df.columns]
        cat_dtype_ok = all(
            pd.api.types.is_object_dtype(df[c])
            for c in CATEGORICAL_COLS
            if c in df.columns
        )
        label_dtype_ok = (
            "label" in df.columns
            and pd.api.types.is_object_dtype(df["label"])
        )
        roles["dtype_check"] = {
            "categorical_is_object": bool(cat_dtype_ok),
            "label_is_object": bool(label_dtype_ok),
            "roles_cover_all_columns": not missing,
            "missing_columns": list(missing),
        }
    return roles


def clean_dataset(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Apply the full cleaning chain (duplicates -> missing values)."""
    if verbose:
        print(f"[load] input shape: {df.shape}")
    df = drop_duplicates(df)
    df = drop_missing(df)
    if verbose:
        print(f"[load] cleaned shape: {df.shape}")
    return df


def save_clean(df: pd.DataFrame,
               out_path: str | Path = "data/kdd_clean.csv") -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[save] -> {out_path} ({df.shape[0]} rows, {df.shape[1]} cols)")
    return out_path

