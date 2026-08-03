"""NSL-KDD dataset loading utilities (FR-1 pipeline stage 1).

    NSL-KDD CSV -> Load Dataset -> Data Validation -> Clean Dataset -> ...

Implements:
    * downloading KDDTrain+.txt / KDDTest+.txt (with mirror fallback)
    * loading with the official 43-column schema
    * optional train/test merge
    * quick EDA helpers
    * saving the raw dataframe -> data/kdd_raw.csv
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pandas as pd

# --------------------------------------------------------------------------- #
# NSL-KDD 43-column schema (41 features + label + difficulty)
# --------------------------------------------------------------------------- #
COLUMN_NAMES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

# Primary URL + fallback mirrors per dataset file.
DATASET_URLS = {
    "KDDTrain+.txt": [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt",
        "https://raw.githubusercontent.com/arindam036/NSL-KDD-Dataset/master/KDDTrain+.txt",
        "https://raw.githubusercontent.com/datarminto/NSL-KDD-Dataset/master/KDDTrain+.txt",
    ],
    "KDDTest+.txt": [
        "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest+.txt",
        "https://raw.githubusercontent.com/arindam036/NSL-KDD-Dataset/master/KDDTest+.txt",
        "https://raw.githubusercontent.com/datarminto/NSL-KDD-Dataset/master/KDDTest+.txt",
    ],
}


# --------------------------------------------------------------------------- #
# Download helpers
# --------------------------------------------------------------------------- #
def download_file(url: str, dest_path: str, chunk_size: int = 8192) -> None:
    """Stream a single file from ``url`` to ``dest_path``."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=90) as resp, open(dest_path, "wb") as fh:
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            fh.write(chunk)


def download_dataset(datasets_dir: str | Path = "datasets",
                     force: bool = False) -> dict[str, Path]:
    """Download ``KDDTrain+.txt`` and ``KDDTest+.txt`` into ``datasets/``.

    Tries the canonical URL first and falls back to mirrors on failure.
    """
    datasets_dir = Path(datasets_dir)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for fname, urls in DATASET_URLS.items():
        dest = datasets_dir / fname
        if dest.exists() and not force:
            print(f"[ok] {fname} already present -> {dest}")
            paths[fname] = dest
            continue

        last_err: Exception | None = None
        for url in urls:
            try:
                print(f"[..] downloading {fname} from {url}")
                download_file(url, str(dest))
                print(f"[ok] saved -> {dest}")
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001 - try the next mirror
                last_err = exc
                print(f"[!] mirror failed: {exc}", file=sys.stderr)
        if last_err is not None:
            raise ConnectionError(f"Could not download {fname}: {last_err}")
        paths[fname] = dest

    return paths


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def _read_nsl_kdd(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, header=None, names=COLUMN_NAMES)


def load_train(datasets_dir: str | Path = "datasets",
               download_if_missing: bool = True) -> pd.DataFrame:
    """Load KDDTrain+ as a 43-column dataframe."""
    path = Path(datasets_dir) / "KDDTrain+.txt"
    if not path.exists() and download_if_missing:
        download_dataset(datasets_dir)
    df = _read_nsl_kdd(path)
    print(f"[load] KDDTrain+ -> {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def load_test(datasets_dir: str | Path = "datasets",
              download_if_missing: bool = True) -> pd.DataFrame:
    """Load KDDTest+ as a 43-column dataframe."""
    path = Path(datasets_dir) / "KDDTest+.txt"
    if not path.exists() and download_if_missing:
        download_dataset(datasets_dir)
    df = _read_nsl_kdd(path)
    print(f"[load] KDDTest+  -> {df.shape[0]} rows x {df.shape[1]} cols")
    return df


def load_all(datasets_dir: str | Path = "datasets",
             merge: bool = True,
             download_if_missing: bool = True) -> pd.DataFrame | tuple:
    """Load both splits. If ``merge`` is True, return a single dataframe with
    an extra ``split`` column, otherwise return ``(train, test)``."""
    train = load_train(datasets_dir, download_if_missing)
    test = load_test(datasets_dir, download_if_missing)
    if merge:
        tr, te = train.copy(), test.copy()
        tr["split"] = "train"
        te["split"] = "test"
        df = pd.concat([tr, te], ignore_index=True)
        print(f"[load] merged -> {df.shape[0]} rows x {df.shape[1]} cols")
        return df
    return train, test


# --------------------------------------------------------------------------- #
# EDA helpers
# --------------------------------------------------------------------------- #
def inspect_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Print quick EDA facts and return ``describe(include='all').T``."""
    try:
        from IPython.display import display
    except Exception:  # noqa: BLE001 - IPython not available (plain script)
        display = lambda x: print(x)  # noqa: E731

    print("=" * 60)
    print(f"shape            : {df.shape}")
    print(f"dtype counts     :\n{df.dtypes.value_counts().to_string()}")
    print(f"missing values   : {int(df.isna().sum().sum())}")
    print(f"duplicate rows   : {int(df.duplicated().sum())}")
    print(f"label counts     :\n{df['label'].value_counts().to_string()}")
    print("=" * 60)
    display(df.head())
    return df.describe(include="all").T


# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #
def save_raw(df: pd.DataFrame, out_path: str | Path = "data/kdd_raw.csv") -> Path:
    """Save the raw dataframe. Drops the temporary ``split`` column if present
    so the CSV keeps the exact 43-column NSL-KDD schema."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = df.drop(columns=["split"], errors="ignore")
    df.to_csv(out_path, index=False)
    print(f"[save] -> {out_path} ({df.shape[0]} rows, {df.shape[1]} cols)")
    return out_path

