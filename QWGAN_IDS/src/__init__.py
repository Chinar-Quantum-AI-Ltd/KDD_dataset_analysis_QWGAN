"""QWGAN_IDS source package.

FR-1 / FR-2 pipeline: NSL-KDD -> quantum-ready latent representation.

Modules
-------
loader            : download / load / EDA of NSL-KDD           -> data/kdd_raw.csv
preprocessing     : cleaning + schema validation               -> data/kdd_clean.csv
encoding          : OneHot + log1p + RobustScaler              -> data/feature_matrix.pkl
feature_selection : MI -> top-k -> PCA / autoencoder           -> data/latent_features.npy
quantum_encoding  : MinMax -> angles [0, pi] + invertible decode + PennyLane verify
"""

__version__ = "0.1.0"

