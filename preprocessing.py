import numpy as np
import csv

N_FEATURES = 30
CLASSES = ["B", "M"]


def one_hot(labels, classes=CLASSES):
    # ["M", "B"] -> [[0, 1], [1, 0]]
    labels = np.asarray(labels)
    unknown = sorted({str(l) for l in labels if l not in classes})
    if unknown:
        valid = ", ".join(classes)
        raise ValueError(f"Unknown labels {unknown}. Valid classes: {valid}")
    index = {c: i for i, c in enumerate(classes)}
    encoded = np.zeros((len(labels), len(classes)))
    for i, l in enumerate(labels):
        encoded[i, index[l]] = 1
    return encoded


def load_dataset(path):
    # data.csv format: no header, one row per patient: id, diagnosis, 30 features
    # returns the features (N, 30) and the labels (N,)
    features = []
    labels = []
    with open(path, newline="") as f:
        for line_no, row in enumerate(csv.reader(f), start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            if len(row) != N_FEATURES + 2:
                raise ValueError(
                    f"Line {line_no}: expected {N_FEATURES + 2} columns, got {len(row)}"
                )
            try:
                values = [float(v) for v in row[2:]]
            except ValueError:
                raise ValueError(f"Line {line_no}: a feature is not a number")
            labels.append(row[1].strip())
            features.append(values)
    if not features:
        raise ValueError(f"No data rows found in {path}")
    return np.array(features, dtype=float), np.array(labels)


class Scaler:
    # standardization: (x - mean) / std, per feature
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean = X.mean(axis=0)
        std = X.std(axis=0)
        self.std = np.where(std == 0, 1.0, std)
        return self

    def transform(self, X):
        if self.mean is None or self.std is None:
            raise RuntimeError("Scaler is not fitted yet: call fit() first")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.mean.shape[0]:
            raise ValueError(
                f"Expected {self.mean.shape[0]} features, got shape {X.shape}"
            )
        return (X - self.mean) / self.std


def split_indices(n, valid_ratio=0.2, seed=None):
    # returns shuffled (train_idx, valid_idx)
    if not 0 < valid_ratio < 1:
        raise ValueError(f"valid_ratio must be strictly between 0 and 1, got {valid_ratio}")
    n_valid = int(round(n * valid_ratio))
    if n_valid == 0 or n_valid == n:
        raise ValueError(f"n={n} with valid_ratio={valid_ratio} leaves one side empty")
    order = np.random.default_rng(seed).permutation(n)
    return order[n_valid:], order[:n_valid]
