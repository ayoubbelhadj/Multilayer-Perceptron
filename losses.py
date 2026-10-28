import numpy as np

EPSILON = 1e-15  # keeps p away from 0 and 1 so log() stays finite


def binary_cross_entropy(y, p):
    # E = -1/N * sum(y*log(p) + (1-y)*log(1-p))
    # y is 0/1 (1 = malignant), p is the predicted probability of malignant
    y = np.asarray(y)
    p = np.asarray(p)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y {y.shape} vs p {p.shape}")
    p = np.clip(p, EPSILON, 1 - EPSILON)
    L = -np.mean((y * np.log(p)) + ((1 - y) * np.log(1 - p)))
    return float(L)


def accuracy(Y, P):
    # fraction of patients whose most likely class is the true one
    Y = np.asarray(Y)
    P = np.asarray(P)

    if Y.shape != P.shape:
        raise ValueError(f"shape mismatch: Y {Y.shape} vs P {P.shape}")
    return float(np.mean(np.argmax(P, axis=1) == np.argmax(Y, axis=1)))