from preprocessing import CLASSES

POS = CLASSES.index("M")  # positive class = malignant


def confusion(Y, P):
    # Y is one-hot, P are probabilities; returns (tp, fp, fn, tn)
    if Y.shape != P.shape:
        raise ValueError(f"shape mismatch: Y {Y.shape} vs P {P.shape}")
    tp = fp = fn = tn = 0
    for truth, pred in zip(Y.argmax(axis=1), P.argmax(axis=1)):
        if truth == POS and pred == POS:
            tp += 1
        elif truth != POS and pred == POS:
            fp += 1
        elif truth == POS and pred != POS:
            fn += 1
        else:
            tn += 1
    return tp, fp, fn, tn


def precision_recall_f1(Y, P):
    # a 0/0 gives 0.0
    tp, fp, fn, _ = confusion(Y, P)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1
