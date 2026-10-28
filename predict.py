import argparse
import sys
from preprocessing import load_dataset, one_hot, CLASSES
from model_io import load_model
from losses import binary_cross_entropy, accuracy
from metrics import confusion, precision_recall_f1

M = CLASSES.index("M")  # malignant column of the output


def evaluate(net, scaler, X, labels):
    # returns (binary cross-entropy, accuracy)
    Y = one_hot(labels)
    P = net.forward(scaler.transform(X))
    bce = binary_cross_entropy(Y[:, M], P[:, M])
    acc = accuracy(Y, P)
    return bce, acc


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved model on a labelled dataset.")
    parser.add_argument("--dataset", default="data_test.csv")
    parser.add_argument("--model", default="saved_model.json")
    parser.add_argument("--metrics", action="store_true",
                        help="also print the confusion matrix and precision / recall / F1")
    args = parser.parse_args()
    try:
        print(f"> loading model '{args.model}'...")
        net, scaler = load_model(args.model)
        print(f"> network: {net.sizes}, {net.hidden_activation} / {net.output_activation}")
        X, labels = load_dataset(args.dataset)
        print(f"> predicting {len(X)} patients from '{args.dataset}'...")
        bce, acc = evaluate(net, scaler, X, labels)
        if args.metrics:
            Y, P = one_hot(labels), net.forward(scaler.transform(X))
            tp, fp, fn, tn = confusion(Y, P)
            precision, recall, f1 = precision_recall_f1(Y, P)
    except (OSError, ValueError) as e:
        sys.exit(f"error: {e}")
    correct = int(round(acc * len(X)))
    print(f"accuracy             : {acc:.4f} ({correct}/{len(X)})")
    print(f"binary cross-entropy : {bce:.4f}")
    if args.metrics:
        print("confusion matrix (positive = M):")
        print("              pred M   pred B")
        print(f"  truly M   {tp:8d} {fn:8d}")
        print(f"  truly B   {fp:8d} {tn:8d}")
        print(f"precision : {precision:.4f} - recall : {recall:.4f} - f1 : {f1:.4f}")

if __name__ == "__main__":
    main()