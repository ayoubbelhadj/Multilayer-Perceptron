import argparse
import json
import os
import sys
import numpy as np
from preprocessing import load_dataset, one_hot, Scaler, split_indices, N_FEATURES, CLASSES
from network import Network
from model_io import save_model
import matplotlib
import matplotlib.pyplot as plt

def save_history(folder, args, history):
    os.makedirs(folder, exist_ok=True)
    n = 1
    while os.path.exists(os.path.join(folder, f"run_{n:03d}.json")):
        n += 1
    path = os.path.join(folder, f"run_{n:03d}.json")
    with open(path, "w") as f:
        json.dump({"config": vars(args), "history": history}, f, indent=2)
    return path


def plot_history(history, path, show, metrics=False):
    
    if not show:
        matplotlib.use("Agg")  # no window
    
    epochs = range(1, len(history["loss"]) + 1)
    n_plots = 3 if metrics else 2
    fig, axes = plt.subplots(1, n_plots, figsize=(5.3 * n_plots, 4))
    ax1, ax2 = axes[0], axes[1]
    ax1.plot(epochs, history["loss"], label="training loss")
    ax1.plot(epochs, history["val_loss"], "--", label="validation loss")
    ax1.set_title("Loss")
    ax2.plot(epochs, history["acc"], label="training acc")
    ax2.plot(epochs, history["val_acc"], "--", label="validation acc")
    ax2.set_title("Accuracy")
    if metrics:
        ax3 = axes[2]
        ax3.plot(epochs, history["val_precision"], label="val precision")
        ax3.plot(epochs, history["val_recall"], label="val recall")
        ax3.plot(epochs, history["val_f1"], label="val F1")
        ax3.set_title("Validation metrics (positive = M)")
    for ax in axes:
        if "best_epoch" in history:
            ax.axvline(history["best_epoch"], color="gray", linestyle=":",
                       label=f"restored epoch {history['best_epoch']}")
        ax.set_xlabel("epoch")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    if show:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Train the multilayer perceptron.")
    parser.add_argument("--dataset", default="data_training.csv")
    parser.add_argument("--layer", type=int, nargs="+", default=[24, 24, 24], help="hidden layer sizes")
    parser.add_argument("--epochs", type=int, default=84)
    parser.add_argument("--patience", type=int, default=0,
                        help="early stopping: epochs without val_loss improvement before stopping (0 = off)")
    parser.add_argument("--learning_rate", type=float, default=0.0314)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--activation", default="sigmoid", choices=["sigmoid", "tanh", "relu"])
    parser.add_argument("--output-activation", default="softmax", choices=["softmax"])
    parser.add_argument("--initializer", default="heUniform", choices=["heUniform", "xavierUniform"])
    parser.add_argument("--valid_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--model", default="saved_model.json")
    parser.add_argument("--plot", default="learning_curves.png")
    parser.add_argument("--runs_dir", default="runs", help="folder where every run is saved")
    parser.add_argument("--metrics", action="store_true",
                        help="also print validation precision / recall / F1 every epoch")
    parser.add_argument("--quiet", action="store_true", help="do not print the per-epoch progress")
    parser.add_argument("--no_show", action="store_true", help="save the curves without opening a window")
    args = parser.parse_args()

    try:
        if args.seed is not None:
            np.random.seed(args.seed)
        X, labels = load_dataset(args.dataset)
        Y = one_hot(labels)
        train_idx, valid_idx = split_indices(len(X), args.valid_ratio, args.seed)
        scaler = Scaler().fit(X[train_idx])
        x_train, x_valid = scaler.transform(X[train_idx]), scaler.transform(X[valid_idx])
        y_train, y_valid = Y[train_idx], Y[valid_idx]
        print(f"x_train shape : {x_train.shape}")
        print(f"x_valid shape : {x_valid.shape}")
        net = Network([N_FEATURES] + args.layer + [len(CLASSES)], args.activation, args.output_activation,
                      args.initializer)
        history = net.fit(x_train, y_train, x_valid, y_valid,
                          epochs=args.epochs, batch_size=args.batch_size,
                          learning_rate=args.learning_rate, patience=args.patience,
                          verbose=not args.quiet, print_metrics=args.metrics)
        print(f"> saving model '{args.model}' to disk...")
        save_model(args.model, net, scaler)
        path = save_history(args.runs_dir, args, history)
        print(f"> saving history '{path}'")
        plot_history(history, args.plot, show=not args.no_show, metrics=args.metrics)
    except (OSError, ValueError) as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()