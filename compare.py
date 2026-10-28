import argparse
import glob
import json
import os
import sys


def label(config):
    # short description of a run, e.g. "24 24 24, relu, lr 0.0314"
    layers = " ".join(str(n) for n in config["layer"])
    return f"{layers}, {config['activation']}, lr {config['learning_rate']}"


def load(paths):
    # returns (name, config, history) for every file we can read
    runs = []
    for path in paths:
        name = os.path.basename(path)
        try:
            with open(path) as f:
                data = json.load(f)
            config, history = data["config"], data["history"]
            label(config)  # raises KeyError if the config is incomplete
            if "val_loss" not in history:
                raise KeyError("val_loss")
            runs.append((name, config, history))
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"warning: skipping '{name}': {e}", file=sys.stderr)
    return runs


def plot(runs, out, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, config, history in runs:
        val = history["val_loss"]
        epochs = range(1, len(val) + 1)
        line, = ax.plot(epochs, val, label=name + ": " + label(config))
        best = val.index(min(val))
        ax.plot(best + 1, val[best], "o", color=line.get_color())  # dot on the best epoch
    ax.set_title("Validation loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("val_loss")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    if show:
        plt.show()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Draw several runs' validation loss on one graph.")
    parser.add_argument("runs", nargs="*", help="run files to compare (default: every file in --runs_dir)")
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--out", default="compare.png")
    parser.add_argument("--no_show", action="store_true")
    args = parser.parse_args()
    paths = args.runs or sorted(glob.glob(os.path.join(args.runs_dir, "*.json")))
    runs = load(paths)
    if not runs:
        sys.exit("error: no runs to compare")
    plot(runs, args.out, show=not args.no_show)
    print(f"> {len(runs)} runs drawn in '{args.out}'")


if __name__ == "__main__":
    main()
