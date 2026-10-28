import argparse
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="List every saved training run, best first.")
    parser.add_argument("--runs_dir", default="runs")
    parser.add_argument("--sort", default="best", choices=["best", "run"],
                        help="best = lowest val_loss first, run = run_001, run_002, ...")
    args = parser.parse_args()

    runs = []
    names = []
    if os.path.isdir(args.runs_dir):
        names = sorted(f for f in os.listdir(args.runs_dir) if f.endswith(".json"))
    for name in names:
        try:
            with open(os.path.join(args.runs_dir, name)) as f:
                data = json.load(f)
            config, val_loss = data["config"], data["history"]["val_loss"]
            best = min(val_loss)
            i = val_loss.index(best)
            runs.append((best, [
                name,
                " ".join(str(n) for n in config["layer"]),
                config["activation"],
                str(config["learning_rate"]),
                str(config["batch_size"]),
                f"{len(val_loss)}/{config['epochs']}",
                f"{best:.4f} @ {i + 1}",
                f"{data['history']['val_f1'][i]:.4f}",  # val_f1 at the best epoch
            ]))
        except (OSError, ValueError, KeyError, TypeError) as e:
            print(f"warning: skipping '{name}': {e}", file=sys.stderr)

    if not runs:
        print(f"no runs in '{args.runs_dir}'")
        return
    rows = [["run", "layers", "act", "lr", "batch", "epochs", "best val_loss", "val_f1"]]
    if args.sort == "best":
        runs.sort(key=lambda r: r[0])  # with --sort run we keep the file order
    rows += [row for _, row in runs]
    w = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    left = (0, 1, 2)  # text columns, the rest are right-aligned
    for row in rows:
        print("   ".join(f"{c:<{w[i]}}" if i in left else f"{c:>{w[i]}}" for i, c in enumerate(row)))


if __name__ == "__main__":
    main()
