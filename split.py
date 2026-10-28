import argparse
import csv
import sys
from preprocessing import split_indices


def main():
    parser = argparse.ArgumentParser(description="Split a dataset into a training file and a test file.")
    parser.add_argument("dataset", nargs="?", default="data.csv", help="CSV file to split (no header), default: data.csv")
    parser.add_argument("--test_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=None, help="random seed, for a repeatable split")
    parser.add_argument("--train_out", default="data_training.csv")
    parser.add_argument("--test_out", default="data_test.csv")
    args = parser.parse_args()

    try:
        with open(args.dataset, newline="") as f:
            rows = [r for r in csv.reader(f) if any(c.strip() for c in r)]
        train_idx, test_idx = split_indices(len(rows), args.test_ratio, args.seed)
        for path, idx in ((args.train_out, train_idx), (args.test_out, test_idx)):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerows(rows[i] for i in idx)
    except (OSError, ValueError) as e:
        sys.exit(f"error: {e}")

    print(f"{len(train_idx)} rows -> {args.train_out}")
    print(f"{len(test_idx)} rows -> {args.test_out}")


if __name__ == "__main__":
    main()