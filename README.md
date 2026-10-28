# Multilayer Perceptron

A feed-forward neural network written from scratch with NumPy. It predicts whether a breast tumor is **malignant (M)** or **benign (B)** from 30 numeric measurements (the Wisconsin breast cancer dataset in `data.csv`).

## What it does

- Standardizes the features (mean/std computed on the training part only).
- Trains a fully connected network with mini-batch gradient descent: configurable hidden layers, sigmoid/tanh/relu activations, softmax output, He or Xavier initialization.
- Supports early stopping, which restores the best epoch.
- Saves the model, the scaler, a plot of the learning curves and a JSON log of every run.
- Evaluates a saved model on new data (accuracy, binary cross-entropy, and optionally confusion matrix, precision, recall and F1).

## Requirements

Python 3 with `numpy` and `matplotlib`:

```
pip install numpy matplotlib
```

## Dataset format

CSV with no header, one row per patient: `id, diagnosis (M or B), 30 features`.

## Usage

### 1. Split the data

```
python split.py data.csv --test_ratio 0.2 --seed 42
```

Writes `data_training.csv` and `data_test.csv`. Options: `--train_out`, `--test_out`.

### 2. Train

```
python train.py
```

Defaults: 3 hidden layers of 24 neurons, sigmoid, 84 epochs, learning rate 0.0314, batch size 8. Example with options:

```
python train.py --layer 32 16 --activation relu --epochs 200 --learning_rate 0.01 \
                --patience 15 --metrics --seed 1 --no_show
```

| Option | Meaning |
|---|---|
| `--dataset` | training CSV (default `data_training.csv`) |
| `--layer` | hidden layer sizes, e.g. `24 24 24` |
| `--epochs`, `--batch_size`, `--learning_rate` | training hyperparameters |
| `--activation` | `sigmoid`, `tanh` or `relu` |
| `--initializer` | `heUniform` or `xavierUniform` |
| `--valid_ratio` | share of the training file used for validation (default 0.2) |
| `--patience` | early stopping patience on validation loss (0 = off) |
| `--seed` | reproducible run |
| `--metrics` | also report validation precision / recall / F1 |
| `--quiet` | hide per-epoch output |
| `--no_show` | save the plot without opening a window |
| `--model`, `--plot`, `--runs_dir` | output locations (`saved_model.json`, `learning_curves.png`, `runs/`) |

Each run is saved as `runs/run_NNN.json` (config + history).

### 3. Predict / evaluate

```
python predict.py --dataset data_test.csv --model saved_model.json --metrics
```

Prints accuracy and binary cross-entropy; with `--metrics` also the confusion matrix, precision, recall and F1 (positive class = M).

### 4. Review past runs

```
python history.py                 # list runs, best validation loss first
python history.py --sort run      # in run order
python compare.py                 # plot validation loss of all runs to compare.png
python compare.py runs/run_001.json runs/run_002.json --no_show
```

## Project layout

| File | Role |
|---|---|
| `split.py` | split a dataset into train/test files |
| `train.py` | train the network, save model, plot and run log |
| `predict.py` | evaluate a saved model |
| `history.py`, `compare.py` | list and compare saved runs |
| `network.py`, `layers.py` | network and layer implementation (forward/backprop) |
| `activations.py`, `losses.py`, `metrics.py` | activation functions, loss, metrics |
| `preprocessing.py` | CSV loading, one-hot encoding, scaler, split |
| `model_io.py` | save/load model to JSON |
