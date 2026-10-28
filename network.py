import numpy as np
from layers import DenseLayer
from losses import binary_cross_entropy, accuracy
from metrics import precision_recall_f1


class Network:
    def __init__(self, sizes, hidden_activation="sigmoid", output_activation="softmax",
                 weights_initializer="heUniform"):
        # e.g. [30, 24, 24, 2] = 30 features, two hidden layers of 24, 2 classes
        sizes = list(sizes)
        if len(sizes) < 2:
            raise ValueError(f"sizes needs at least 2 entries, got {len(sizes)}")
        if not all(isinstance(s, (int, np.integer)) and not isinstance(s, bool) and s > 0
                   for s in sizes):
            raise ValueError(f"sizes must all be positive integers, got {sizes}")
        # with one output neuron softmax always gives 1.0, nothing to learn
        if output_activation == "softmax" and sizes[-1] < 2:
            raise ValueError(f"softmax needs at least 2 output neurons, got {sizes[-1]}")
        self.sizes = [int(s) for s in sizes]
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.weights_initializer = weights_initializer

        self.layers = []
        n_layers = len(sizes) - 1
        for n in range(1, n_layers + 1):
            activation = output_activation if n == n_layers else hidden_activation
            self.layers.append(DenseLayer(sizes[n - 1], sizes[n], activation, weights_initializer))

    def forward(self, X):
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backward(self, Y):
        if self.output_activation != "softmax":
            raise ValueError("backward() only supports softmax + cross-entropy")
        P = self.layers[-1].A
        if P is None:
            raise RuntimeError("forward() must be called before backward()")
        if Y.shape != P.shape:
            raise ValueError(f"Y shape {Y.shape} does not match output shape {P.shape}")
        dZ = (P - Y) / Y.shape[0]
        dA = self.layers[-1].backward_from_dZ(dZ)
        for layer in reversed(self.layers[:-1]):
            dA = layer.backward(dA)

    def update(self, lr):
        for layer in self.layers:
            layer.update(lr)

    @staticmethod
    def compute_loss(Y, P):
        if Y.shape[1] != 2:
            raise ValueError("binary cross-entropy needs exactly 2 output classes")
        return binary_cross_entropy(Y[:, 1], P[:, 1])  # column 1 = malignant

    def fit(self, X, Y, X_val, Y_val, epochs, batch_size, learning_rate, verbose=True, patience=0,
            print_metrics=False):
        for name, (A, B) in {"train": (X, Y), "val": (X_val, Y_val)}.items():
            if len(A) != len(B):
                raise ValueError(f"X and Y ({name}) have different sizes: {len(A)} vs {len(B)}")
        for name, v in (("epochs", epochs), ("batch_size", batch_size)):
            if not isinstance(v, (int, np.integer)) or isinstance(v, bool) or v <= 0:
                raise ValueError(f"{name} must be a positive integer, got {v}")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError(f"learning_rate must be a positive number, got {learning_rate}")

        if not isinstance(patience, (int, np.integer)) or isinstance(patience, bool) or patience < 0:
            raise ValueError(f"patience must be an integer >= 0, got {patience}")

        history = {"loss": [], "val_loss": [], "acc": [], "val_acc": [],
                   "precision": [], "recall": [], "f1": [],
                   "val_precision": [], "val_recall": [], "val_f1": []}
        best_val, best_epoch, best_weights, wait, stopped_epoch = np.inf, 0, None, 0, None
        N = len(X)
        for epoch in range(1, epochs + 1):
            order = np.random.permutation(N)
            for start in range(0, N, batch_size):
                idx = order[start: start+batch_size]
                self.forward(X[idx])
                self.backward(Y[idx])
                self.update(learning_rate)
            P = self.forward(X)
            P_val = self.forward(X_val)
            loss = self.compute_loss(Y, P)
            val_loss = self.compute_loss(Y_val, P_val)
            history["loss"].append(loss)
            history["val_loss"].append(val_loss)
            history["acc"].append(accuracy(Y, P))
            history["val_acc"].append(accuracy(Y_val, P_val))
            for prefix, (A, B) in (("", (Y, P)), ("val_", (Y_val, P_val))):
                for key, value in zip(("precision", "recall", "f1"), precision_recall_f1(A, B)):
                    history[prefix + key].append(value)
            if verbose:
                line = f"epoch {epoch:02d}/{epochs:02d} - loss: {loss:.4f} - val_loss: {val_loss:.4f}"
                if print_metrics:
                    line += (f" - val_precision: {history['val_precision'][-1]:.4f}"
                             f" - val_recall: {history['val_recall'][-1]:.4f}"
                             f" - val_f1: {history['val_f1'][-1]:.4f}")
                print(line)

            if patience:
                if val_loss < best_val:
                    best_val, best_epoch, wait = val_loss, epoch, 0
                    best_weights = [(layer.W.copy(), layer.b.copy()) for layer in self.layers]
                else:
                    wait += 1
                    if wait >= patience:
                        stopped_epoch = epoch
                        break

        if patience and best_weights is not None:  # still None if val_loss was always NaN
            for layer, (W, b) in zip(self.layers, best_weights):
                layer.W, layer.b = W, b
            history["best_epoch"] = best_epoch
            history["stopped_epoch"] = stopped_epoch
            if verbose:
                if stopped_epoch is not None:
                    print(f"early stopping at epoch {stopped_epoch}, "
                          f"restoring epoch {best_epoch} (val_loss {best_val:.4f})")
                else:
                    print(f"restoring best epoch {best_epoch} (val_loss {best_val:.4f})")
        return history

    def topology(self):
        # what is needed to rebuild the structure (not the weights)
        return {
            "sizes": list(self.sizes),
            "hidden_activation": self.hidden_activation,
            "output_activation": self.output_activation,
            "weights_initializer": self.weights_initializer,
        }
