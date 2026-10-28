import numpy as np
from activations import get_activation


def he_uniform(n_in, n_out):
    limit = np.sqrt(6 / n_in)
    return np.random.uniform(-limit, limit, size=(n_out, n_in))


def xavier_uniform(n_in, n_out):
    limit = np.sqrt(6 / (n_in + n_out))
    return np.random.uniform(-limit, limit, size=(n_out, n_in))


INITIALIZERS = {
    "heUniform": he_uniform,
    "xavierUniform": xavier_uniform
}


def get_initializer(name):
    if name not in INITIALIZERS:
        valid = ", ".join(INITIALIZERS)
        raise ValueError(f"Unknown initializer '{name}'. Valid names: {valid}")
    return INITIALIZERS[name]


class DenseLayer:
    def __init__(self, n_in, n_out, activation="sigmoid", weights_initializer="heUniform"):
        self.n_in = n_in
        self.n_out = n_out
        self.activation = get_activation(activation)
        self.W = get_initializer(weights_initializer)(n_in, n_out)
        self.b = np.zeros(n_out)
        self.A_prev = None
        self.A = None
        self.Z = None
        self.dW = None  # gradients, set by backward
        self.db = None


    def forward(self, A_prev):
        if A_prev.ndim != 2 or A_prev.shape[1] != self.n_in:
            hint = " (for one patient, use x.reshape(1, -1))" if A_prev.ndim == 1 else ""
            raise ValueError(
                f"Expected input of shape (batch, {self.n_in}), got {A_prev.shape}{hint}"
            )
        self.A_prev = A_prev
        self.Z = A_prev @ self.W.T + self.b
        self.A = self.activation.forward(self.Z)
        return self.A

    def backward(self, dA):
        if self.Z is None:
            raise RuntimeError("forward() must be called before backward()")
        dZ = dA * self.activation.derivative(self.Z)
        return self.backward_from_dZ(dZ)

    def backward_from_dZ(self, dZ):
        if self.A_prev is None:
            raise RuntimeError("forward() must be called before backward_from_dZ()")
        self.dW = dZ.T @ self.A_prev
        self.db = dZ.sum(axis=0)
        return dZ @ self.W

    def update(self, lr):
        if self.dW is None:
            raise RuntimeError("backward() must be called before update()")
        self.W -= lr * self.dW
        self.b -= lr * self.db
