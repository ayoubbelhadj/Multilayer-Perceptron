import numpy as np

class Sigmoid:
    @staticmethod
    def forward(z):
        z = np.clip(z, -500, 500)
        e = np.exp(-z)
        return 1 / (1 + e)

    @staticmethod
    def derivative(z):
        s = Sigmoid.forward(z)
        return s * (1 - s)


class Tanh:
    @staticmethod
    def forward(z):
        return np.tanh(z)

    @staticmethod
    def derivative(z):
        return 1 - np.tanh(z) ** 2


class ReLU:
    @staticmethod
    def forward(z):
        return np.maximum(0, z)

    @staticmethod
    def derivative(z):
        return (z > 0).astype(float)

class Softmax:
    @staticmethod
    def forward(z):
        # one softmax per row (one row per patient)
        e = np.exp(z - z.max(axis=1, keepdims=True))  # subtract the max to avoid overflow
        return e / e.sum(axis=1, keepdims=True)

    @staticmethod
    def derivative(z):
        raise NotImplementedError(
            "Softmax's derivative is not element-wise (each output depends on "
            "every input), so it is handled together with cross-entropy in backprop."
        )


ACTIVATIONS = {
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "relu": ReLU,
    "softmax": Softmax
}


def get_activation(name):
    if name not in ACTIVATIONS:
        valid = ", ".join(ACTIVATIONS)
        raise ValueError(f"Unknown activation '{name}'. Valid names: {valid}")
    return ACTIVATIONS[name]