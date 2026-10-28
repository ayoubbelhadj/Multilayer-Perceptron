import json
import numpy as np
from network import Network
from preprocessing import Scaler


def save_model(path, net, scaler):
    # one JSON file with the topology, the weights and the scaler
    data = {
        "topology": net.topology(),
        "layers": [{"W": l.W.tolist(), "b": l.b.tolist()} for l in net.layers],
        "scaler": {"mean": scaler.mean.tolist(), "std": scaler.std.tolist()},
    }
    with open(path, "w") as f:
        json.dump(data, f)


def load_model(path):
    # returns (net, scaler) as they were saved
    with open(path) as f:
        data = json.load(f)
    try:
        net = Network(**data["topology"])
        saved_layers = data["layers"]
        mean, std = data["scaler"]["mean"], data["scaler"]["std"]
    except (KeyError, TypeError) as e:
        raise ValueError(f"not a valid model file: missing or malformed {e}")
    if len(saved_layers) != len(net.layers):
        raise ValueError(
            f"not a valid model file: {len(saved_layers)} layers saved, topology needs {len(net.layers)}"
        )
    for i, (layer, saved) in enumerate(zip(net.layers, saved_layers)):
        try:
            W, b = np.array(saved["W"], dtype=float), np.array(saved["b"], dtype=float)
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"not a valid model file: layer {i} malformed ({e})")
        if W.shape != layer.W.shape or b.shape != layer.b.shape:
            raise ValueError(
                f"not a valid model file: layer {i} has W{W.shape}/b{b.shape}, "
                f"expected W{layer.W.shape}/b{layer.b.shape}"
            )
        layer.W, layer.b = W, b
    scaler = Scaler()
    try:
        scaler.mean, scaler.std = np.array(mean, dtype=float), np.array(std, dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError(f"not a valid model file: scaler malformed ({e})")
    n_in = net.sizes[0]
    if scaler.mean.shape != (n_in,) or scaler.std.shape != (n_in,):
        raise ValueError(f"not a valid model file: scaler must have shape ({n_in},)")
    return net, scaler
