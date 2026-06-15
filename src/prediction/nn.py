import os
import pickle
import numpy as np

from .base import StructurePrediction, PredictionResult
from ..data.amino_acids import encode_sequence_one_hot, AMINO_ACIDS


def _generate_nn_weights():
    np.random.seed(42)
    input_dim = 21 * 21
    hidden1 = 128
    hidden2 = 64
    output_dim = 3

    return {
        "W1": np.random.randn(input_dim, hidden1).astype(np.float32) * 0.1,
        "b1": np.zeros(hidden1, dtype=np.float32),
        "W2": np.random.randn(hidden1, hidden2).astype(np.float32) * 0.1,
        "b2": np.zeros(hidden2, dtype=np.float32),
        "W3": np.random.randn(hidden2, output_dim).astype(np.float32) * 0.1,
        "b3": np.zeros(output_dim, dtype=np.float32),
    }


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


class NNPredictor(StructurePrediction):
    name = "Neural Network"

    def __init__(self, weights_file: str = None):
        self.window_size = 10
        self.weights = self._load_weights(weights_file)

    def _load_weights(self, weights_file: str = None):
        if weights_file and os.path.exists(weights_file):
            try:
                with open(weights_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return _generate_nn_weights()

    def _extract_window(self, encoded: np.ndarray, pos: int) -> np.ndarray:
        window_size = self.window_size
        total_window = 2 * window_size + 1
        feat_dim = encoded.shape[1]
        window = np.zeros(total_window * feat_dim, dtype=np.float32)

        for w in range(-window_size, window_size + 1):
            idx = pos + w
            if 0 <= idx < len(encoded):
                start = (w + window_size) * feat_dim
                window[start:start + feat_dim] = encoded[idx]

        return window

    def predict(self, sequence: str) -> PredictionResult:
        n = len(sequence)
        encoded = encode_sequence_one_hot(sequence)
        probabilities = np.zeros((n, 3), dtype=np.float32)

        for i in range(n):
            x = self._extract_window(encoded, i)
            h1 = _relu(x @ self.weights["W1"] + self.weights["b1"])
            h2 = _relu(h1 @ self.weights["W2"] + self.weights["b2"])
            logits = h2 @ self.weights["W3"] + self.weights["b3"]
            probabilities[i] = _softmax(logits.reshape(1, -1))[0]

        return PredictionResult(
            sequence=sequence,
            method=self.name,
            probabilities=probabilities,
        )
