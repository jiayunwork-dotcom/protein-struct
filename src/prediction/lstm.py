import os
import pickle
import numpy as np

from .base import StructurePrediction, PredictionResult
from ..data.amino_acids import encode_sequence_with_properties


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / exp_x.sum(axis=-1, keepdims=True)


def _generate_lstm_weights():
    np.random.seed(42)
    input_dim = 24
    hidden_dim = 64
    output_dim = 3

    def _lstm_weights(in_dim, hid):
        return {
            "W_ih": np.random.randn(in_dim, 4 * hid).astype(np.float32) * 0.1,
            "W_hh": np.random.randn(hid, 4 * hid).astype(np.float32) * 0.1,
            "b_ih": np.zeros(4 * hid, dtype=np.float32),
            "b_hh": np.zeros(4 * hid, dtype=np.float32),
        }

    return {
        "lstm_fwd1": _lstm_weights(input_dim, hidden_dim),
        "lstm_bwd1": _lstm_weights(input_dim, hidden_dim),
        "lstm_fwd2": _lstm_weights(hidden_dim, hidden_dim),
        "lstm_bwd2": _lstm_weights(hidden_dim, hidden_dim),
        "W_out": np.random.randn(2 * hidden_dim, output_dim).astype(np.float32) * 0.1,
        "b_out": np.zeros(output_dim, dtype=np.float32),
    }


class _LSTMCell:
    def __init__(self, weights):
        self.W_ih = weights["W_ih"]
        self.W_hh = weights["W_hh"]
        self.b_ih = weights["b_ih"]
        self.b_hh = weights["b_hh"]
        self.hidden_dim = self.W_hh.shape[0]

    def step(self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray):
        gates = x @ self.W_ih + self.b_ih + h_prev @ self.W_hh + self.b_hh
        i = _sigmoid(gates[:, :self.hidden_dim])
        f = _sigmoid(gates[:, self.hidden_dim:2*self.hidden_dim])
        g = _tanh(gates[:, 2*self.hidden_dim:3*self.hidden_dim])
        o = _sigmoid(gates[:, 3*self.hidden_dim:])
        c = f * c_prev + i * g
        h = o * _tanh(c)
        return h, c


def _run_lstm(encoded: np.ndarray, weights_fwd, weights_bwd) -> np.ndarray:
    n = len(encoded)
    hidden_dim = weights_fwd["W_hh"].shape[0]

    cell_fwd = _LSTMCell(weights_fwd)
    cell_bwd = _LSTMCell(weights_bwd)

    h_fwd = np.zeros((n, hidden_dim), dtype=np.float32)
    h_bwd = np.zeros((n, hidden_dim), dtype=np.float32)

    h_prev_f = np.zeros((1, hidden_dim), dtype=np.float32)
    c_prev_f = np.zeros((1, hidden_dim), dtype=np.float32)
    for i in range(n):
        h_prev_f, c_prev_f = cell_fwd.step(encoded[i:i+1], h_prev_f, c_prev_f)
        h_fwd[i] = h_prev_f[0]

    h_prev_b = np.zeros((1, hidden_dim), dtype=np.float32)
    c_prev_b = np.zeros((1, hidden_dim), dtype=np.float32)
    for i in range(n - 1, -1, -1):
        h_prev_b, c_prev_b = cell_bwd.step(encoded[i:i+1], h_prev_b, c_prev_b)
        h_bwd[i] = h_prev_b[0]

    return np.concatenate([h_fwd, h_bwd], axis=-1)


class LSTMPredictor(StructurePrediction):
    name = "Bidirectional LSTM"

    def __init__(self, weights_file: str = None):
        self.weights = self._load_weights(weights_file)

    def _load_weights(self, weights_file: str = None):
        if weights_file and os.path.exists(weights_file):
            try:
                with open(weights_file, "rb") as f:
                    return pickle.load(f)
            except Exception:
                pass
        return _generate_lstm_weights()

    def predict(self, sequence: str) -> PredictionResult:
        n = len(sequence)
        if n == 0:
            return PredictionResult(sequence="", method=self.name, states="", probabilities=np.zeros((0, 3)))

        encoded = encode_sequence_with_properties(sequence)

        h1 = _run_lstm(encoded, self.weights["lstm_fwd1"], self.weights["lstm_bwd1"])
        hidden_dim = h1.shape[-1] // 2
        h1_encoded = h1[:, :hidden_dim] + h1[:, hidden_dim:]

        h2 = _run_lstm(h1_encoded, self.weights["lstm_fwd2"], self.weights["lstm_bwd2"])

        logits = h2 @ self.weights["W_out"] + self.weights["b_out"]
        probabilities = _softmax(logits)

        return PredictionResult(
            sequence=sequence,
            method=self.name,
            probabilities=probabilities,
        )
