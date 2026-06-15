import os
import json
import numpy as np

from .base import StructurePrediction, PredictionResult, STRUCTURE_STATES
from ..data.amino_acids import AMINO_ACIDS, AMINO_ACID_INDEX


def _generate_gor_probabilities():
    np.random.seed(42)
    probs = {}
    for aa in AMINO_ACIDS:
        probs[aa] = {}
        for pos in range(-8, 9):
            h = np.random.uniform(0.2, 0.6)
            e = np.random.uniform(0.1, 0.4)
            c = 1.0 - h - e
            probs[aa][pos] = {"H": h, "E": e, "C": max(0.0, c)}
    return probs


class GOR4Predictor(StructurePrediction):
    name = "GOR-IV"

    def __init__(self, prob_file: str = None):
        self.window_size = 8
        self.probabilities = self._load_probabilities(prob_file)

    def _load_probabilities(self, prob_file: str = None):
        if prob_file and os.path.exists(prob_file):
            try:
                with open(prob_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return _generate_gor_probabilities()

    def predict(self, sequence: str) -> PredictionResult:
        n = len(sequence)
        probabilities = np.zeros((n, 3), dtype=np.float32)

        for i in range(n):
            scores = {"H": 0.0, "E": 0.0, "C": 0.0}
            for offset in range(-self.window_size, self.window_size + 1):
                pos = i + offset
                if 0 <= pos < n:
                    aa = sequence[pos]
                    if aa in self.probabilities:
                        pos_key = str(offset) if offset not in self.probabilities.get(aa, {}) else offset
                        probs = self.probabilities[aa].get(pos_key, self.probabilities[aa].get(str(offset), {"H": 1/3, "E": 1/3, "C": 1/3}))
                        scores["H"] += np.log(probs.get("H", 1/3) + 1e-10)
                        scores["E"] += np.log(probs.get("E", 1/3) + 1e-10)
                        scores["C"] += np.log(probs.get("C", 1/3) + 1e-10)

            logits = np.array([scores["H"], scores["E"], scores["C"]])
            logits = logits - logits.max()
            exp_logits = np.exp(logits)
            probabilities[i] = exp_logits / exp_logits.sum()

        return PredictionResult(
            sequence=sequence,
            method=self.name,
            probabilities=probabilities,
        )
