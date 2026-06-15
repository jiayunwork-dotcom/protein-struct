from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List
import numpy as np


STRUCTURE_STATES = ["H", "E", "C"]
STRUCTURE_NAMES = {"H": "α-Helix", "E": "β-Sheet", "C": "Coil"}
STRUCTURE_COLORS = {"H": "#E74C3C", "E": "#3498DB", "C": "#95A5A6"}


@dataclass
class PredictionResult:
    sequence: str
    method: str
    states: str = ""
    probabilities: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))

    def __post_init__(self):
        if not self.states and self.probabilities.size > 0:
            self.states = "".join(
                STRUCTURE_STATES[np.argmax(p)] for p in self.probabilities
            )
        if self.probabilities.size == 0 and self.states:
            self.probabilities = np.zeros((len(self.states), 3))
            for i, s in enumerate(self.states):
                if s in STRUCTURE_STATES:
                    self.probabilities[i, STRUCTURE_STATES.index(s)] = 1.0

    def __len__(self) -> int:
        return len(self.states)


class StructurePrediction(ABC):
    name: str = "base"

    @abstractmethod
    def predict(self, sequence: str) -> PredictionResult:
        pass

    def predict_batch(self, sequences: List[str]) -> List[PredictionResult]:
        return [self.predict(seq) for seq in sequences]
