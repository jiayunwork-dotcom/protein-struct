from typing import List, Dict, Tuple
import numpy as np

from ..prediction.base import PredictionResult, STRUCTURE_STATES


def compute_q3(predicted: str, actual: str) -> float:
    if len(predicted) != len(actual) or len(predicted) == 0:
        return 0.0
    correct = sum(1 for p, a in zip(predicted, actual) if p == a)
    return correct / len(predicted)


def compute_per_state_metrics(
    predicted: str,
    actual: str,
) -> Dict[str, Dict[str, float]]:
    results = {}
    for state in STRUCTURE_STATES:
        tp = sum(1 for p, a in zip(predicted, actual) if p == state and a == state)
        fp = sum(1 for p, a in zip(predicted, actual) if p == state and a != state)
        fn = sum(1 for p, a in zip(predicted, actual) if p != state and a == state)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results[state] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }
    return results


def _find_segments(sequence: str, state: str) -> List[Tuple[int, int]]:
    segments = []
    start = None
    for i, s in enumerate(sequence):
        if s == state and start is None:
            start = i
        elif s != state and start is not None:
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, len(sequence) - 1))
    return segments


def compute_sov(predicted: str, actual: str) -> float:
    if len(predicted) != len(actual) or len(predicted) == 0:
        return 0.0

    total_score = 0.0
    total_norm = 0.0

    for state in STRUCTURE_STATES:
        actual_segs = _find_segments(actual, state)
        predicted_segs = _find_segments(predicted, state)

        for (s1, e1) in actual_segs:
            max_overlap = 0
            len_seg = e1 - s1 + 1
            for (s2, e2) in predicted_segs:
                overlap_start = max(s1, s2)
                overlap_end = min(e1, e2)
                overlap = max(0, overlap_end - overlap_start + 1)
                if overlap > max_overlap:
                    max_overlap = overlap

            delta = min(len_seg, max_overlap, (e1 - s1 + 1) // 2)
            sov_score = (max_overlap + delta) / len_seg if len_seg > 0 else 0.0
            total_score += sov_score * len_seg
            total_norm += len_seg

    return total_score / total_norm if total_norm > 0 else 0.0


def evaluate_predictions(
    predictions: List[PredictionResult],
    actual_states_list: List[str],
) -> Dict[str, Dict]:
    if len(predictions) != len(actual_states_list):
        return {}

    results = {}
    method = predictions[0].method if predictions else "unknown"

    q3_scores = []
    sov_scores = []
    all_precision = {s: [] for s in STRUCTURE_STATES}
    all_recall = {s: [] for s in STRUCTURE_STATES}

    for pred, actual in zip(predictions, actual_states_list):
        q3 = compute_q3(pred.states, actual)
        sov = compute_sov(pred.states, actual)
        per_state = compute_per_state_metrics(pred.states, actual)

        q3_scores.append(q3)
        sov_scores.append(sov)
        for s in STRUCTURE_STATES:
            all_precision[s].append(per_state[s]["precision"])
            all_recall[s].append(per_state[s]["recall"])

    results = {
        "method": method,
        "q3_mean": float(np.mean(q3_scores)),
        "q3_std": float(np.std(q3_scores)),
        "sov_mean": float(np.mean(sov_scores)),
        "sov_std": float(np.std(sov_scores)),
        "per_state": {},
    }

    for s in STRUCTURE_STATES:
        results["per_state"][s] = {
            "precision_mean": float(np.mean(all_precision[s])),
            "precision_std": float(np.std(all_precision[s])),
            "recall_mean": float(np.mean(all_recall[s])),
            "recall_std": float(np.std(all_recall[s])),
        }

    return results
