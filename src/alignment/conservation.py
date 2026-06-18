from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

import numpy as np

from ..data.amino_acids import AMINO_ACIDS, AMINO_ACID_INDEX
from ..data.blosum62 import blosum62_score, BLOSUM62


@dataclass
class ConservationResult:
    shannon_entropy: np.ndarray
    weighted_score: np.ndarray
    smoothed_entropy: np.ndarray
    column_amino_acid_counts: List[Dict[str, int]]
    column_consensus: List[str]
    total_columns: int
    valid_columns: int
    conserved_regions: List[Dict] = field(default_factory=list)
    variable_regions: List[Dict] = field(default_factory=list)


def compute_shannon_entropy(column_counts: Dict[str, int], n_valid: int) -> float:
    if n_valid == 0:
        return 1.0
    entropy = 0.0
    for count in column_counts.values():
        freq = count / n_valid
        if freq > 0:
            entropy -= freq * np.log2(freq)
    max_entropy = np.log2(20)
    normalized = entropy / max_entropy if max_entropy > 0 else 1.0
    return max(0.0, min(1.0, normalized))


def compute_weighted_conservation(column_counts: Dict[str, int], n_valid: int) -> float:
    if n_valid <= 1:
        return 1.0 if n_valid == 1 else 0.0

    aas = list(column_counts.keys())
    if len(aas) == 1:
        return 1.0

    total_score = 0.0
    total_pairs = 0
    blosum_min = -4
    blosum_max = 11

    for i, aa1 in enumerate(aas):
        count1 = column_counts[aa1]
        for j, aa2 in enumerate(aas):
            count2 = column_counts[aa2]
            if i == j:
                pairs = count1 * (count1 - 1) // 2
                if pairs > 0:
                    score = blosum62_score(aa1, aa1)
                    norm_score = (score - blosum_min) / (blosum_max - blosum_min)
                    total_score += norm_score * pairs
                    total_pairs += pairs
            elif i < j:
                pairs = count1 * count2
                if pairs > 0:
                    score = blosum62_score(aa1, aa2)
                    norm_score = (score - blosum_min) / (blosum_max - blosum_min)
                    total_score += norm_score * pairs
                    total_pairs += pairs

    if total_pairs == 0:
        return 0.0
    return max(0.0, min(1.0, total_score / total_pairs))


def sliding_window_smooth(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 1 or len(values) == 0:
        return values.copy()
    half = window_size // 2
    smoothed = np.zeros_like(values, dtype=np.float64)
    n = len(values)
    for i in range(n):
        start = max(0, i - half)
        end = min(n, i + half + 1)
        smoothed[i] = np.mean(values[start:end])
    return smoothed


def get_column_consensus(column_counts: Dict[str, int]) -> str:
    if not column_counts:
        return "-"
    sorted_items = sorted(column_counts.items(), key=lambda x: -x[1])
    return sorted_items[0][0]


def analyze_conservation(
    aligned_sequences: List[str],
    window_size: int = 7,
    conserved_threshold: float = 0.3,
    variable_threshold: float = 0.7,
    min_conserved_length: int = 5,
    min_variable_length: int = 3,
) -> ConservationResult:
    if not aligned_sequences:
        return ConservationResult(
            shannon_entropy=np.array([]),
            weighted_score=np.array([]),
            smoothed_entropy=np.array([]),
            column_amino_acid_counts=[],
            column_consensus=[],
            total_columns=0,
            valid_columns=0,
        )

    n_seq = len(aligned_sequences)
    aln_len = len(aligned_sequences[0])

    shannon_entropy = np.zeros(aln_len, dtype=np.float64)
    weighted_score = np.zeros(aln_len, dtype=np.float64)
    column_counts_list: List[Dict[str, int]] = []
    column_consensus_list: List[str] = []
    valid_columns = 0

    for col in range(aln_len):
        counts: Dict[str, int] = {}
        n_valid = 0
        for seq in aligned_sequences:
            aa = seq[col]
            if aa != "-":
                counts[aa] = counts.get(aa, 0) + 1
                n_valid += 1

        if n_valid > 0:
            valid_columns += 1

        column_counts_list.append(counts)
        column_consensus_list.append(get_column_consensus(counts))
        shannon_entropy[col] = compute_shannon_entropy(counts, n_valid)
        weighted_score[col] = compute_weighted_conservation(counts, n_valid)

    smoothed_entropy = sliding_window_smooth(shannon_entropy, window_size)

    conserved_regions = _find_regions(
        smoothed_entropy,
        threshold=conserved_threshold,
        region_type="conserved",
        comparator=lambda v, t: v < t,
        min_length=min_conserved_length,
    )

    variable_regions = _find_regions(
        smoothed_entropy,
        threshold=variable_threshold,
        region_type="variable",
        comparator=lambda v, t: v > t,
        min_length=min_variable_length,
    )

    conserved_regions, variable_regions = _resolve_overlaps(
        conserved_regions, variable_regions, smoothed_entropy
    )

    _annotate_regions(
        conserved_regions,
        aligned_sequences,
        shannon_entropy,
        smoothed_entropy,
        column_consensus_list,
    )
    _annotate_regions(
        variable_regions,
        aligned_sequences,
        shannon_entropy,
        smoothed_entropy,
        column_consensus_list,
    )

    return ConservationResult(
        shannon_entropy=shannon_entropy,
        weighted_score=weighted_score,
        smoothed_entropy=smoothed_entropy,
        column_amino_acid_counts=column_counts_list,
        column_consensus=column_consensus_list,
        total_columns=aln_len,
        valid_columns=valid_columns,
        conserved_regions=conserved_regions,
        variable_regions=variable_regions,
    )


def _find_regions(
    smoothed_values: np.ndarray,
    threshold: float,
    region_type: str,
    comparator,
    min_length: int,
) -> List[Dict]:
    regions = []
    n = len(smoothed_values)
    start = None

    for i in range(n):
        if comparator(smoothed_values[i], threshold):
            if start is None:
                start = i
        else:
            if start is not None:
                length = i - start
                if length >= min_length:
                    regions.append({
                        "type": region_type,
                        "start": start,
                        "end": i - 1,
                        "length": length,
                    })
                start = None

    if start is not None:
        length = n - start
        if length >= min_length:
            regions.append({
                "type": region_type,
                "start": start,
                "end": n - 1,
                "length": length,
            })

    return regions


def _resolve_overlaps(
    conserved_regions: List[Dict],
    variable_regions: List[Dict],
    smoothed_entropy: np.ndarray,
) -> Tuple[List[Dict], List[Dict]]:
    if not conserved_regions or not variable_regions:
        return conserved_regions, variable_regions

    occupied = set()
    for reg in conserved_regions:
        for i in range(reg["start"], reg["end"] + 1):
            occupied.add(i)

    filtered_variable = []
    for reg in variable_regions:
        new_start = None
        new_end = None
        segments = []
        for i in range(reg["start"], reg["end"] + 1):
            if i not in occupied:
                if new_start is None:
                    new_start = i
                new_end = i
            else:
                if new_start is not None and new_end is not None:
                    length = new_end - new_start + 1
                    if length >= 3:
                        segments.append({
                            "type": "variable",
                            "start": new_start,
                            "end": new_end,
                            "length": length,
                        })
                new_start = None
                new_end = None
        if new_start is not None and new_end is not None:
            length = new_end - new_start + 1
            if length >= 3:
                segments.append({
                    "type": "variable",
                    "start": new_start,
                    "end": new_end,
                    "length": length,
                })
        filtered_variable.extend(segments)

    return conserved_regions, filtered_variable


def _annotate_regions(
    regions: List[Dict],
    aligned_sequences: List[str],
    shannon_entropy: np.ndarray,
    smoothed_entropy: np.ndarray,
    column_consensus: List[str],
):
    for idx, reg in enumerate(regions):
        start = reg["start"]
        end = reg["end"]
        entropies = shannon_entropy[start:end + 1]
        reg["id"] = idx + 1
        reg["avg_entropy"] = float(np.mean(entropies))
        reg["max_entropy"] = float(np.max(entropies))
        reg["min_entropy"] = float(np.min(entropies))
        reg["avg_smoothed_entropy"] = float(np.mean(smoothed_entropy[start:end + 1]))
        reg["consensus_sequence"] = "".join(column_consensus[start:end + 1])


def get_top_amino_acids(
    column_counts: Dict[str, int], n_valid: int, top_n: int = 3
) -> List[Tuple[str, float]]:
    if not column_counts or n_valid == 0:
        return []
    sorted_items = sorted(column_counts.items(), key=lambda x: -x[1])
    result = []
    for aa, count in sorted_items[:top_n]:
        result.append((aa, count / n_valid * 100))
    return result


def find_position_in_regions(
    col_idx: int, conserved_regions: List[Dict], variable_regions: List[Dict]
) -> Optional[Dict]:
    for reg in conserved_regions:
        if reg["start"] <= col_idx <= reg["end"]:
            return {"type": "conserved", "region": reg}
    for reg in variable_regions:
        if reg["start"] <= col_idx <= reg["end"]:
            return {"type": "variable", "region": reg}
    return None
