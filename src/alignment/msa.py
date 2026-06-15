from dataclasses import dataclass, field
from typing import List, Dict, Tuple

import numpy as np

from .pairwise import needleman_wunsch
from ..data.amino_acids import AMINO_ACIDS, AMINO_ACID_INDEX
from ..data.blosum62 import blosum62_score


@dataclass
class MultipleAlignmentResult:
    names: List[str]
    aligned_sequences: List[str]
    consensus: str = ""
    column_scores: np.ndarray = field(default_factory=lambda: np.array([]))

    def __len__(self) -> int:
        return len(self.aligned_sequences)

    @property
    def alignment_length(self) -> int:
        return len(self.aligned_sequences[0]) if self.aligned_sequences else 0


def _compute_profile(sequences: List[str]) -> np.ndarray:
    if not sequences:
        return np.zeros((0, len(AMINO_ACIDS) + 1))

    n_seq = len(sequences)
    length = len(sequences[0])
    profile = np.zeros((length, len(AMINO_ACIDS) + 1), dtype=np.float64)

    for seq in sequences:
        for j, aa in enumerate(seq):
            if aa in AMINO_ACID_INDEX:
                profile[j, AMINO_ACID_INDEX[aa]] += 1.0
            else:
                profile[j, -1] += 1.0

    if n_seq > 0:
        profile /= n_seq

    return profile


def _profile_score(p1_col: np.ndarray, p2_col: np.ndarray) -> float:
    score = 0.0
    for i, freq_i in enumerate(p1_col[:-1]):
        if freq_i == 0:
            continue
        for j, freq_j in enumerate(p2_col[:-1]):
            if freq_j == 0:
                continue
            aa_i = AMINO_ACIDS[i]
            aa_j = AMINO_ACIDS[j]
            score += freq_i * freq_j * blosum62_score(aa_i, aa_j)
    return score


def _align_profiles(
    profile1: np.ndarray,
    profile2: np.ndarray,
    seqs1: List[str],
    seqs2: List[str],
    gap_open: float = -10,
    gap_extend: float = -0.5,
) -> List[str]:
    n, m = profile1.shape[0], profile2.shape[0]
    n_feat = profile1.shape[1]

    NEG_INF = float("-inf")

    M = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    Ix = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    Iy = np.full((n + 1, m + 1), NEG_INF, dtype=np.float64)
    T = np.zeros((n + 1, m + 1), dtype=np.int8)
    Tx = np.zeros((n + 1, m + 1), dtype=np.int8)
    Ty = np.zeros((n + 1, m + 1), dtype=np.int8)

    M[0, 0] = 0

    for i in range(1, n + 1):
        Ix[i, 0] = gap_open + (i - 1) * gap_extend

    for j in range(1, m + 1):
        Iy[0, j] = gap_open + (j - 1) * gap_extend

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score_mm = _profile_score(profile1[i-1], profile2[j-1])

            m_vals = np.array([M[i-1, j-1], Ix[i-1, j-1], Iy[i-1, j-1]])
            M[i, j] = score_mm + m_vals.max()
            T[i, j] = int(np.argmax(m_vals))

            ix_vals = np.array([M[i-1, j] + gap_open, Ix[i-1, j] + gap_extend, Iy[i-1, j] + gap_open])
            Ix[i, j] = ix_vals.max()
            Tx[i, j] = int(np.argmax(ix_vals))

            iy_vals = np.array([M[i, j-1] + gap_open, Ix[i, j-1] + gap_open, Iy[i, j-1] + gap_extend])
            Iy[i, j] = iy_vals.max()
            Ty[i, j] = int(np.argmax(iy_vals))

    final_vals = np.array([M[n, m], Ix[n, m], Iy[n, m]])
    current = int(np.argmax(final_vals))

    path1, path2 = [], []
    i, j = n, m

    while i > 0 or j > 0:
        if current == 0 and i > 0 and j > 0:
            path1.append(i - 1)
            path2.append(j - 1)
            prev_current = T[i, j]
            i -= 1
            j -= 1
            current = prev_current
        elif (current == 1 or j == 0) and i > 0:
            path1.append(i - 1)
            path2.append(-1)
            prev_current = Tx[i, j]
            i -= 1
            current = prev_current
        elif j > 0:
            path1.append(-1)
            path2.append(j - 1)
            prev_current = Ty[i, j]
            j -= 1
            current = prev_current
        else:
            break

    path1.reverse()
    path2.reverse()

    aligned = []
    for seq in seqs1:
        new_seq = []
        for p in path1:
            if p >= 0:
                new_seq.append(seq[p])
            else:
                new_seq.append("-")
        aligned.append("".join(new_seq))

    for seq in seqs2:
        new_seq = []
        for p in path2:
            if p >= 0:
                new_seq.append(seq[p])
            else:
                new_seq.append("-")
        aligned.append("".join(new_seq))

    return aligned


def _pairwise_distance(seq1: str, seq2: str) -> float:
    result = needleman_wunsch(seq1, seq2)
    a1, a2 = result.seq1_aligned, result.seq2_aligned
    matches = sum(1 for c1, c2 in zip(a1, a2) if c1 == c2 and c1 != "-")
    aligned_positions = sum(1 for c1, c2 in zip(a1, a2) if c1 != "-" and c2 != "-")
    if aligned_positions == 0:
        return 1.0
    return 1.0 - (matches / aligned_positions)


def _upgma(distances: np.ndarray) -> List[Tuple[int, int]]:
    n = distances.shape[0]
    clusters = {i: [i] for i in range(n)}
    merge_order = []

    dist = distances.copy().astype(np.float64)
    np.fill_diagonal(dist, np.inf)

    while len(clusters) > 1:
        cluster_ids = list(clusters.keys())
        min_dist = np.inf
        min_i, min_j = -1, -1

        for idx_i in range(len(cluster_ids)):
            for idx_j in range(idx_i + 1, len(cluster_ids)):
                ci, cj = cluster_ids[idx_i], cluster_ids[idx_j]
                d = dist[ci, cj]
                if d < min_dist:
                    min_dist = d
                    min_i, min_j = ci, cj

        new_id = max(clusters.keys()) + 1
        merge_order.append((min_i, min_j))
        new_cluster = clusters[min_i] + clusters[min_j]
        clusters[new_id] = new_cluster

        old_ids = [k for k in clusters.keys() if k != min_i and k != min_j]
        for old_id in old_ids:
            size_i = len(clusters[min_i])
            size_j = len(clusters[min_j])
            if old_id < dist.shape[0] and min_i < dist.shape[0] and min_j < dist.shape[0]:
                new_dist = (size_i * dist[old_id, min_i] + size_j * dist[old_id, min_j]) / (size_i + size_j)
            else:
                new_dist = np.inf
            if new_id >= dist.shape[0]:
                new_shape = (new_id + 1, new_id + 1)
                new_dist_matrix = np.full(new_shape, np.inf, dtype=np.float64)
                new_dist_matrix[:dist.shape[0], :dist.shape[1]] = dist
                dist = new_dist_matrix
            dist[old_id, new_id] = new_dist
            dist[new_id, old_id] = new_dist

        del clusters[min_i]
        del clusters[min_j]

    return merge_order


def progressive_alignment(
    names: List[str],
    sequences: List[str],
    gap_open: float = -10,
    gap_extend: float = -0.5,
) -> MultipleAlignmentResult:
    if len(sequences) == 0:
        return MultipleAlignmentResult(names=[], aligned_sequences=[])

    if len(sequences) == 1:
        result = MultipleAlignmentResult(names=names, aligned_sequences=list(sequences))
        result.consensus = sequences[0]
        result.column_scores = np.ones(len(sequences[0]))
        return result

    if len(sequences) == 2:
        nw = needleman_wunsch(sequences[0], sequences[1], gap_open, gap_extend)
        aligned = [nw.seq1_aligned, nw.seq2_aligned]
        result = MultipleAlignmentResult(names=names, aligned_sequences=aligned)
        result.consensus = _compute_consensus(aligned)
        result.column_scores = _compute_conservation(aligned)
        return result

    n = len(sequences)
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = _pairwise_distance(sequences[i], sequences[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    merge_order = _upgma(dist_matrix)

    clusters = {i: {"names": [names[i]], "seqs": [sequences[i]]} for i in range(n)}

    for ci, cj in merge_order:
        if ci not in clusters or cj not in clusters:
            continue

        cluster1 = clusters[ci]
        cluster2 = clusters[cj]

        if len(cluster1["seqs"]) == 1 and len(cluster2["seqs"]) == 1:
            nw = needleman_wunsch(cluster1["seqs"][0], cluster2["seqs"][0], gap_open, gap_extend)
            new_seqs = [nw.seq1_aligned, nw.seq2_aligned]
        else:
            profile1 = _compute_profile(cluster1["seqs"])
            profile2 = _compute_profile(cluster2["seqs"])
            new_seqs = _align_profiles(profile1, profile2, cluster1["seqs"], cluster2["seqs"], gap_open, gap_extend)

        new_id = max(clusters.keys()) + 1
        clusters[new_id] = {
            "names": cluster1["names"] + cluster2["names"],
            "seqs": new_seqs,
        }
        del clusters[ci]
        del clusters[cj]

    final_id = list(clusters.keys())[0]
    final = clusters[final_id]

    result = MultipleAlignmentResult(
        names=final["names"],
        aligned_sequences=final["seqs"],
    )
    result.consensus = _compute_consensus(final["seqs"])
    result.column_scores = _compute_conservation(final["seqs"])

    name_to_idx = {name: i for i, name in enumerate(final["names"])}
    ordered_names = []
    ordered_seqs = []
    for name in names:
        if name in name_to_idx:
            idx = name_to_idx[name]
            ordered_names.append(name)
            ordered_seqs.append(final["seqs"][idx])

    result.names = ordered_names
    result.aligned_sequences = ordered_seqs

    return result


def _compute_consensus(aligned_sequences: List[str]) -> str:
    if not aligned_sequences:
        return ""

    length = len(aligned_sequences[0])
    n_seq = len(aligned_sequences)
    consensus = []

    for j in range(length):
        counts: Dict[str, int] = {}
        for seq in aligned_sequences:
            aa = seq[j]
            counts[aa] = counts.get(aa, 0) + 1

        sorted_counts = sorted(counts.items(), key=lambda x: -x[1])
        best_aa, best_count = sorted_counts[0]

        if best_count == n_seq and best_aa != "-":
            consensus.append(best_aa.upper())
        elif best_count / n_seq > 0.5 and best_aa != "-":
            consensus.append(best_aa.lower())
        else:
            consensus.append(".")

    return "".join(consensus)


def _compute_conservation(aligned_sequences: List[str]) -> np.ndarray:
    if not aligned_sequences:
        return np.array([])

    length = len(aligned_sequences[0])
    n_seq = len(aligned_sequences)
    scores = np.zeros(length, dtype=np.float64)

    for j in range(length):
        counts: Dict[str, int] = {}
        n_valid = 0
        for seq in aligned_sequences:
            aa = seq[j]
            if aa != "-":
                counts[aa] = counts.get(aa, 0) + 1
                n_valid += 1

        if n_valid == 0:
            scores[j] = 0.0
            continue

        entropy = 0.0
        for aa, count in counts.items():
            freq = count / n_valid
            if freq > 0:
                entropy -= freq * np.log2(freq)

        max_entropy = np.log2(min(len(counts), 20)) if counts else 1.0
        if max_entropy > 0:
            scores[j] = 1.0 - (entropy / max_entropy)
        else:
            scores[j] = 1.0

    return scores
