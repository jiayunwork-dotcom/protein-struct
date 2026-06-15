from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..data.blosum62 import blosum62_score


@dataclass
class AlignmentResult:
    seq1_aligned: str
    seq2_aligned: str
    score: float
    start1: int = 0
    end1: int = 0
    start2: int = 0
    end2: int = 0
    is_global: bool = True


def needleman_wunsch(
    seq1: str,
    seq2: str,
    gap_open: float = -10,
    gap_extend: float = -0.5,
) -> AlignmentResult:
    n, m = len(seq1), len(seq2)

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
            score_mm = blosum62_score(seq1[i-1], seq2[j-1])

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
    final_score = final_vals.max()
    current = int(np.argmax(final_vals))

    aligned1, aligned2 = [], []
    i, j = n, m

    while i > 0 or j > 0:
        if current == 0 and i > 0 and j > 0:
            aligned1.append(seq1[i-1])
            aligned2.append(seq2[j-1])
            prev_current = T[i, j]
            i -= 1
            j -= 1
            current = prev_current
        elif (current == 1 or j == 0) and i > 0:
            aligned1.append(seq1[i-1])
            aligned2.append("-")
            prev_current = Tx[i, j]
            i -= 1
            current = prev_current
        elif j > 0:
            aligned1.append("-")
            aligned2.append(seq2[j-1])
            prev_current = Ty[i, j]
            j -= 1
            current = prev_current
        else:
            break

    return AlignmentResult(
        seq1_aligned="".join(reversed(aligned1)),
        seq2_aligned="".join(reversed(aligned2)),
        score=final_score,
        start1=0,
        end1=n,
        start2=0,
        end2=m,
        is_global=True,
    )


def smith_waterman(
    seq1: str,
    seq2: str,
    gap_open: float = -10,
    gap_extend: float = -0.5,
) -> AlignmentResult:
    n, m = len(seq1), len(seq2)

    H = np.zeros((n + 1, m + 1), dtype=np.float64)
    E = np.zeros((n + 1, m + 1), dtype=np.float64)
    F = np.zeros((n + 1, m + 1), dtype=np.float64)
    T = np.zeros((n + 1, m + 1), dtype=np.int8)

    max_score = 0.0
    max_i, max_j = 0, 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            E[i, j] = max(H[i, j-1] + gap_open, E[i, j-1] + gap_extend, 0)
            F[i, j] = max(H[i-1, j] + gap_open, F[i-1, j] + gap_extend, 0)

            score_mm = blosum62_score(seq1[i-1], seq2[j-1])
            diag = H[i-1, j-1] + score_mm

            vals = np.array([0, diag, F[i, j], E[i, j]])
            H[i, j] = vals.max()
            T[i, j] = int(np.argmax(vals))

            if H[i, j] > max_score:
                max_score = H[i, j]
                max_i, max_j = i, j

    aligned1, aligned2 = [], []
    i, j = max_i, max_j
    end1, end2 = i, j

    while i > 0 and j > 0 and H[i, j] > 0:
        t = T[i, j]
        if t == 1:
            aligned1.append(seq1[i-1])
            aligned2.append(seq2[j-1])
            i -= 1
            j -= 1
        elif t == 2:
            aligned1.append(seq1[i-1])
            aligned2.append("-")
            i -= 1
        elif t == 3:
            aligned1.append("-")
            aligned2.append(seq2[j-1])
            j -= 1
        else:
            break

    return AlignmentResult(
        seq1_aligned="".join(reversed(aligned1)),
        seq2_aligned="".join(reversed(aligned2)),
        score=max_score,
        start1=i,
        end1=end1,
        start2=j,
        end2=end2,
        is_global=False,
    )


def format_alignment(result: AlignmentResult) -> str:
    a1 = result.seq1_aligned
    a2 = result.seq2_aligned
    middle = []
    for c1, c2 in zip(a1, a2):
        if c1 == c2 and c1 != "-":
            middle.append("|")
        elif c1 != "-" and c2 != "-":
            middle.append(".")
        else:
            middle.append(" ")
    return "\n".join([a1, "".join(middle), a2])
