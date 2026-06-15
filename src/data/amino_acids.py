import numpy as np

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
AMINO_ACID_SET = set(AMINO_ACIDS)
VALID_AMINO_ACIDS = AMINO_ACID_SET

AMINO_ACID_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

_amino_acid_one_hot = np.eye(len(AMINO_ACIDS), dtype=np.float32)


def amino_acid_one_hot(aa: str) -> np.ndarray:
    if aa in AMINO_ACID_INDEX:
        return _amino_acid_one_hot[AMINO_ACID_INDEX[aa]].copy()
    return np.zeros(len(AMINO_ACIDS), dtype=np.float32)


HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

MOLECULAR_WEIGHT = {
    "A": 89.09, "R": 174.20, "N": 132.12, "D": 133.10, "C": 121.16,
    "Q": 146.15, "E": 147.13, "G": 75.07, "H": 155.16, "I": 131.17,
    "L": 131.17, "K": 146.19, "M": 149.21, "F": 165.19, "P": 115.13,
    "S": 105.09, "T": 119.12, "W": 204.23, "Y": 181.19, "V": 117.15,
}

ISOELECTRIC_POINT = {
    "A": 6.01, "R": 10.76, "N": 5.41, "D": 2.77, "C": 5.07,
    "Q": 5.65, "E": 3.22, "G": 5.97, "H": 7.59, "I": 6.02,
    "L": 5.98, "K": 9.74, "M": 5.74, "F": 5.48, "P": 6.30,
    "S": 5.68, "T": 5.60, "W": 5.89, "Y": 5.66, "V": 5.96,
}

CHARGE_PH7 = {
    "A": 0, "R": +1, "N": 0, "D": -1, "C": 0,
    "Q": 0, "E": -1, "G": 0, "H": 0, "I": 0,
    "L": 0, "K": +1, "M": 0, "F": 0, "P": 0,
    "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0,
}


def physicochemical_properties(aa: str) -> np.ndarray:
    props = np.zeros(4, dtype=np.float32)
    if aa in HYDROPHOBICITY:
        props[0] = HYDROPHOBICITY[aa]
        props[1] = MOLECULAR_WEIGHT[aa]
        props[2] = ISOELECTRIC_POINT[aa]
        props[3] = CHARGE_PH7[aa]
    return props


def encode_sequence_one_hot(sequence: str, window_size: int = None) -> np.ndarray:
    encoded = np.zeros((len(sequence), len(AMINO_ACIDS) + 1), dtype=np.float32)
    for i, aa in enumerate(sequence):
        if aa in AMINO_ACID_INDEX:
            encoded[i, AMINO_ACID_INDEX[aa]] = 1.0
        else:
            encoded[i, -1] = 1.0
    return encoded


def encode_sequence_blosum62(sequence: str) -> np.ndarray:
    from .blosum62 import BLOSUM62
    encoded = np.zeros((len(sequence), len(AMINO_ACIDS)), dtype=np.float32)
    for i, aa in enumerate(sequence):
        if aa in BLOSUM62:
            for j, ref_aa in enumerate(AMINO_ACIDS):
                encoded[i, j] = BLOSUM62[aa].get(ref_aa, -4)
    return encoded


def encode_sequence_with_properties(sequence: str) -> np.ndarray:
    blosum = encode_sequence_blosum62(sequence)
    props = np.zeros((len(sequence), 4), dtype=np.float32)
    for i, aa in enumerate(sequence):
        props[i] = physicochemical_properties(aa)
    return np.concatenate([blosum, props], axis=1)


def blosum62_encoding(aa: str) -> np.ndarray:
    from .blosum62 import BLOSUM62
    encoded = np.zeros(len(AMINO_ACIDS), dtype=np.float32)
    if aa in BLOSUM62:
        for j, ref_aa in enumerate(AMINO_ACIDS):
            encoded[j] = BLOSUM62[aa].get(ref_aa, -4)
    return encoded
