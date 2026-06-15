import io
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import requests

from ..data.amino_acids import VALID_AMINO_ACIDS


@dataclass
class Sequence:
    name: str
    sequence: str
    source: str = "manual"
    description: str = ""
    id: str = field(default_factory=lambda: "")

    def __post_init__(self):
        if not self.id:
            self.id = self.name

    def __len__(self) -> int:
        return len(self.sequence)


def validate_sequence(sequence: str) -> Tuple[bool, List[int]]:
    invalid_positions = []
    for i, aa in enumerate(sequence.upper()):
        if aa not in VALID_AMINO_ACIDS and aa != "-":
            invalid_positions.append(i)
    return len(invalid_positions) == 0, invalid_positions


def parse_fasta_text(text: str, source: str = "paste") -> List[Sequence]:
    sequences = []
    lines = text.strip().splitlines()
    current_name = None
    current_desc = ""
    current_seq = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_name is not None and current_seq:
                seq_str = "".join(current_seq).upper().replace(" ", "")
                sequences.append(Sequence(
                    name=current_name,
                    sequence=seq_str,
                    source=source,
                    description=current_desc,
                ))
            header = line[1:].strip()
            parts = header.split(None, 1)
            current_name = parts[0] if parts else "unknown"
            current_desc = parts[1] if len(parts) > 1 else ""
            current_seq = []
        else:
            current_seq.append(line)

    if current_name is not None and current_seq:
        seq_str = "".join(current_seq).upper().replace(" ", "")
        sequences.append(Sequence(
            name=current_name,
            sequence=seq_str,
            source=source,
            description=current_desc,
        ))

    return sequences


def parse_fasta_file(file_content: bytes) -> List[Sequence]:
    text = file_content.decode("utf-8", errors="replace")
    return parse_fasta_text(text, source="file")


def fetch_uniprot_sequence(accession: str) -> Optional[Sequence]:
    accession = accession.strip()
    if not accession:
        return None

    try:
        url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            seqs = parse_fasta_text(response.text, source=f"UniProt:{accession}")
            if seqs:
                return seqs[0]
    except requests.RequestException:
        pass
    return None
