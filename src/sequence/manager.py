import uuid
from typing import List, Optional, Dict

from .io import Sequence


class SequenceManager:
    def __init__(self):
        self._sequences: Dict[str, Sequence] = {}
        self._order: List[str] = []

    def add(self, sequence: Sequence) -> str:
        if not sequence.id:
            sequence.id = str(uuid.uuid4())
        if sequence.id not in self._sequences:
            self._order.append(sequence.id)
        self._sequences[sequence.id] = sequence
        return sequence.id

    def remove(self, seq_id: str) -> bool:
        if seq_id in self._sequences:
            del self._sequences[seq_id]
            self._order.remove(seq_id)
            return True
        return False

    def get(self, seq_id: str) -> Optional[Sequence]:
        return self._sequences.get(seq_id)

    def get_all(self) -> List[Sequence]:
        return [self._sequences[sid] for sid in self._order]

    def get_ids(self) -> List[str]:
        return list(self._order)

    def clear(self):
        self._sequences.clear()
        self._order.clear()

    def __len__(self) -> int:
        return len(self._sequences)

    def __contains__(self, seq_id: str) -> bool:
        return seq_id in self._sequences

    def __iter__(self):
        return iter(self.get_all())
