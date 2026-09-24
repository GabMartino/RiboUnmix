from __future__ import annotations

from typing import NamedTuple


class ValidatedTranscriptMetadata(NamedTuple):
    """CPU-collated transcript invariants for the model's trusted input path.

    Collate validates repeated codons and optional sequence features before
    creating this value. Masks and position features follow from those lengths;
    dense biological inputs are constructed from the same canonical codons.
    Keep these fields as Python integers, so worker IPC, pinning and Lightning
    device transfer preserve host lengths without a CUDA-to-CPU copy.

    This metadata describes the exact collated row order. Callers that change
    batch rows or sequence inputs must discard it and use model validation.
    """

    group_indices: tuple[int, ...]
    rows_by_transcript: tuple[tuple[int, ...], ...]
    lengths: tuple[int, ...]

    @property
    def canonical_rows(self) -> tuple[int, ...]:
        return tuple(rows[0] for rows in self.rows_by_transcript)

    @property
    def unique_lengths(self) -> tuple[int, ...]:
        return tuple(self.lengths[row] for row in self.canonical_rows)
