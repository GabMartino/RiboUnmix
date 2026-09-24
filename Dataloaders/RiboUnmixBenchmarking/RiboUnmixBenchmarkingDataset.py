from __future__ import annotations

import numpy as np

from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDataset import (
    RiboUnmixMultiDataset,
)


class RiboUnmixBenchmarkingDataset(RiboUnmixMultiDataset):
    """RiboUnmix adapter for the benchmarking CDS parquet format.

    The benchmarking files store a CDS as a one-dimensional array of codon
    strings (``["ATG", "GCT", ...]``), while the original multi-dataset loader
    receives three nucleotide one-hot vectors per codon.  This subclass accepts
    the compact codon representation directly and produces the same 97 sequence
    features and collated batch contract as ``RiboUnmixMultiDataset``.
    """

    def _sequence_cell_to_codon_ids(self, sequence_cell) -> np.ndarray:
        """Accept compact IDs (the datamodule default) or raw codon strings."""
        raw = np.asarray(sequence_cell)
        if raw.ndim == 1 and np.issubdtype(raw.dtype, np.integer):
            codon_ids = np.ascontiguousarray(raw, dtype=np.int64)
            if codon_ids.size == 0 or np.any(codon_ids < 0) or np.any(
                codon_ids >= self.num_codons
            ):
                raise ValueError("Benchmarking CDS contains an invalid compact codon ID.")
            return np.ascontiguousarray(codon_ids, dtype=np.uint8)

        is_codon_strings = raw.ndim == 1 and (
            raw.dtype.kind in {"U", "S"}
            or (raw.dtype == object and (raw.size == 0 or isinstance(raw[0], str)))
        )
        if not is_codon_strings:
            return super()._sequence_cell_to_codon_ids(sequence_cell)
        if raw.size == 0:
            raise ValueError("Encountered an empty CDS in benchmarking data.")

        codon_to_id = {
            str(codon).upper().replace("U", "T"): int(codon_id)
            for codon, codon_id in self.codon_map.items()
        }
        codons = np.char.replace(np.char.upper(raw.astype(str)), "U", "T")
        invalid = sorted({str(codon) for codon in codons if codon not in codon_to_id})
        if invalid:
            raise ValueError(
                "Benchmarking CDS contains codons missing from codon_encoding: "
                f"{invalid[:10]}"
            )
        return np.ascontiguousarray(
            np.fromiter(
                (codon_to_id[str(codon)] for codon in codons),
                dtype=np.uint8,
                count=len(codons),
            )
        )
