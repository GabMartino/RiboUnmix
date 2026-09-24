from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import unittest

import numpy as np
import torch
from torch.nn.utils.rnn import pad_packed_sequence
import yaml

from Dataloaders.RiboUnmixMultiDataset.RiboUnmixMultiDataset import (
    RiboUnmixMultiDataset,
)


ROOT = Path(__file__).resolve().parents[1]


def _yaml(relative_path: str):
    with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class CompactSequenceCollateTests(unittest.TestCase):
    def test_collate_packs_one_dense_biological_sequence_per_transcript(self) -> None:
        nt = _yaml("Datasets/encodings/nt_encoding.yaml")
        codon_to_aa = _yaml("Datasets/encodings/codon2aa.yaml")
        codon = _yaml("Datasets/encodings/codon_encoding.yaml")
        aa = _yaml("Datasets/encodings/aa_encoding.yaml")
        codons_by_transcript = {
            "t0": ["ATG", "AAA", "TAA"],
            "t1": ["ATG", "CCC", "GGG", "TGA"],
        }

        profiles = defaultdict(dict)
        replicas = defaultdict(dict)
        weights = defaultdict(dict)
        for transcript_id, transcript_codons in codons_by_transcript.items():
            for dataset_name in ("d0", "d1"):
                profile = np.arange(1, len(transcript_codons) + 1, dtype=np.float32)
                profiles[transcript_id][dataset_name] = profile
                replicas[transcript_id][dataset_name] = np.stack((profile, profile + 1.0))
                weights[transcript_id][dataset_name] = 1.0

        transcript_ids = list(codons_by_transcript)
        lengths = np.asarray(
            [len(codons_by_transcript[transcript_id]) for transcript_id in transcript_ids]
        )
        shared = {
            "transcript_id": np.asarray(transcript_ids),
            "ref": np.asarray(
                [codons_by_transcript[transcript_id] for transcript_id in transcript_ids],
                dtype=object,
            ),
            "sequence_representation": "codon_tokens",
            "css": np.asarray([[], []], dtype=object),
            "ribo_profiles": profiles,
            "ribo_replicas": replicas,
            "sample_weights": weights,
            "dataset_quality_ranks": {"d0": 1.0, "d1": 2.0},
            "dataset_quality_weights": {"d0": 1.0, "d1": 1.0},
            "lengths": lengths,
            "datasets_names": ["d0", "d1"],
            "sequence_features": {},
        }
        dataset = RiboUnmixMultiDataset(
            nt_encoding=nt,
            codon_to_aa_encoding=codon_to_aa,
            codon_encoding=codon,
            aa_encoding=aa,
            datasets_encoding={"d0": 0, "d1": 1},
            transcripts_ids=["t1", "t0"],
            data=shared,
            lengths=lengths,
        )
        batch = dataset.collate_fn([dataset[index] for index in range(len(dataset))])
        biological_padded, biological_lengths = pad_packed_sequence(
            batch[2],
            batch_first=True,
        )

        self.assertEqual(int(batch[0].numel()), 4)
        self.assertEqual(int(batch[2].batch_sizes[0]), 2)
        self.assertEqual(batch[11].tolist(), [0, 0, 1, 1])
        self.assertEqual(tuple(biological_padded.shape), (2, 4, 97))
        self.assertEqual(biological_lengths.tolist(), [4, 3])
        self.assertFalse(hasattr(dataset, "_feature_cache"))
        compact_bytes = sum(
            value.nbytes for value in dataset._codon_id_cache if value is not None
        )
        dense_bytes = sum(lengths) * 97 * np.dtype(np.float32).itemsize
        self.assertLess(compact_bytes, dense_bytes / 20)


if __name__ == "__main__":
    unittest.main()
