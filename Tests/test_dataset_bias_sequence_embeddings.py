from __future__ import annotations

from pathlib import Path
import unittest

import torch
import yaml

from Models.RiboUnmixModel.DatasetBiasSubmodel import DatasetBiasSubmodel
from Models.RiboUnmixModel.RiboUnmixModel import (
    build_bias_sequence_embedding_tables,
)


ROOT = Path(__file__).resolve().parents[1]


def _encoding(name: str) -> dict:
    with (ROOT / "Datasets" / "encodings" / name).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


class DatasetBiasSequenceEmbeddingTests(unittest.TestCase):
    @staticmethod
    def _minimal_bias_config(**overrides) -> dict:
        config = {
            "position_features": ["rel_pos"],
            "codon_embeddings_size": 2,
            "dataset_embeddings_size": 2,
            "num_datasets": 1,
            "num_codons": 64,
            "context_gru_hidden_size": 2,
            "context_gru_num_layers": 1,
            "dataset_multiplicative_allocation_bias_submodule_params": {
                "hidden_size": 2,
            },
            "dataset_log_sigma_submodule_params": {"hidden_size": 2},
        }
        config.update(overrides)
        return config

    def test_tables_match_direct_nucleotide_and_amino_acid_encodings(self) -> None:
        nt = _encoding("nt_encoding.yaml")
        codon_to_aa = _encoding("codon2aa.yaml")
        codon = _encoding("codon_encoding.yaml")
        aa = _encoding("aa_encoding.yaml")
        tables = build_bias_sequence_embedding_tables(
            nt_encoding=nt,
            codon_to_aa_encoding=codon_to_aa,
            codon_encoding=codon,
            aa_encoding=aa,
        )
        atg = int(codon["ATG"])
        self.assertEqual(tables["codon_nucleotide_ids"][atg], [0, 2, 3])
        self.assertEqual(tables["codon_amino_acid_ids"][atg], int(aa["M"]))

    def test_bias_branch_uses_all_three_token_embedding_families(self) -> None:
        nt = _encoding("nt_encoding.yaml")
        codon_to_aa = _encoding("codon2aa.yaml")
        codon = _encoding("codon_encoding.yaml")
        aa = _encoding("aa_encoding.yaml")
        tables = build_bias_sequence_embedding_tables(
            nt_encoding=nt,
            codon_to_aa_encoding=codon_to_aa,
            codon_encoding=codon,
            aa_encoding=aa,
        )
        model = DatasetBiasSubmodel(
            {
                "position_features": ["rel_pos"],
                "codon_embeddings_size": 5,
                "dataset_embeddings_size": 4,
                "num_datasets": 2,
                "num_codons": 64,
                "use_nucleotide_amino_acid_embeddings": True,
                "nucleotide_embeddings_size": 3,
                "amino_acid_embeddings_size": 2,
                **tables,
                "context_gru_hidden_size": 4,
                "context_gru_num_layers": 1,
                "dataset_multiplicative_allocation_bias_submodule_params": {
                    "hidden_size": 4,
                },
                "dataset_log_sigma_submodule_params": {"hidden_size": 4},
            }
        )
        # Dataset + codon + three nucleotide positions + amino acid.
        self.assertEqual(model.local_context_gru.rnn.input_size, 4 + 5 + 3 * 3 + 2)
        with torch.no_grad():
            model.observation_bias_head.log_bias_head.weight.fill_(0.1)
        ids = torch.tensor([[int(codon["ATG"]), int(codon["AAA"]), 0]])
        mask = torch.tensor([[True, True, False]])
        out = model(
            dataset_ids=torch.tensor([0]),
            mask=mask,
            codon_ids=ids,
            position_features=torch.zeros(1, 3, 1),
        )
        self.assertEqual(tuple(out["gamma_raw"].shape), (1, 3))
        self.assertEqual(float(out["gamma_raw"][0, 2].detach()), 0.0)
        self.assertTrue(torch.isfinite(out["log_sigma"]).all())
        out["gamma_raw"].sum().backward()
        self.assertGreater(
            float(model.nucleotide_embedding.weight.grad.detach().abs().sum()), 0.0
        )
        self.assertGreater(
            float(model.amino_acid_embedding.weight.grad.detach().abs().sum()), 0.0
        )

    def test_raw_log_gamma_is_bounded_before_centering(self) -> None:
        model = DatasetBiasSubmodel(
            self._minimal_bias_config(raw_log_gamma_bound=4.0)
        )
        with torch.no_grad():
            model.observation_bias_head.log_bias_head.weight.zero_()
            model.observation_bias_head.log_bias_head.bias.fill_(10.0)

        mask = torch.tensor([[True, True, False]])
        out = model(
            dataset_ids=torch.tensor([0]),
            mask=mask,
            codon_ids=torch.zeros(1, 3, dtype=torch.long),
            position_features=torch.zeros(1, 3, 1),
            compute_log_sigma=False,
        )

        torch.testing.assert_close(
            out["gamma_raw"],
            torch.tensor([[4.0, 4.0, 0.0]]),
        )
        torch.testing.assert_close(
            out["gamma_raw_bound_active_fraction"],
            torch.tensor([1.0]),
        )

    def test_raw_log_gamma_bound_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "raw_log_gamma_bound"):
            DatasetBiasSubmodel(
                self._minimal_bias_config(raw_log_gamma_bound=0.0)
            )


if __name__ == "__main__":
    unittest.main()
