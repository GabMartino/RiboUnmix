import unittest

from analyses.fetch_hek293_ncbi_metadata import characteristics, parse_soft, relation_ids, sample_row


class GeoMetadataTests(unittest.TestCase):
    def setUp(self):
        self.record = {
            "Sample_title": ["HEK293T_RiboSeq_rep1"],
            "Sample_characteristics_ch1": ["cell line: HEK293T", "treatment: none"],
            "Sample_series_id": ["GSE151986", "GSE151989"],
            "Sample_relation": [
                "BioSample: https://www.ncbi.nlm.nih.gov/biosample/SAMN15164537",
                "SRA: https://www.ncbi.nlm.nih.gov/sra?term=SRX8494456",
            ],
            "Sample_library_strategy": ["RNA-Seq"],
        }

    def test_repeated_soft_fields_are_preserved(self):
        result = parse_soft("^SAMPLE = GSM1\n!Sample_characteristics_ch1 = treatment: none\n"
                            "!Sample_characteristics_ch1 = cell line: HEK293T\n"
                            "ID_REF\tVALUE\n1\t20\n^SAMPLE = GSM2\n!Sample_title = another\n")
        self.assertEqual(len(result["GSM1"]["Sample_characteristics_ch1"]), 2)
        self.assertEqual(result["GSM2"]["Sample_title"], ["another"])

    def test_biosample_and_sra_accessions_not_confused(self):
        self.assertEqual(relation_ids(self.record, "SAMN"), ["SAMN15164537"])
        self.assertEqual(relation_ids(self.record, "SRX"), ["SRX8494456"])

    def test_rnaseq_library_strategy_is_not_assay_evidence(self):
        row = sample_row("li_2022_std", "GSE151986", "SAMN15164537", "GSM4594578", self.record)
        self.assertNotIn("check_assay", row["validation_flags"])
        self.assertEqual(row["replicate_annotation"], "not explicitly annotated")

    def test_cell_line_not_inferred_from_alias(self):
        self.record["Sample_characteristics_ch1"] = ["cell type: embryonic kidney cells"]
        row = sample_row("hek293_alias", "GSE151986", "SAMN15164537", "GSM4594578", self.record)
        self.assertEqual(row["cell_line"], "not reported")

    def test_characteristic_values_with_colons_are_retained(self):
        self.record["Sample_characteristics_ch1"] = ["treatment: CHX: 2 min"]
        self.assertEqual(characteristics(self.record), {"treatment": ["CHX: 2 min"]})


if __name__ == "__main__":
    unittest.main()
