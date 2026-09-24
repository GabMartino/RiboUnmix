import unittest

import pandas as pd

from analyses.plot_gamma_projection_metadata import attach_metadata, platform_label


class MetadataOverlayTests(unittest.TestCase):
    def test_mixed_platform_is_not_silently_reduced(self):
        self.assertEqual(platform_label("Illumina HiSeq 2000 | NextSeq 500"), "Mixed platforms")
        self.assertEqual(platform_label("Illumina NextSeq 500"), "NextSeq 500")

    def test_join_preserves_original_coordinates_and_order(self):
        coordinates = pd.DataFrame({
            "dataset": ["b", "a"], "gse": ["GSE2", "GSE1"],
            "coordinate_1": [.3, -.7], "coordinate_2": [.1, .4],
        })
        datasets = pd.DataFrame({
            "dataset": ["a", "b"], "gse": ["GSE1", "GSE2"],
            "instruments": ["NextSeq 500", "Illumina HiSeq 2000"],
            "gsms": ["GSM1", "GSM2"], "study_url": ["url1", "url2"],
        })
        kits = pd.DataFrame({
            "dataset": ["a", "b"], "gse": ["GSE1", "GSE2"],
            "riboseq_library_kit": ["Not reported / unclear", "SMARTer smRNA-Seq"],
        })
        result = attach_metadata(coordinates, datasets, kits)
        pd.testing.assert_frame_equal(result[coordinates.columns], coordinates)
        self.assertEqual(result.sequencing_platform.tolist(), ["HiSeq 2000", "NextSeq 500"])
        with self.assertRaises(ValueError):
            attach_metadata(coordinates, datasets, kits.iloc[:1])


if __name__ == "__main__":
    unittest.main()
