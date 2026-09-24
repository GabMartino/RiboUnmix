from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_synthetic_pi_demo as pi_demo


class SyntheticPiDemoLauncherTests(unittest.TestCase):
    def _make_source_tree(self, root: Path) -> None:
        for dataset_name, depth in pi_demo.PANEL:
            source = (
                root
                / "Datasets/data/weighted_synthetic"
                / depth
                / f"{dataset_name}.parquet"
            )
            source.parent.mkdir(parents=True, exist_ok=True)
            source.touch()

    def test_canonical_names_and_expected_reference_pi(self) -> None:
        expected_pi = {
            "equal": (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
            "quality": (1.0 / 6.0, 1.0 / 3.0, 1.0 / 2.0),
            "reversed": (1.0 / 2.0, 1.0 / 3.0, 1.0 / 6.0),
        }
        canonical_names = [dataset_name for dataset_name, _ in pi_demo.PANEL]
        measured_quality = {
            canonical_names[0]: {
                "median_read_density": 0.28,
                "median_coverage": 0.26,
            },
            canonical_names[1]: {
                "median_read_density": 2.73,
                "median_coverage": 0.87,
            },
            canonical_names[2]: {
                "median_read_density": 23.18,
                "median_coverage": 0.998,
            },
        }

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self._make_source_tree(root)
            with (
                patch.object(pi_demo, "ROOT", root),
                patch.object(
                    pi_demo,
                    "_measured_panel_quality",
                    return_value=measured_quality,
                ),
            ):
                for policy, expected in expected_pi.items():
                    output = root / "outputs" / policy
                    names, sources, encoding, quality = (
                        pi_demo._write_launcher_inputs(
                            output_dir=output,
                            policy=policy,
                            seed=42,
                        )
                    )
                    self.assertEqual(names, canonical_names)
                    self.assertEqual([source.stem for source in sources], names)
                    self.assertEqual(
                        [line.split(":", 1)[0] for line in encoding.read_text().splitlines()],
                        names,
                    )
                    self.assertEqual(
                        [line.split("\t", 1)[0] for line in quality.read_text().splitlines()[1:]],
                        names,
                    )

                    manifest = json.loads(
                        (output / "pi_demo_design_manifest.json").read_text()
                    )
                    actual = tuple(
                        row["expected_reference_pi"] for row in manifest["panel"]
                    )
                    for observed, target in zip(actual, expected, strict=True):
                        self.assertAlmostEqual(observed, target)


if __name__ == "__main__":
    unittest.main()
