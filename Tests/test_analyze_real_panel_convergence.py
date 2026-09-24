from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")


def _load_analysis_module():
    path = ROOT / "analyses/analyze_real_panel_convergence.py"
    spec = importlib.util.spec_from_file_location("real_panel_analysis_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ANALYSIS = _load_analysis_module()


class RealPanelConvergenceAnalysisTests(unittest.TestCase):
    def test_raw_duplicates_are_checked_across_streaming_batches_without_pandas(self):
        with tempfile.TemporaryDirectory() as temporary:
            prediction = Path(temporary) / "prediction.parquet"
            rows = [dict(transcript_id="t1", length=3, mask=[True, True, True, False],
                         L_bio=[0.5, 1.0, 1.5, 0.0]) for _ in range(35)]
            pd.DataFrame(rows).to_parquet(prediction, index=False)
            kwargs = dict(panel_name="panel_01", run_identifier="test_run",
                          prediction_path=prediction, expected_ids={"t1"}, mean_one_tolerance=1e-4)
            with patch.object(ANALYSIS.pd, "read_parquet", side_effect=AssertionError("No raw pandas reads")):
                profiles, checks = ANALYSIS._extract_panel_profiles(**kwargs)
            np.testing.assert_allclose(profiles["t1"]["values"], [0.5, 1.0, 1.5])
            self.assertEqual(checks.iloc[0].number_of_prediction_dataset_rows, 35)
            rows[-1]["L_bio"] = [0.4, 1.0, 1.6, 0.0]
            pd.DataFrame(rows).to_parquet(prediction, index=False)
            with self.assertRaisesRegex(ValueError, "differs across dataset rows"):
                ANALYSIS._extract_panel_profiles(**kwargs)

    def test_no_argument_default_targets_requested_a100_run(self) -> None:
        args = ANALYSIS.parse_args([])
        self.assertEqual(
            args.run_root,
            ROOT / "results/my_panels_a100_b32_20260906_114323",
        )
        self.assertFalse(args.require_all_panels)

    def test_six_pairs_common_ids_and_mean_one_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            transcript_ids = ["t1", "t2", "t3"]
            panels = {f"panel_{index:02d}": [f"dataset_{index}"] for index in range(1, 5)}
            (run_root / "common_split_manifest.json").write_text(
                json.dumps({"common_test_ids": transcript_ids}), encoding="utf-8"
            )
            (run_root / "panel_manifest.json").write_text(
                json.dumps(
                    {
                        "outer_run_identifier": "test_run",
                        "panels": panels,
                    }
                ),
                encoding="utf-8",
            )
            assignment_rows = []
            base_profiles = {
                "t1": np.asarray([0.5, 1.0, 1.5], dtype=np.float32),
                "t2": np.asarray([0.3, 0.7, 1.1, 1.9], dtype=np.float32),
                "t3": np.asarray([1.4, 0.8, 0.8], dtype=np.float32),
            }
            for panel_index, panel_name in enumerate(sorted(panels), start=1):
                prediction_dir = run_root / panel_name / "predictions"
                prediction_dir.mkdir(parents=True)
                prediction_rows = []
                for transcript_id in transcript_ids:
                    base = base_profiles[transcript_id].astype(np.float64)
                    perturbation = np.linspace(-1.0, 1.0, len(base))
                    perturbation -= perturbation.mean()
                    profile = base + panel_index * 0.01 * perturbation
                    self.assertAlmostEqual(float(profile.mean()), 1.0, places=7)
                    for _ in range(2):
                        prediction_rows.append(
                            {
                                "transcript_id": transcript_id,
                                "length": len(profile),
                                "mask": np.ones(len(profile), dtype=bool),
                                "L_bio": profile.astype(np.float32),
                            }
                        )
                prediction_path = prediction_dir / "prediction.parquet"
                pd.DataFrame(prediction_rows).to_parquet(prediction_path, index=False)
                if panel_name == "panel_04":
                    # Exercise fallback from an absent final-consolidation
                    # wrapper and remapping of an HPC absolute artifact path.
                    (prediction_dir / "prediction_checkpoint_manifest.json").write_text(
                        json.dumps(
                            {
                                "best_val_loss": {
                                    "split_name": "test",
                                    "output_path": (
                                        "/leonardo_work/copied_run/"
                                        f"{prediction_path.name}"
                                    ),
                                }
                            }
                        ),
                        encoding="utf-8",
                    )
                else:
                    (
                        run_root
                        / panel_name
                        / "scientific_checkpoint_manifest.json"
                    ).write_text(
                        json.dumps(
                            {
                                "checkpoint_variant": "best_val_loss",
                                "prediction_path": str(prediction_path),
                            }
                        ),
                        encoding="utf-8",
                    )
                assignment_rows.append(
                    {
                        "dataset_name": f"dataset_{panel_index}",
                        "panel": panel_name,
                        "median_read_density": float(panel_index),
                        "log1p_median_read_density": float(np.log1p(panel_index)),
                        "median_positive_codon_coverage": 0.2 * panel_index,
                        "number_of_eligible_transcripts": 100 + panel_index,
                        "median_replica_PCC": 0.7 + 0.02 * panel_index,
                    }
                )
            pd.DataFrame(assignment_rows).to_csv(
                run_root / "panel_assignment.csv", index=False
            )

            status = ANALYSIS.main(["--run-root", str(run_root)])
            self.assertEqual(status, 0)
            agreement = pd.read_csv(
                run_root / "analysis/cross_panel_L_agreement_long.csv"
            )
            self.assertEqual(len(agreement), 18)
            self.assertEqual(set(agreement["panel_pair"]), {"1-2", "1-3", "1-4", "2-3", "2-4", "3-4"})
            self.assertEqual(set(agreement["transcript_id"]), set(transcript_ids))
            self.assertTrue(
                (run_root / "analysis/cross_panel_L_agreement_summary.csv").exists()
            )
            for panel_name in panels:
                self.assertTrue(
                    (run_root / panel_name / "predictions/common_test_L_profiles.parquet").exists()
                )

    def test_valid_compact_profile_is_used_when_raw_copy_is_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            panel_dir = Path(temporary) / "panel_01"
            prediction_dir = panel_dir / "predictions" / "model"
            prediction_dir.mkdir(parents=True)
            raw_path = prediction_dir / "predictions_main_test_best_val_loss.parquet"
            raw_path.write_bytes(b"PAR1-incomplete-copy-without-footer")
            compact_path = prediction_dir / "common_test_L_profiles.parquet"
            pd.DataFrame(
                {
                    "transcript_id": ["t1"],
                    "transcript_length": [3],
                    "L_t": [np.asarray([0.5, 1.0, 1.5], dtype=np.float32)],
                }
            ).to_parquet(compact_path, index=False)
            manifest = prediction_dir / "prediction_checkpoint_manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "best_val_loss": {
                            "split_name": "test",
                            "output_path": str(raw_path),
                        }
                    }
                ),
                encoding="utf-8",
            )

            located, source, detail = ANALYSIS._locate_panel_prediction(panel_dir)
            self.assertEqual(located, compact_path.resolve())
            self.assertEqual(source, manifest)
            self.assertIn("compact_common_test_L", detail)
            profiles, checks = ANALYSIS._extract_panel_profiles(
                panel_name="panel_01",
                run_identifier="test_run",
                prediction_path=located,
                expected_ids={"t1"},
                mean_one_tolerance=1.0e-4,
            )
            np.testing.assert_allclose(profiles["t1"]["values"], [0.5, 1.0, 1.5])
            self.assertEqual(checks.iloc[0]["number_of_prediction_dataset_rows"], 1)


if __name__ == "__main__":
    unittest.main()
