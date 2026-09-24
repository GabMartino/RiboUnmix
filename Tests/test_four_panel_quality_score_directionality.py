"""Scientific controls for the four-panel directional score experiment."""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from analyses.analyze_four_panel_quality_score_directionality import (
    aggregate_cross_panel_metrics,
    aggregate_policy_contrasts,
)
from run_cumulative_stability import ROOT
from run_four_panel_quality_score_directionality import (
    BALANCED_DESIGN,
    REFERENCE_VARIANTS,
    TASK_COUNT,
    build_panel_collections,
    prepare,
)


class FourPanelQualityScoreTests(unittest.TestCase):
    def setUp(self):
        self.names = [f"source_{index:03d}" for index in range(1, 115)]
        self.ranking = pd.DataFrame({
            "dataset": self.names,
            "quality_rank": np.arange(1, 115, dtype=float),
            "quality_rank_score": np.linspace(200.0, 1000.0, 114),
        })

    def _write_inputs(self, root):
        mapping = {name: f"/data/{name}.parquet" for name in self.names}
        datasets = root / "datasets.yaml"
        datasets.write_text(yaml.safe_dump({"dataset_path": mapping}))
        ranking = root / "ranking.tsv"
        self.ranking.to_csv(ranking, sep="\t", index=False)
        panel_sizes = [29, 29, 28, 28]
        rows, start = [], 0
        for panel_index, size in enumerate(panel_sizes, 1):
            for name in self.names[start:start + size]:
                rows.append(dict(
                    dataset_name=name,
                    panel=f"panel_{panel_index:02d}",
                    source_identifier=name,
                ))
            start += size
        panels = root / "panels.csv"
        pd.DataFrame(rows).to_csv(panels, index=False)
        return datasets, ranking, panels

    def test_reference_matrix_has_seven_nonredundant_policies(self):
        self.assertEqual(TASK_COUNT, 28)
        self.assertEqual(len(REFERENCE_VARIANTS), 7)
        self.assertEqual(REFERENCE_VARIANTS[0], ("equal", "equal", 0))
        self.assertIn(("best_first", "score_p5", 5), REFERENCE_VARIANTS)
        self.assertIn(("worst_first", "score_p5", 5), REFERENCE_VARIANTS)

    def test_balanced_and_strata_memberships_are_exhaustive(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            datasets, ranking_path, panels = self._write_inputs(root)
            mapping = yaml.safe_load(datasets.read_text())["dataset_path"]
            balanced, assignment, _, _ = build_panel_collections(
                "balanced", panels, mapping, self.ranking
            )
            strata, strata_assignment, _, _ = build_panel_collections(
                "quality_strata", panels, mapping, self.ranking
            )
            self.assertEqual([row["N"] for row in balanced], [29, 29, 28, 28])
            self.assertEqual([row["N"] for row in strata], [29, 29, 28, 28])
            self.assertEqual(assignment.dataset_id.nunique(), 114)
            self.assertEqual(strata_assignment.dataset_id.nunique(), 114)
            self.assertEqual(strata[0]["datasets"], self.names[:29])
            self.assertEqual(strata[-1]["datasets"], self.names[-28:])

    def test_preparation_fixes_cohort_and_orients_score_weights(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            datasets, ranking, panels = self._write_inputs(root)
            output = root / "experiment"
            args = argparse.Namespace(
                output_root=output,
                panel_design="balanced",
                seed=42,
                config=ROOT / "config/config_ribounmix_multidataset.yaml",
                datasets=datasets,
                ranking=ranking,
                panels=panels,
                task_index=None,
                prepare_only=True,
                dry_run=False,
                max_pair_rows_per_forward=None,
                max_padded_codon_tokens_per_forward=None,
                require_resume_checkpoint=False,
            )
            panel_ids = [f"panel_{index:02d}" for index in range(1, 5)]
            split = dict(
                panel_train_eligible_ids={panel: ["train"] for panel in panel_ids},
                panel_validation_ids={panel: ["validation"] for panel in panel_ids},
                common_test_ids=["test"],
                cohort_limitation="complete case",
            )

            def reliability(**kwargs):
                return {
                    "panel_name": kwargs["panel_name"],
                    "datasets": {name: {"tau": 1.0, "median": 1.0}
                                 for name in kwargs["panel_datasets"]},
                }

            with patch(
                "run_four_panel_quality_score_directionality.build_fixed_cumulative_split",
                return_value=split,
            ) as split_builder, patch(
                "run_four_panel_quality_score_directionality.fit_panel_reliability_manifest",
                side_effect=reliability,
            ) as reference_builder:
                plan = prepare(args)
                same_plan = prepare(args)

            self.assertEqual(plan, same_plan)
            self.assertEqual(plan["experiment_design"], BALANCED_DESIGN)
            self.assertEqual(len(plan["tasks"]), 28)
            self.assertEqual(split_builder.call_count, 1)
            self.assertEqual(reference_builder.call_count, 1)
            self.assertEqual(
                {(task["panel_id"], task["arm"]) for task in plan["tasks"]},
                {(panel, "equal") for panel in panel_ids}
                | {(panel, f"{orientation}_{policy}")
                   for panel in panel_ids
                   for orientation, policy, _ in REFERENCE_VARIANTS[1:]},
            )
            for fold in plan["source_folds"].values():
                self.assertEqual(fold["train_ids"], ["train"])
                self.assertEqual(fold["validation_ids"], ["validation"])
                self.assertEqual(fold["test_ids"], ["test"])

            weights = pd.read_csv(output / "reference_weights.csv")
            panel = weights[weights.panel_id.eq("panel_01")]
            equal = panel[panel.arm.eq("equal")].sort_values("quality_rank_score")
            best = panel[panel.arm.eq("best_first_score_p3")].sort_values("quality_rank_score")
            worst = panel[panel.arm.eq("worst_first_score_p3")].sort_values("quality_rank_score")
            np.testing.assert_allclose(equal.pi, np.full(len(equal), 1 / len(equal)))
            self.assertTrue(np.all(np.diff(best.pi.to_numpy()) < 0))
            self.assertTrue(np.all(np.diff(worst.pi.to_numpy()) > 0))
            self.assertAlmostEqual(best.pi.sum(), 1.0)
            self.assertAlmostEqual(worst.pi.sum(), 1.0)

            task = next(
                task for task in plan["tasks"]
                if task["panel_id"] == "panel_01" and task["arm"] == "worst_first_score_p5"
            )
            cfg = yaml.safe_load(Path(task["config_path"]).read_text())
            self.assertEqual(cfg["experiment"]["dataset"], task["datasets"])
            self.assertEqual(
                cfg["orchestrator"]["reference_weight_orientation"], "worst_first"
            )
            self.assertEqual(cfg["orchestrator"]["reference_weight_power"], 5)
            self.assertTrue((output / "design_report.html").is_file())

    def test_primary_analysis_averages_pairs_within_transcript_before_bootstrap(self):
        rows = []
        values = {
            ("equal", "tx1"): (0.8, 0.4),
            ("equal", "tx2"): (0.6, 0.6),
            ("best_first_score_p1", "tx1"): (0.7, 0.5),
            ("best_first_score_p1", "tx2"): (0.5, 0.7),
        }
        for (arm, transcript), (pcc_mean, rmse_mean) in values.items():
            for pair, offset in (("p1__p2", -0.1), ("p1__p3", 0.1)):
                rows.append(dict(
                    arm=arm,
                    transcript_id=transcript,
                    pair=pair,
                    PCC=pcc_mean + offset,
                    RMSE=rmse_mean + offset,
                    reason="ok",
                ))
        aggregate, summary = aggregate_cross_panel_metrics(
            pd.DataFrame(rows), replicates=200, seed=17
        )
        self.assertTrue(aggregate.complete_six_pair_record.all())
        self.assertTrue(aggregate.n_valid_pairs.eq(2).all())
        equal = summary.set_index("arm").loc["equal"]
        self.assertAlmostEqual(equal.mean_PCC, 0.7)
        self.assertAlmostEqual(equal.mean_RMSE, 0.5)

        metrics, effects = aggregate_policy_contrasts(
            aggregate, replicates=200, seed=17
        )
        self.assertEqual(len(metrics), 2)
        effect = effects.iloc[0]
        self.assertAlmostEqual(effect.mean_delta_PCC, -0.1)
        self.assertAlmostEqual(effect.mean_delta_RMSE, 0.1)
        self.assertEqual(effect.fraction_PCC_improved, 0.0)
        self.assertEqual(effect.fraction_RMSE_improved, 0.0)


if __name__ == "__main__":
    unittest.main()
