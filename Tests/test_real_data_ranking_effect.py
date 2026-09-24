"""Numerical/contract tests; artificial fixtures are never manuscript data."""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from analyses import create_real_data_ranking_effect_figure as f


def bootstrap_reference(equal, ranked, statistic, draws=5000):
    rng = np.random.default_rng(20260910)
    values = []
    reduce = (lambda a: np.median(a, axis=0)) if statistic == "C" else (
        lambda a: np.array([a[:, i:i+3].mean(axis=0).mean() for i in range(0, 15, 3)]))
    for _ in range(draws):
        draw = rng.integers(0, len(equal), size=len(equal))
        values.append(reduce(ranked[draw])-reduce(equal[draw]))
    return np.quantile(values, [.025, .975], axis=0, method="linear")


def fixture_summaries():
    c = pd.DataFrame(dict(comparison_id=f.PAIR_ORDER, estimate=np.linspace(-.06, .04, 6)))
    c["ci_lower"], c["ci_upper"] = c.estimate-.03, c.estimate+.035
    d = pd.DataFrame(dict(N=f.N_VALUES, estimate=[-.06, .01, .05, -.01, .015]))
    d["ci_lower"], d["ci_upper"] = d.estimate-.015, d.estimate+.03
    pairs = pd.DataFrame([dict(N=n, comparison_id=p, estimate=d.estimate.iloc[i]+offset)
        for i, n in enumerate(f.N_VALUES) for p, offset in zip(f.PAIR_IDS, [-.01, 0, .01])])
    return c, d, pairs


class RankingContractTests(unittest.TestCase):
    def test_complete_table_R_and_checksum(self):
        ranks, q, meta = f.load_ranking(f.RANKING)
        self.assertEqual(meta["component_count"], 10)
        self.assertEqual(meta["R"], 115)
        self.assertEqual(meta["sha256"], f.MANUSCRIPT_RANKING_SHA256)
        chosen = sorted(ranks, key=ranks.get)[:3]
        np.testing.assert_allclose([q[d] for d in chosen], [(116-ranks[d])/115 for d in chosen])
        self.assertNotAlmostEqual(q[chosen[-1]], (3-ranks[chosen[-1]]+1)/3)

    def test_six_components_are_not_substituted(self):
        with self.assertRaisesRegex(ValueError, "10-component"):
            f.load_ranking(f.ROOT/"Datasets/data/HEK_riboseq_profile_quality_rank.tsv")

    def test_missing_duplicate_and_incomplete_ranking_fail(self):
        original = pd.read_csv(f.RANKING, sep="\t")
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/"fixture.tsv"
            original.iloc[:3].to_csv(path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "frozen manuscript"):
                f.load_ranking(path)
            bad = original.copy()
            bad.loc[0, "quality_rank"] = np.nan
            bad.to_csv(path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "global ranks"):
                f.load_ranking(path, required_sha256=None)
            duplicate = pd.concat([original, original.iloc[[0]]])
            duplicate.to_csv(path, sep="\t", index=False)
            with self.assertRaisesRegex(ValueError, "unique"):
                f.load_ranking(path, required_sha256=None)

    def test_actual_design_is_source_disjoint_and_cumulative_is_rejected(self):
        runs, _ = f.read_runs(f.uniform.DEFAULT_PANEL_ROOT, "C")
        self.assertEqual([r.N for r in runs], [29,29,28,28])
        self.assertEqual(sum(len(r.sources) for r in runs), 85)
        runs, _ = f.read_runs(f.uniform.DEFAULT_STABILITY_ROOT, "D")
        self.assertEqual(len(runs), 30)
        self.assertEqual({r.N for r in runs}, {2,5,10,20,40})
        with self.assertRaisesRegex(ValueError, "Cumulative"):
            f.read_runs(f.ROOT/"results/real_exp8_L_stability_quality_rank_10components/cumulative_qrank10components_p1.0_seed42", "D")

    def test_numerical_configuration_changes_block_matching(self):
        base = dict(name="equal", model=dict(gamma_centering=dict(reference=dict(weighting="equal", quality_rank_power=0))),
                    optim=dict(lr=.001), loss=dict(sample_reduction="transcript_balanced"), experiment=dict(seed=42))
        ranked = deepcopy(base)
        ranked["name"] = "ranked"
        ranked["model"]["gamma_centering"]["reference"].update(weighting="quality_rank", quality_rank_power=1)
        self.assertTrue(all(d["permitted"] for d in f.compare_configs(base, ranked)))
        ranked["optim"]["lr"] = .0001
        with self.assertRaisesRegex(ValueError, "optim.lr"):
            f.compare_configs(base, ranked)
        ranked = deepcopy(base)
        ranked["experiment"]["seed"] = 43
        with self.assertRaisesRegex(ValueError, "experiment.seed"):
            f.compare_configs(base, ranked)

    def test_weight_reconstruction_uses_global_q_and_rejects_posthoc_equal_pi(self):
        runs, _ = f.read_runs(f.uniform.DEFAULT_PANEL_ROOT, "C")
        run = runs[0]
        _, q, _ = f.load_ranking(f.RANKING)
        cfg = f.config_for(run)
        cfg["model"]["gamma_centering"]["reference"].update(weighting="quality_rank", quality_rank_power=1)
        raw = np.array([q[d] for d in run.datasets])
        declaration = dict(fixed_gamma_reference=dict(weighting="quality_rank", pi=dict(zip(run.datasets,raw/raw.sum()))))
        with patch.object(f, "declaration", return_value=declaration):
            got, pi = f.require_weights(run, "ranked", cfg, q)
            np.testing.assert_array_equal(got, raw)
            self.assertTrue(np.all(pi > 0))
            self.assertAlmostEqual(pi.sum(), 1)
            declaration["fixed_gamma_reference"]["pi"] = dict.fromkeys(run.datasets, 1/run.N)
            with self.assertRaises(AssertionError):
                f.require_weights(run, "ranked", cfg, q)

    def test_historical_yaml_is_not_training_code_provenance(self):
        runs, _ = f.read_runs(f.uniform.DEFAULT_PANEL_ROOT, "C")
        with self.assertRaisesRegex(ValueError, "NOT VERIFIED"):
            f.execution_evidence(runs[0])

    def test_checkpoint_raw_q_buffer_and_normalized_runtime_pi(self):
        import torch
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            (directory/"predictions").mkdir()
            (directory/"checkpoints").mkdir()
            run = f.Run("D", "fixture", 2, directory, directory, ("d1","d2"), ("s1","s2"), ("t1",), "fixture")
            raw = directory/"predictions/raw.parquet"
            pq.write_table(pa.table({"transcript_id":["t1"]}),raw)
            checkpoint = directory/"checkpoints/val-loss-fixture.ckpt"
            f.write_json(directory/"predictions/prediction_checkpoint_manifest.json", dict(best_val_loss=dict(
                split_name="test", transcript_count=1, transcript_id_hash=f.transcript_id_hash(["t1"]),
                output_path=str(raw), checkpoint_path=str(checkpoint))))
            gamma = dict(centering_mode="fixed_reference", weighting="quality_rank", quality_rank_power=1,
                         reference_dataset_names=["d1","d2"], reference_raw_weights=[1.,.25], reference_pi=[.8,.2])
            f.write_json(directory/"predictions/gamma_reference_manifest.json",gamma)
            state = {"model._extra_state":dict(gamma_reference_dataset_names=["d1","d2"],
                         gamma_centering_weighting="quality_rank", gamma_centering_mode="fixed_reference",
                         gamma_centering_quality_rank_power=1),
                     "model.gamma_reference_weights":torch.tensor([1.,.25]),
                     "model.gamma_reference_dataset_ids":torch.tensor([0,1])}
            saved = dict(state_dict=state,callbacks={"ModelCheckpoint":dict(monitor="val_loss",best_model_path=str(checkpoint))})
            torch.save(saved,checkpoint)
            self.assertEqual(f.prediction_evidence(run,"ranked",{"d1":1.,"d2":.25})["dataset_ids"],{"d1":0,"d2":1})
            state["model.gamma_reference_weights"]=torch.tensor([.8,.2])
            torch.save(saved,checkpoint)
            with self.assertRaises(AssertionError):
                f.prediction_evidence(run,"ranked",{"d1":1.,"d2":.25})


class PairedStatisticsTests(unittest.TestCase):
    def test_C_summaries_keep_pair_order_and_difference_of_medians(self):
        a, b = [.1,.3,.9], [.9,.1,.2]
        frame = pd.DataFrame([dict(experiment="C", N=np.nan, comparison_id=pair, training_seed=42,
             transcript_id=f"t{i}", PCC_equal=a[i], PCC_ranked=b[i], included=True)
             for pair in reversed(f.PAIR_ORDER) for i in range(3)])
        summary, pairs, _ = f.summarize_records(frame, "C")
        self.assertEqual(tuple(summary.comparison_id), f.PAIR_ORDER)
        np.testing.assert_allclose(summary.estimate, -.1)
        self.assertTrue(pairs.empty)

    def test_C_difference_of_medians_and_all_5000_shared_draws(self):
        equal = np.repeat(np.array([.1, .3, .9])[:, None], 6, axis=1)
        ranked = np.repeat(np.array([.9, .1, .2])[:, None], 6, axis=1)
        a, b, low, high, _ = f.bootstrap_effects(equal, ranked, "C")
        np.testing.assert_allclose(b-a, -.1)
        self.assertNotAlmostEqual(float(np.median(ranked[:, 0]-equal[:, 0])), float(b[0]-a[0]))
        np.testing.assert_allclose([low, high], bootstrap_reference(equal, ranked, "C"))

    def test_D_three_pair_means_and_shared_resampling_over_N(self):
        equal = np.arange(12*15).reshape(12,15)/240
        ranked = equal + np.sin(np.arange(12*15).reshape(12,15))*.09
        a, b, low, high, digest = f.bootstrap_effects(equal, ranked, "D")
        np.testing.assert_allclose(b-a, (ranked-equal).mean(axis=0).reshape(5,3).mean(axis=1))
        np.testing.assert_allclose([low, high], bootstrap_reference(equal, ranked, "D"))
        self.assertEqual(digest, f.bootstrap_effects(equal, ranked, "D")[4])
        ranked[0, 3] = np.nan
        with self.assertRaisesRegex(ValueError, "common complete cohort"):
            f.bootstrap_effects(equal, ranked, "D")

    def test_summary_round_trip_and_duplicate_rejection(self):
        rows = [dict(experiment="D", N=n, comparison_id=p, training_seed=42, transcript_id=t,
                     PCC_equal=.2+i*.05, PCC_ranked=.18+i*.05+j*.003, included=True)
                for j, (n,p) in enumerate(itertools.product(f.N_VALUES, f.PAIR_IDS))
                for i,t in enumerate(["t0", "t1", "t2", "t3"])]
        frame = pd.DataFrame(rows)
        summary, pairs, _ = f.summarize_records(frame, "D")
        np.testing.assert_allclose(summary.estimate, pairs.groupby("N").estimate.mean())
        self.assertEqual(set(summary.n_transcripts), {4})
        with self.assertRaisesRegex(ValueError, "Duplicated"):
            f.summarize_records(pd.concat([frame,frame.iloc[[0]]]), "D")
        frame.loc[0, "training_seed"] = 43
        with self.assertRaisesRegex(ValueError, "mixed training seeds"):
            f.summarize_records(frame, "D")

    def test_D_common_cohort_removes_failure_from_every_N_and_pair(self):
        runs, _ = f.read_runs(f.uniform.DEFAULT_STABILITY_ROOT, "D")
        runs = [f.Run(**{**vars(r), "test_ids": ("good", "bad")}) for r in runs]
        evidence = {r.collection: {p: dict(raw_path=r.collection, policy=p) for p in ("equal", "ranked")} for r in runs}
        values = {t: f.uniform.Profile(np.array([.5, 1., 1.5]), 3) for t in ("good", "bad")}
        def profiles(meta, ids):
            invalid = {"bad": "nonfinite_profile"} if meta["raw_path"] == "N005_pair02_A" and meta["policy"] == "ranked" else {}
            return values, invalid
        with patch.object(f, "read_profiles", side_effect=profiles):
            frame = f.comparison_records("D", runs, runs, evidence)
        self.assertEqual(len(frame), 30)
        self.assertEqual(set(frame[frame.included].transcript_id), {"good"})
        self.assertEqual(frame.included.sum(), 15)
        self.assertEqual((frame.exclusion_reason == "excluded_by_common_cohort_across_all_N_and_pairs").sum(), 14)


class ProfileAndRenderingTests(unittest.TestCase):
    def test_actual_array_reader_preserves_amplitudes_and_excludes_invalid_masks(self):
        with tempfile.TemporaryDirectory() as temp:
            temp = Path(temp)
            seq = temp/"seq.parquet"
            pq.write_table(pa.Table.from_pylist([dict(transcript_id=t, codons=["AAA"]*3) for t in ("good", "mask", "length", "mean", "dupe")]), seq)
            rows = [dict(transcript_id="good", length=3, mask=[True]*3+[False], L_bio=[.2, .7, 2.1, 0.]),
                    dict(transcript_id="good", length=3, mask=[True]*3, L_bio=[.2, .7, 2.1]),
                    dict(transcript_id="mask", length=3, mask=[True,False,True,True], L_bio=[.2, .7, 2.1, .3]),
                    dict(transcript_id="length", length=2, mask=[True]*2, L_bio=[.5,1.5]),
                    dict(transcript_id="mean", length=3, mask=[True]*3, L_bio=[1.,2.,3.]),
                    dict(transcript_id="dupe", length=3, mask=[True]*3, L_bio=[.5,1.,1.5]),
                    dict(transcript_id="dupe", length=3, mask=[True]*3, L_bio=[.2,.7,2.1])]
            raw = temp/"raw.parquet"
            pq.write_table(pa.Table.from_pylist(rows), raw)
            profiles, bad = f.read_profiles(dict(raw_path=str(raw), sequence_path=str(seq)), ["good","mask","length","mean","dupe"])
            np.testing.assert_array_equal(profiles["good"].values, [.2,.7,2.1])
            self.assertEqual(set(bad), {"mask", "length", "mean", "dupe"})

    def test_constant_nonfinite_and_length_mismatch_are_not_zero_PCC(self):
        regular = f.uniform.Profile(np.array([.5,1.,1.5]),3)
        for other in (f.uniform.Profile(np.ones(3),3), f.uniform.Profile(np.ones(4),4),
                      f.uniform.Profile(np.array([.5, np.nan,1.5]),3)):
            result = f.uniform.profile_agreement(regular, other)
            self.assertNotEqual(result["status"], "valid")
            self.assertTrue(np.isnan(result["PCC"]))

    def test_vector_coordinates_intervals_titles_and_log_axis(self):
        c, d, pairs = fixture_summaries()
        geometry = dict(width_inches=7.2666667, height_inches=2.7666667)
        fig = f.build_figure(c, d, pairs, geometry, no_tex=True)
        f.verify_plot(fig, c, d, pairs)
        self.assertEqual(fig.axes[0].get_title(loc="left"), "C  Ranking effect on reproducibility")
        self.assertEqual(len(fig.axes[0].lines), 1)
        with tempfile.TemporaryDirectory() as temp:
            fig.text(.5,.5,"TEST FIXTURE — NOT EXPERIMENTAL RESULTS", ha="center", fontsize=8, alpha=.5)
            fig.savefig(Path(temp)/"test.pdf")
            fig.savefig(Path(temp)/"test.png", dpi=600)
            self.assertGreater((Path(temp)/"test.pdf").stat().st_size, 1000)
        plt.close(fig)

    def test_blocked_audit_does_not_create_dummy_effects_or_alter_manifests(self):
        inputs = [f.uniform.DEFAULT_PANEL_ROOT/"panel_manifest.json", f.uniform.DEFAULT_STABILITY_ROOT/"experiment_manifest.json", f.RANKING]
        before = {p:f.sha256(p) for p in inputs}
        with tempfile.TemporaryDirectory() as temp:
            with patch.object(f, "choose_ranked_root", return_value=(None, [])):
                code = f.main(["--output-dir", temp])
            self.assertEqual(code, 2)
            self.assertFalse((Path(temp)/"real_data_ranking_effect.pdf").exists())
            self.assertFalse((Path(temp)/"real_data_ranking_effect.png").exists())
            source = Path(temp)/"real_data_ranking_effect_source"
            self.assertTrue(pd.read_csv(source/"panel_c_summary.csv").empty)
            self.assertTrue(pd.read_csv(source/"panel_d_paired_per_transcript.csv").empty)
            self.assertEqual(len(pd.read_csv(source/"missing_runs.csv")), 34)
            self.assertEqual(json.loads((source/"figure_manifest.json").read_text())["missing_ranked_matches"], {"C":4,"D":30})
        self.assertEqual(before, {p:f.sha256(p) for p in inputs})


if __name__ == "__main__":
    unittest.main()
