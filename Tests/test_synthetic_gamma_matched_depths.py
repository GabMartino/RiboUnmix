"""Small numerical and alignment tests for the main-figure gamma audit."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyses.analyze_synthetic_gamma_matched_depths import (
    DEPTH_ORDER, iter_transcripts, joint_log_gamma_gauge, summarize, _pcc,
)


def test_gauge_and_mask_have_hand_computed_values():
    raw = np.array([[0., 2., 0., 2.], [2., 0., 2., 0.]])
    expected = np.array([[-1., 1., -1., 1.], [1., -1., 1., -1.]])
    np.testing.assert_allclose(joint_log_gamma_gauge(raw), expected)
    shifted = raw + np.array([[3.], [7.]]) + np.array([[-2., 8., 0., 9.]])
    np.testing.assert_allclose(joint_log_gamma_gauge(shifted), expected)
    np.testing.assert_allclose(joint_log_gamma_gauge(raw, position_mask=[1, 1, 0, 0]), expected[:, :2])
    assert np.isnan(_pcc(np.ones(4), np.arange(4)))


def test_interleaved_transcript_rows_are_not_split(tmp_path: Path):
    rows = [
        {"transcript_id": tid, "dataset_id": dataset, "mask": [True]*3, "log_gamma": [0., 1., 2.]}
        for tid, dataset in [("a", 1), ("b", 0), ("b", 1), ("a", 0)]
    ]
    path = tmp_path / "interleaved.parquet"
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)
    run = SimpleNamespace(prediction_path=path, run_id="test", n_datasets=2,
                          dataset_id_to_name={0: "f0", 1: "f1"})
    groups = dict(iter_transcripts(run, {"a", "b"}))
    assert set(groups) == {"a", "b"}
    assert all(set(rows) == {"f0", "f1"} for rows in groups.values())


def test_bootstrap_excludes_undefined_on_a_fixed_cohort():
    rows = [{"run_id": f"{depth}_{n}", "depth": depth, "n_datasets": n,
             "transcript_id": tid,
             "mean_dataset_pcc": np.nan if tid == "c" and n == 2 else value,
             "joint_log_gamma_rmse": 1.0 - value,
             "joint_calibration_slope": value + 0.1}
            for depth in DEPTH_ORDER for n in range(2, 11)
            for tid, value in [("a", .8), ("b", .9), ("c", .2)]]
    source = pd.DataFrame(rows)
    a, b = summarize(source, 100, 123), summarize(source, 100, 123)
    pd.testing.assert_frame_equal(a, b)
    assert a.n_transcripts.eq(2).all() and a.excluded_transcripts.eq(1).all()
    np.testing.assert_allclose(a["median"], .85)
    np.testing.assert_allclose(a["pcc_median"], .85)
    np.testing.assert_allclose(a["log_rmse_median"], .15)
    np.testing.assert_allclose(a["calibration_slope_median"], .95)
    # Identical values across N and common resamples give identical intervals.
    assert a.groupby("depth").bootstrap_ci_low.nunique().eq(1).all()
    assert a.groupby("depth").bootstrap_ci_high.nunique().eq(1).all()
    assert a.groupby("depth").log_rmse_bootstrap_ci_low.nunique().eq(1).all()
    assert a.groupby("depth").log_rmse_bootstrap_ci_high.nunique().eq(1).all()


if __name__ == "__main__":
    # The analysis environment need not install pytest just to run these tests.
    def temporary_parquet_test():
        with tempfile.TemporaryDirectory() as directory:
            test_interleaved_transcript_rows_are_not_split(Path(directory))

    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_gauge_and_mask_have_hand_computed_values,
        temporary_parquet_test,
        test_bootstrap_excludes_undefined_on_a_fixed_cohort,
    ))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
