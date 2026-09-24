from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.stats import nbinom

from analyses.analyze_synthetic_observation_layers import (
    ESTABLISHED_CONDITION_ORDER,
    describe,
    independent_nb2_equality_probability,
    load_config,
    pearson_with_reason,
)


ROOT = Path(__file__).resolve().parents[1]


def test_config_preserves_established_condition_and_depth_order() -> None:
    config = load_config(ROOT / "config" / "synthetic_observation_layer_audit.yaml")
    assert tuple(item["key"] for item in config["conditions"]) == ESTABLISHED_CONDITION_ORDER
    assert tuple(float(item["value"]) for item in config["depths"]) == (0.25, 2.0, 20.0)


def test_pearson_undefined_cases_are_nan_with_reason() -> None:
    value, reason = pearson_with_reason(np.ones(5), np.arange(5, dtype=float))
    assert math.isnan(value)
    assert reason == "constant_left"

    value, reason = pearson_with_reason(np.ones(5), np.ones(5))
    assert math.isnan(value)
    assert reason == "constant_both"

    value, reason = pearson_with_reason(np.array([1.0]), np.array([1.0]))
    assert math.isnan(value)
    assert reason == "too_few_positions"


def test_pearson_matches_exact_linear_relationship() -> None:
    value, reason = pearson_with_reason(
        np.array([1.0, 2.0, 4.0, 8.0]),
        np.array([3.0, 5.0, 9.0, 17.0]),
    )
    assert reason == ""
    assert math.isclose(value, 1.0, rel_tol=0.0, abs_tol=1e-14)


def test_independent_nb2_equality_probability_matches_explicit_pmf_sum() -> None:
    alpha = 0.1
    theta = 1.0 / alpha
    means = np.array([0.0, 0.25, 2.0, 20.0])
    calculated = independent_nb2_equality_probability(
        means, dispersion_alpha=alpha
    )
    expected = []
    for mean in means:
        probability = theta / (theta + mean)
        # At these means, k<=2000 leaves a negligible omitted tail.
        k = np.arange(2001)
        pmf = nbinom.pmf(k, theta, probability)
        expected.append(float(np.square(pmf).sum()))
    np.testing.assert_allclose(calculated, expected, rtol=1e-12, atol=1e-14)


def test_describe_retains_undefined_count_without_zero_imputation() -> None:
    result = describe(np.array([0.2, np.nan, 0.8]), total=3)
    assert result["n_valid_pcc"] == 2
    assert result["n_undefined_pcc"] == 1
    assert math.isclose(result["median"], 0.5)
    assert result["minimum"] == 0.2
    assert result["maximum"] == 0.8
