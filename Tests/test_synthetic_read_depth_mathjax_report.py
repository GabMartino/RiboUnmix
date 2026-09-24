"""Notation/rendering checks and the balanced-reference identity documented in HTML."""
from pathlib import Path
from string import Template
import unittest

import numpy as np
import pandas as pd

from analyses.report_synthetic_read_depth_audit import table

TEMPLATE = Path(__file__).resolve().parents[1] / "analyses/templates/synthetic_read_depth_audit.html"


class TestMathjaxReport(unittest.TestCase):
    def test_template_has_balanced_math_and_no_unresolved_placeholders(self):
        source = TEMPLATE.read_text()
        template = Template(source)
        document = template.substitute({name: "27" for name in template.get_identifiers()})
        self.assertNotIn("${", document)
        body = document.split("<body>", 1)[1]
        self.assertEqual(body.count(r"\("), body.count(r"\)"))
        self.assertEqual(body.count(r"\["), body.count(r"\]"))
        self.assertGreater(body.count(r"\["), 15)
        self.assertIn("mathjax@3.2.2", document)
        self.assertIn("defaultPageReady", document)
        self.assertNotIn("\x08", document)  # A non-raw Python \bar string would contain backspace.

    def test_table_has_tex_headers_but_escapes_html_data(self):
        result = table(pd.DataFrame({"depth": ["0p25_per_codon"],
                                     "rmse_Kg_trim10_mean": [.1], "note": ["<script>bad</script>"]}))
        self.assertIn(r"\(\bar E(L,H)\)", result)
        self.assertIn("0.25", result)
        self.assertIn("&lt;script&gt;", result)
        self.assertNotIn("<script>", result)

    def test_balanced_depth_weights_preserve_known_two_way_reference(self):
        # Two distinct bias families, repeated at three depths. This tests the
        # stated algebra, not a fit to fabricated experimental observations.
        bias = np.array([[1., 3., 1., 1.], [1., 1., 5.5, 1.]])
        log_bias = np.log(np.repeat(bias, 3, axis=0))
        equal = np.ones(6)/6
        ranked = np.tile([1., 2., 3.], 2)/12

        def reference(weights):
            center = weights @ log_bias
            means = log_bias.mean(axis=1)
            corrected = log_bias-center-means[:, None]+weights @ means
            return center, np.exp(corrected)

        equal_center, equal_target = reference(equal)
        ranked_center, ranked_target = reference(ranked)
        np.testing.assert_allclose(equal_center, ranked_center, atol=1e-14)
        np.testing.assert_allclose(equal_target, ranked_target, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
