"""Exercise actual PDF/PNG rendering without an external TeX installation."""
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from Utils.publication_plot_style import LATEX_PAPER_RC, latex_paper_style, publication_rc
from Utils.real_panel_convergence import plot_panel_balance
from analyses.analyze_real_panel_convergence import _save_pair_agreement_figure


class PublicationPlotStyleTests(unittest.TestCase):
    def test_missing_tex_uses_bundled_fonts(self):
        with patch.dict("os.environ", RIBOUNMIX_PLOT_TEX="auto"), \
             patch("Utils.publication_plot_style.shutil.which", return_value=None), \
             self.assertWarnsRegex(RuntimeWarning, "Missing latex, dvipng"):
            style = publication_rc()
        self.assertFalse(style["text.usetex"])
        self.assertEqual(style["font.serif"], ["DejaVu Serif"])
        self.assertEqual(style["mathtext.fontset"], "cm")
        self.assertTrue(LATEX_PAPER_RC["text.usetex"])

    def test_missing_dvipng_also_falls_back_for_png_export(self):
        with patch.dict("os.environ", RIBOUNMIX_PLOT_TEX="auto"), \
             patch("Utils.publication_plot_style.shutil.which",
                   side_effect=lambda name: "/usr/bin/latex" if name == "latex" else None), \
             self.assertWarnsRegex(RuntimeWarning, "Missing dvipng"):
            self.assertFalse(publication_rc()["text.usetex"])

    def test_available_tex_preserves_canonical_style(self):
        for mode in ("auto", "1"):
            with self.subTest(mode=mode), patch.dict("os.environ", RIBOUNMIX_PLOT_TEX=mode), \
                 patch("Utils.publication_plot_style.shutil.which", return_value="/available"):
                self.assertEqual(publication_rc(), LATEX_PAPER_RC)

    def test_explicit_off_does_not_probe_or_use_tex(self):
        with patch.dict("os.environ", RIBOUNMIX_PLOT_TEX="0"), \
             patch("Utils.publication_plot_style.shutil.which") as which:
            self.assertFalse(publication_rc()["text.usetex"])
            which.assert_not_called()

    def test_explicit_tex_fails_with_actionable_message(self):
        with patch.dict("os.environ", RIBOUNMIX_PLOT_TEX="1"), \
             patch("Utils.publication_plot_style.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "Set RIBOUNMIX_PLOT_TEX=0"):
                publication_rc()

    def test_invalid_mode_is_not_silently_ignored(self):
        with patch.dict("os.environ", RIBOUNMIX_PLOT_TEX="typo"):
            with self.assertRaisesRegex(ValueError, "RIBOUNMIX_PLOT_TEX must be"):
                publication_rc()

    def test_context_restored_and_non_rendering_errors_not_retried(self):
        calls = []

        @latex_paper_style
        def broken_plot():
            calls.append(1)
            self.assertFalse(matplotlib.rcParams["text.usetex"])
            raise RuntimeError("Invalid scientific input")

        with matplotlib.rc_context({"text.usetex": True, "font.size": 7}), \
             patch.dict("os.environ", RIBOUNMIX_PLOT_TEX="0"):
            with self.assertRaisesRegex(RuntimeError, "Invalid scientific input"):
                broken_plot()
            self.assertTrue(matplotlib.rcParams["text.usetex"])
            self.assertEqual(matplotlib.rcParams["font.size"], 7)
        self.assertEqual(calls, [1])

    def test_setup_and_analysis_pdf_png_render_without_tex(self):
        assignment = pd.DataFrame([
            dict(panel=f"panel_{p:02}", median_read_density=1 + i,
                 median_positive_codon_coverage=.2 + .1*i, median_replica_PCC=.3 + .08*i)
            for p in range(1, 5) for i in range(5)
        ])
        agreement = pd.DataFrame([
            dict(panel_pair=f"P{a:02}–P{b:02}", PCC=.25 + .08*i)
            for a, b in itertools.combinations(range(1, 5), 2) for i in range(8)
        ])
        before = assignment.copy(deep=True)
        figures_before = plt.get_fignums()
        with tempfile.TemporaryDirectory() as directory, \
             patch.dict("os.environ", RIBOUNMIX_PLOT_TEX="auto"), \
             patch("Utils.publication_plot_style.shutil.which", return_value=None), \
             patch("matplotlib.texmanager.TexManager.make_dvi",
                   side_effect=AssertionError("External TeX must not be invoked")):
            out = Path(directory)
            with self.assertWarns(RuntimeWarning):
                plot_panel_balance(assignment, output_directory=out)
            with self.assertWarns(RuntimeWarning):
                _save_pair_agreement_figure(agreement, out)
            for stem in ("dataset_panel_balance", "cross_panel_L_PCC_by_pair", "cross_panel_L_PCC_pooled"):
                self.assertTrue((out/f"{stem}.pdf").read_bytes().startswith(b"%PDF-"))
                self.assertTrue((out/f"{stem}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        pd.testing.assert_frame_equal(assignment, before)
        self.assertEqual(plt.get_fignums(), figures_before)


if __name__ == "__main__":
    unittest.main()
