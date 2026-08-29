"""Guard the manuscript generator.

Every number in the paper is produced by ``analysis.build_manuscript``, so a
formatting slip there is a wrong number in a submitted article with nothing to
catch it. These tests cover the failure modes that have actually occurred:
a p-value at the randomisation floor reported as an equality, an inequality
losing its ``<`` on the way into a table, markdown emphasis surviving into
LaTeX, and a citation key that no bibliography entry defines.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from analysis import build_manuscript as bm

REPO = Path(__file__).resolve().parents[1]
MANUSCRIPT = REPO / "manuscript"


class FormatterTests(unittest.TestCase):
    """The numeric formatters decide how every quantity reads."""

    def test_randomisation_floor_is_reported_as_an_inequality(self) -> None:
        # (exceed + 1) / (draws + 1) with no exceedances is the smallest value
        # the test can return; reporting it as "p =" claims a measurement.
        floor = 1.0 / (bm.RANDOMIZATION_DRAWS + 1.0)
        self.assertEqual(bm.prel(floor), "p<10^{-6}")
        self.assertEqual(bm.prel(0.0), "p<10^{-6}")

    def test_measurable_p_values_are_reported_as_equalities(self) -> None:
        self.assertEqual(bm.prel(0.03963896), "p=0.040")
        self.assertEqual(bm.prel(0.557), "p=0.557")
        self.assertTrue(bm.prel(0.5).startswith("p="))

    def test_small_but_resolvable_p_keeps_scientific_notation(self) -> None:
        rendered = bm.pval(7.5e-11)
        self.assertIn(r"\times10^{-11}", rendered)

    def test_degree_and_interval_precision_is_explicit(self) -> None:
        self.assertEqual(bm.deg(4.747666, 3), "4.748")
        self.assertEqual(bm.deg(4.747666, 2), "4.75")
        self.assertEqual(bm.ci([0.25365, 0.43558], 3), "0.254 to 0.436")
        self.assertEqual(bm.signed(-0.00707, 5), "-0.00707")
        self.assertEqual(bm.signed(0.01304, 5), "+0.01304")

    def test_checkpoint_label_renders_as_a_descent_step(self) -> None:
        self.assertEqual(bm.checkpoint_step("step_13_00400001"), "400001")
        self.assertEqual(bm.checkpoint_step("step_00_00000000"), "0")


class ControlTableTests(unittest.TestCase):
    """Table 2 carries the surrogate comparison, including a floored p-value."""

    @staticmethod
    def _data(identity_p: float) -> dict:
        condition = {
            "mean_improvement_deg": 0.3452,
            "bootstrap_95pct_ci_deg": [0.25365, 0.43558],
            "two_sided_randomization_p": identity_p,
        }
        surrogate = {
            "mean_improvement_deg": 0.0577,
            "bootstrap_95pct_ci_deg": [0.00477, 0.11329],
            "two_sided_randomization_p": 0.03964,
        }
        contrast = {
            "mean_deg": 0.2876,
            "two_sided_randomization_p": 1e-6,
            "surrogate_share_of_real_effect": 0.17,
        }
        return {
            "controls": {
                "conditions": {
                    "identity": condition,
                    "circular_shift": surrogate,
                    "participant_swap": surrogate,
                    "phase_randomized": surrogate,
                },
                "verdict": {
                    "recorded_vs_surrogate": {
                        "circular_shift": contrast,
                        "participant_swap": contrast,
                        "phase_randomized": contrast,
                    }
                },
            }
        }

    def test_floored_p_keeps_its_inequality_in_the_table(self) -> None:
        # A slice that trimmed one character too many once rendered this as
        # "$10^{-6}$", turning a bound into a claimed measurement.
        table = bm.control_table(self._data(1e-6))
        self.assertIn("$<10^{-6}$", table)
        self.assertNotIn("& $10^{-6}$ &", table)

    def test_resolvable_p_is_rendered_without_an_inequality(self) -> None:
        table = bm.control_table(self._data(0.02))
        recorded = [row for row in table.splitlines() if "Recorded sEMG" in row][0]
        self.assertIn("0.020", recorded)
        self.assertNotIn("<", recorded)

    def test_every_condition_appears_once(self) -> None:
        table = bm.control_table(self._data(1e-6))
        for label in bm.CONTROL_LABELS.values():
            self.assertEqual(table.count(label), 1, label)


class GeneratedManuscriptTests(unittest.TestCase):
    """Checks against the committed manuscript, when one is present."""

    @classmethod
    def setUpClass(cls) -> None:
        tex = MANUSCRIPT / "main.tex"
        if not tex.exists():
            raise unittest.SkipTest("no generated manuscript to check")
        cls.tex = tex.read_text(encoding="utf-8")

    def test_no_markdown_emphasis_survived_into_latex(self) -> None:
        # *word* renders as literal asterisks in the typeset article.
        leaked = re.findall(r"(?<![\\\w*])\*[A-Za-z][^*\n]{0,40}\*(?!\*)", self.tex)
        self.assertEqual(leaked, [], f"markdown emphasis in LaTeX: {leaked}")

    def test_every_citation_key_is_defined(self) -> None:
        bib = (MANUSCRIPT / "references.bib").read_text(encoding="utf-8")
        defined = set(re.findall(r"^@\w+\{([^,]+),", bib, re.M))
        cited: set[str] = set()
        for group in re.findall(r"\\cite\{([^}]*)\}", self.tex):
            cited.update(key.strip() for key in group.split(","))
        self.assertEqual(cited - defined, set(), "cited but not in references.bib")

    def test_no_unresolved_template_placeholders(self) -> None:
        # A surviving "{expr}" means a field was never substituted.
        for suspect in re.findall(r"\{[a-z_]+\[[^}]*\}", self.tex):
            self.fail(f"unsubstituted placeholder: {suspect}")

    def test_window_counts_agree_across_the_paper(self) -> None:
        counts = set(re.findall(r"(\d+) windows", self.tex))
        self.assertLessEqual(
            len(counts), 1, f"paper quotes conflicting window counts: {counts}"
        )

    def test_p_values_at_the_floor_are_never_equalities(self) -> None:
        self.assertNotIn(r"$p=1.00\times10^{-6}$", self.tex)
        self.assertNotIn(r"p=1.00\times10^{-6}", self.tex)


if __name__ == "__main__":
    unittest.main()
