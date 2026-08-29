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
    """Table 2 reports paired t-tests, which are not bounded below."""

    @staticmethod
    def _data(identity_p: float) -> dict:
        def entry(mean: float, p: float, t: float) -> dict:
            return {
                "mean_improvement_deg": mean,
                "mean_deg": mean,
                "bootstrap_95pct_ci_deg": [mean - 0.09, mean + 0.09],
                "two_sided_randomization_p": max(p, 1e-6),
                "surrogate_share_of_real_effect": 0.17,
                "paired_t": {"t": t, "df": 89, "p_two_sided": p},
            }

        surrogate = entry(0.0577, 0.0412, 2.07)
        contrast = entry(0.2876, 1.68e-7, 5.68)
        return {
            "controls": {
                "conditions": {
                    "identity": entry(0.3452, identity_p, 7.39),
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

    def test_small_p_is_not_censored_to_the_randomisation_floor(self) -> None:
        # Routing a t-test p-value through the randomisation formatter once
        # reported 7.5e-11 as "p < 10^-6", discarding five orders of magnitude.
        table = bm.control_table(self._data(7.5e-11))
        self.assertIn(r"10^{-11}", table)
        self.assertNotIn("<10^{-6}", table)
        self.assertNotIn(r"$10^{-6}$", table)

    def test_margin_column_carries_the_margin(self) -> None:
        # The column reports the paired margin; its p-value is given in the text,
        # where it does not push the table past the text block.
        table = bm.control_table(self._data(7.5e-11))
        self.assertIn("+0.288", table)

    def test_moderate_p_renders_as_a_decimal(self) -> None:
        table = bm.control_table(self._data(0.02))
        recorded = [row for row in table.splitlines() if "Recorded sEMG" in row][0]
        self.assertIn("0.020", recorded)
        self.assertNotIn("times", recorded)

    def test_every_condition_appears_once(self) -> None:
        table = bm.control_table(self._data(7.5e-11))
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

    def test_every_label_sits_inside_a_reference_command(self) -> None:
        # A command written into a non-raw Python string loses its backslash
        # and the letter after it, so "\\ref{sec:future}" prints as
        # "efsec:future" in the body text.
        for match in re.finditer(r"(?<!\\label)(?<!\\ref)\{?(sec|tab|fig|eq):[a-z_]+",
                                 self.tex):
            head = self.tex[max(0, match.start() - 12):match.start()]
            self.assertRegex(
                head,
                r"\\(?:ref|label|eqref|autoref|cref)\{$",
                f"label leaked into the text: {self.tex[match.start() - 20:match.end()]!r}",
            )

    def test_no_command_lost_its_leading_backslash(self) -> None:
        # The same accident applied to any command starting with a b f n r t v.
        tails = ("ef", "ho", "egin", "extbf", "imes", "rac", "space", "ewline",
                 "oindent", "extit", "ootnote", "abular", "oprule")
        pattern = r"(?<![A-Za-z\\])(" + "|".join(tails) + r")(?=[{~])"
        for match in re.finditer(pattern, self.tex):
            before = self.tex[max(0, match.start() - 40):match.start()]
            self.assertNotRegex(
                before,
                r"[\n~]\s*$",
                f"command lost its backslash: {self.tex[match.start() - 30:match.end() + 15]!r}",
            )

    def test_p_values_at_the_floor_are_never_equalities(self) -> None:
        self.assertNotIn(r"$p=1.00\times10^{-6}$", self.tex)
        self.assertNotIn(r"p=1.00\times10^{-6}", self.tex)


if __name__ == "__main__":
    unittest.main()
