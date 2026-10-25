"""The evaluation scorer and the offline detector-baseline run. Live mode is never called here."""

import unittest
from unittest import mock

from auditor.rules import load_catalog
from eval import run as evaluation


def violation(rule_id="SEC-001", category="security", file="a.py", line=5):
    return {"rule_id": rule_id, "category": category, "file": file, "line": line}


class ScoreFindingsTests(unittest.TestCase):
    def test_exact_match_is_a_true_positive(self):
        score = evaluation.score_findings([violation()], [violation()])
        self.assertEqual(
            (score.true_positives, score.false_positives, score.false_negatives), (1, [], [])
        )

    def test_unlabelled_finding_is_a_false_positive_and_missed_label_a_false_negative(self):
        score = evaluation.score_findings([violation()], [violation(file="b.py")])
        self.assertEqual(score.true_positives, 0)
        self.assertEqual(len(score.false_positives), 1)
        self.assertEqual(len(score.false_negatives), 1)

    def test_line_tolerance_is_inclusive(self):
        limit = evaluation.LINE_TOLERANCE
        near = evaluation.score_findings([violation(line=5)], [violation(line=5 + limit)])
        far = evaluation.score_findings([violation(line=5)], [violation(line=6 + limit)])
        self.assertEqual(near.true_positives, 1)
        self.assertEqual(far.true_positives, 0)

    def test_null_lines_match_only_each_other(self):
        both = evaluation.score_findings([violation(line=None)], [violation(line=None)])
        label_only = evaluation.score_findings([violation(line=None)], [violation(line=1)])
        found_only = evaluation.score_findings([violation(line=1)], [violation(line=None)])
        self.assertEqual(both.true_positives, 1)
        self.assertEqual(label_only.true_positives, 0)
        self.assertEqual(found_only.true_positives, 0)

    def test_rule_family_decides_not_the_exact_id(self):
        same_family = evaluation.score_findings([violation("SEC-001")], [violation("SEC-099")])
        other_family = evaluation.score_findings(
            [violation("SEC-001")], [violation("CFG-001", category="security")]
        )
        self.assertEqual(same_family.true_positives, 1)
        self.assertEqual(other_family.true_positives, 0)

    def test_category_must_match(self):
        score = evaluation.score_findings([violation()], [violation(category="configuration")])
        self.assertEqual(score.true_positives, 0)

    def test_duplicate_finding_counts_as_false_positive(self):
        score = evaluation.score_findings([violation()], [violation(), violation()])
        self.assertEqual(score.true_positives, 1)
        self.assertEqual(len(score.false_positives), 1)

    def test_precision_recall_and_f1(self):
        score = evaluation.score_findings(
            [violation(line=1), violation(line=10), violation(line=20)],
            [violation(line=1), violation(line=10), violation(file="x.py")],
        )
        self.assertAlmostEqual(score.precision, 2 / 3)
        self.assertAlmostEqual(score.recall, 2 / 3)
        self.assertAlmostEqual(score.f1, 2 / 3)

    def test_empty_labels_and_findings_score_perfectly(self):
        score = evaluation.score_findings([], [])
        self.assertEqual((score.precision, score.recall), (1.0, 1.0))

    def test_findings_on_a_healthy_case_give_zero_precision(self):
        score = evaluation.score_findings([], [violation()])
        self.assertEqual(score.precision, 0.0)
        self.assertEqual(score.recall, 1.0)

    def test_merge_adds_counts_and_lists(self):
        total = evaluation.Score()
        total.merge(evaluation.score_findings([violation()], [violation()]))
        total.merge(evaluation.score_findings([violation()], []))
        self.assertEqual(total.true_positives, 1)
        self.assertEqual(len(total.false_negatives), 1)


class CaseFileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = evaluation.load_cases()

    def test_each_type_has_at_least_three_cases(self):
        kinds = [case.category for case in self.cases]
        for kind in "ABC":
            self.assertGreaterEqual(kinds.count(kind), 3, kind)

    def test_expected_rule_ids_and_categories_are_known(self):
        catalog = {rule.id: rule.category for rule in load_catalog().rules}
        for case in self.cases:
            for label in case.expected:
                self.assertEqual(catalog.get(label["rule_id"]), label["category"], case.name)

    def test_expected_files_exist_or_are_null(self):
        for case in self.cases:
            for label in case.expected:
                if label["file"] is not None:
                    self.assertIn(label["file"], case.files, case.name)

    def test_healthy_cases_have_no_labels(self):
        for case in self.cases:
            if case.category == "C":
                self.assertEqual(case.expected, [], case.name)

    def test_defective_cases_have_labels(self):
        for case in self.cases:
            if case.category != "C":
                self.assertTrue(case.expected, case.name)

    def test_unknown_case_name_is_an_error(self):
        with self.assertRaises(evaluation.EvalError):
            evaluation.load_cases(["no-such-case"])


class OfflineRunTests(unittest.TestCase):
    def test_healthy_cases_produce_no_detector_findings(self):
        for case in evaluation.load_cases():
            if case.category == "C":
                self.assertEqual(evaluation.detector_findings(case), [], case.name)

    def test_defective_cases_produce_no_unlabelled_detector_findings(self):
        for case in evaluation.load_cases():
            score = evaluation.score_findings(case.expected, evaluation.detector_findings(case))
            self.assertEqual(score.false_positives, [], case.name)

    def test_offline_run_labels_the_baseline_and_never_builds_a_client(self):
        args = evaluation.build_parser().parse_args([])
        with mock.patch.object(evaluation, "LLMClient") as client:
            output = evaluation.run(args)
        client.assert_not_called()
        self.assertIn("detector baseline (no LLM); not agent accuracy", output)
        self.assertIn("overall", output)

    def test_case_option_limits_the_run(self):
        args = evaluation.build_parser().parse_args(["--case", "b5-dockerfile-latest"])
        output = evaluation.run(args)
        self.assertIn("b5-dockerfile-latest", output)
        self.assertNotIn("a1-hardcoded-db-password", output)

    def test_main_exits_zero(self):
        with mock.patch("builtins.print"):
            self.assertEqual(evaluation.main([]), 0)

    def test_alias_restores_names_that_gitignore_would_hide(self):
        case = next(c for c in evaluation.load_cases() if c.name == "a5-committed-env")
        self.assertIn(".env", case.files)


if __name__ == "__main__":
    unittest.main()
