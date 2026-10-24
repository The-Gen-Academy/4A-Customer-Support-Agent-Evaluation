import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import requests

from judge_evaluation import (
    JevJudge, JudgeError, OpenAIJudge, Rubric, judge_state,
    load_failure_rows, run_evaluation, save_run,
)

RUBRIC = Rubric("Refund_Missed", "Refund request routed elsewhere", "Refund asked for; wrong route", "No refund request")


def row(label="Refund_Missed", id="one"):
    return dict(id=id, ticket_text="Refund my unused subscription.", true_category="refund_request",
                predicted_category="account_help", failure_category=label)


def response(p=0.9, status=200):
    return SimpleNamespace(status_code=status, headers={}, json=lambda: {
        "model": "jev-test", "answers": {"target_failure": {"type": "noul", "noul": p}},
        "usage": {"input_tokens": 42, "output_tokens": 2}})


class JudgeTests(unittest.TestCase):
    def test_csv_export_preamble_example_passing_and_blank_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "step3.csv"
            with path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerows([["Title"], [], ["#", "Ticket text", "True category", "Predicted category", "Failure category (from Step 2)"],
                             ["EX", "example", "other", "account_help", "Example"],
                             ["pass", "ok", "other", "other", ""], list(row().values()), []])
            self.assertEqual(load_failure_rows(path), [row()])

    def test_missing_labels_and_duplicate_ids_fail(self):
        for rows, pattern in [([row("")], "complete the Step 3"), ([row(), row()], "Duplicate")]:
            with self.subTest(pattern=pattern), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "input.csv"
                with path.open("w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(row()))
                    w.writeheader()
                    w.writerows(rows)
                with self.assertRaisesRegex(ValueError, pattern):
                    load_failure_rows(path)

    def test_judge_never_sees_labels_notes_or_true_route(self):
        source = dict(row(), reviewer_notes="SECRET LABEL", reasoning="ANSWER LEAK")
        session = Mock()
        session.post.return_value = response()
        judge = JevJudge("test", session=session)
        judge.judge(judge_state(source), RUBRIC)
        payload = session.post.call_args.kwargs["json"]
        self.assertEqual(set(payload["state"]), {"ticket_text", "predicted_category"})
        self.assertNotIn("SECRET LABEL", json.dumps(payload))
        self.assertNotIn("ANSWER LEAK", json.dumps(payload))
        self.assertEqual(payload["questions"]["target_failure"]["type"], "noul")
        self.assertEqual(session.post.call_args.args[0], "https://api.typesafe.ai/v1/systemone")

    def test_threshold_and_review_are_separate(self):
        for p, expected, review in [(0, False, False), (.3, False, False), (.49, False, True),
                                     (.5, True, True), (.7, True, False), (1, True, False)]:
            with self.subTest(p=p):
                session = Mock()
                session.post.return_value = response(p)
                verdict = JevJudge("test", session=session).judge(judge_state(row()), RUBRIC)
                self.assertEqual(verdict["judge_prediction"], expected)
                self.assertEqual(verdict["needs_review"], review)
                self.assertEqual(verdict["judge_reasoning"], "")

    def test_bad_probability_is_never_false_by_default(self):
        for p in [True, "0.8", None, float("nan"), float("inf"), -1, 2]:
            with self.subTest(p=p):
                session = Mock()
                session.post.return_value = response(p)
                with self.assertRaises(JudgeError):
                    JevJudge("test", session=session).judge(judge_state(row()), RUBRIC)

    def test_retries_and_safe_auth_error(self):
        session, sleep = Mock(), Mock()
        session.post.side_effect = [response(status=429), requests.Timeout("secret"), response()]
        JevJudge("private-key", session=session, sleep=sleep).judge(judge_state(row()), RUBRIC)
        self.assertEqual(session.post.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        session.post.side_effect = None
        session.post.return_value = response(status=401)
        with self.assertRaisesRegex(JudgeError, "HTTP 401") as err:
            JevJudge("private-key", session=session).judge(judge_state(row()), RUBRIC)
        self.assertNotIn("private-key", str(err.exception))

    def test_openai_structured_result_and_shared_rubric(self):
        runnable = Mock()
        runnable.invoke.return_value = {"parsed": SimpleNamespace(label=False, reasoning="No refund intent."),
                                       "raw": SimpleNamespace(response_metadata={"model_name": "test-openai"})}
        result = OpenAIJudge(runnable=runnable).judge(judge_state(row()), RUBRIC)
        self.assertIs(result["judge_prediction"], False)
        messages = runnable.invoke.call_args.args[0]
        self.assertIn(RUBRIC.instructions, messages[0][1])
        self.assertEqual(json.loads(messages[1][1]), judge_state(row()))
        runnable.invoke.return_value["parsed"].label = "false"
        with self.assertRaises(JudgeError):
            OpenAIJudge(runnable=runnable).judge(judge_state(row()), RUBRIC)

    def test_metrics_errors_and_saved_metadata(self):
        rows = [row(id="tp"), row(id="fn"), row("Other", "fp"), row("Other", "tn"), row("Other", "error")]
        judge = Mock(provider="jev", model="test", threshold=.5, review_low=.3, review_high=.7)
        judge.judge.side_effect = [{"judge_prediction": p} for p in [True, False, True, False]] + [JudgeError("Unavailable")]
        results, metrics = run_evaluation(rows, RUBRIC, judge)
        self.assertEqual([metrics[k] for k in ["tp", "tn", "fp", "fn", "errors"]], [1]*5)
        self.assertEqual(metrics["agreement"], .5)
        self.assertEqual(metrics["completion_rate"], .8)
        self.assertIsNone(results[-1]["match"])
        with tempfile.TemporaryDirectory() as tmp:
            folder = save_run(results, RUBRIC, judge, tmp)
            metadata = json.loads((folder / "run.json").read_text())
            self.assertEqual(metadata["threshold"], .5)
            self.assertEqual(metadata["rubric"]["target"], RUBRIC.target)
            self.assertEqual(len(metadata["dataset_sha256"]), 64)
            with (folder / "results.csv").open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 5)

    def test_no_positive_or_negative_cannot_masquerade_as_accuracy(self):
        for data in [[row()], [row("Other")], []]:
            with self.assertRaises(ValueError):
                run_evaluation(data, RUBRIC, Mock())

    def test_bundled_workbook_needs_participant_labels(self):
        with self.assertRaisesRegex(ValueError, "complete the Step 3"):
            load_failure_rows("Week 4_ AI Evals (E-Commerce Customer Support Agent).xlsx")

    def test_demo_has_both_labels_and_no_example_row(self):
        rows = load_failure_rows("examples/refund_missed_demo.csv")
        self.assertEqual(len(rows), 29)
        self.assertEqual(sum(r["failure_category"] == RUBRIC.target for r in rows), 12)
        self.assertNotIn("EX", [r["id"] for r in rows])


if __name__ == "__main__":
    unittest.main()
