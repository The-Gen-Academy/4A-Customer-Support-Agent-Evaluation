"""Exercise confidence routing and durable audit logs without provider calls."""

import copy
import csv
import json
import tempfile
import unittest
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import Mock

import requests

from judge_evaluation import (
    HybridJudge, JevChoiceJudge, JudgeError, OpenAIJudge, Rubric,
    compare_runs, judge_state, run_evaluation, save_run,
)


RUBRIC = Rubric("Refund_Missed", "Refund request routed elsewhere",
                "Refund asked for; wrong route", "No refund request")
STATE = {"ticket_text": "Refund my unused subscription.", "predicted_category": "account_help"}


def row(label="Refund_Missed", identifier="one"):
    return dict(id=identifier, ticket_text=STATE["ticket_text"], true_category="refund_request",
                predicted_category=STATE["predicted_category"], failure_category=label)


def choice_response(confidence=.9, prediction=True, status=200):
    selected = (1 + confidence) / 2
    true_probability = selected if prediction else 1 - selected
    data = {
        "model": "jev-resolved", "answers": {"target_failure": {
            "type": "choice", "choice": "true" if prediction else "false",
            "confidence": confidence,
            "probabilities": {"true": true_probability, "false": 1 - true_probability},
        }}, "usage": {"input_tokens": 42, "output_tokens": 6},
    }
    return SimpleNamespace(status_code=status, headers={}, json=lambda: data)


def jev_result(confidence=.9, prediction=True):
    return dict(judge_prediction=prediction, probability_true=(1 + confidence) / 2 if prediction
                else (1 - confidence) / 2, judge_reasoning="", confidence=confidence,
                model="jev-resolved", usage={"input_tokens": 42, "output_tokens": 6})


def llm_result(prediction=False):
    return dict(judge_prediction=prediction, judge_reasoning="The rubric does not match." if not prediction
                else "The rubric matches.", probability_true=None, needs_review=None,
                model="llm-resolved", usage={"prompt_tokens": 30, "completion_tokens": 10})


def judges(confidence=.9, prediction=True):
    jev = Mock(provider="jev_choice", model="jev-requested")
    jev.judge.return_value = jev_result(confidence, prediction)
    fallback = Mock(provider="openai", model="llm-requested")
    fallback.judge.return_value = llm_result()
    return jev, fallback


class JevChoiceTests(unittest.TestCase):
    def test_choice_request_and_native_response_have_no_label_leak_or_rationale(self):
        session = Mock()
        session.post.return_value = choice_response(.7, False)
        source = dict(row(), reviewer_notes="SECRET HUMAN NOTES", reasoning="SECRET HUMAN REASON")
        result = JevChoiceJudge("private-key", session=session).judge(judge_state(source), RUBRIC)
        call = session.post.call_args
        self.assertEqual(call.args[0], "https://api.typesafe.ai/v1/systemone")
        self.assertFalse(call.kwargs["allow_redirects"])
        payload = call.kwargs["json"]
        self.assertEqual(payload["state"], STATE)
        self.assertEqual(payload["model"], "jev-1.14.0")
        question = payload["questions"]["target_failure"]
        self.assertEqual(question["type"], "choice")
        self.assertEqual(question["criteria"], {"true": RUBRIC.positive, "false": RUBRIC.negative})
        self.assertEqual(question["instructions"], RUBRIC.instructions)
        self.assertNotIn("SECRET HUMAN", json.dumps(payload))
        self.assertIs(result["judge_prediction"], False)
        self.assertAlmostEqual(result["probability_true"], .15)
        self.assertEqual(result["confidence"], .7)
        self.assertEqual(result["judge_reasoning"], "")

    def test_malformed_choice_responses_are_rejected(self):
        mutations = [
            ("missing answer", lambda d: d["answers"].clear()),
            ("wrong type", lambda d: d["answers"]["target_failure"].update(type="noul")),
            ("unknown choice", lambda d: d["answers"]["target_failure"].update(choice="maybe")),
            ("missing choice probability", lambda d: d["answers"]["target_failure"]["probabilities"].pop("false")),
            ("extra probability", lambda d: d["answers"]["target_failure"]["probabilities"].update(maybe=0)),
            ("not normalized", lambda d: d["answers"]["target_failure"].update(probabilities={"true": .9, "false": .9})),
            ("choice is not mode", lambda d: d["answers"]["target_failure"].update(choice="false")),
            ("inconsistent native confidence", lambda d: d["answers"]["target_failure"].update(confidence=.7)),
            ("missing confidence", lambda d: d["answers"]["target_failure"].pop("confidence")),
            ("missing model", lambda d: d.pop("model")),
            ("malformed model", lambda d: d.update(model=None)),
        ]
        for name, mutate in mutations:
            with self.subTest(name=name):
                response = choice_response()
                mutate(response.json())
                session = Mock()
                session.post.return_value = response
                with self.assertRaises(JudgeError):
                    JevChoiceJudge("test", session=session).judge(STATE, RUBRIC)
        for value in [True, "0.9", None, float("nan"), float("inf"), -.1, 1.1]:
            for field in ["confidence", "probability"]:
                with self.subTest(field=field, value=value):
                    response = choice_response()
                    answer = response.json()["answers"]["target_failure"]
                    if field == "confidence":
                        answer["confidence"] = value
                    else:
                        answer["probabilities"]["true"] = value
                    session = Mock()
                    session.post.return_value = response
                    with self.assertRaises(JudgeError):
                        JevChoiceJudge("test", session=session).judge(STATE, RUBRIC)

    def test_invalid_json_and_transport_failure_are_safe(self):
        for response in [SimpleNamespace(status_code=200, headers={}, json=Mock(side_effect=ValueError("secret"))),
                         SimpleNamespace(status_code=200, headers={}, json=lambda: None)]:
            with self.subTest(response=response):
                session = Mock()
                session.post.return_value = response
                with self.assertRaises(JudgeError) as err:
                    JevChoiceJudge("private-key", session=session).judge(STATE, RUBRIC)
                self.assertNotIn("private-key", str(err.exception))
                self.assertNotIn("secret", str(err.exception))
        session, sleep = Mock(), Mock()
        session.post.side_effect = requests.Timeout("secret response body")
        with self.assertRaises(JudgeError) as err:
            JevChoiceJudge("private-key", session=session, sleep=sleep).judge(STATE, RUBRIC)
        self.assertEqual(session.post.call_count, 3)
        self.assertNotIn("secret", str(err.exception))

    def test_retryable_statuses_and_auth_error(self):
        session, sleep = Mock(), Mock()
        session.post.side_effect = [choice_response(status=429), choice_response(status=529), choice_response()]
        result = JevChoiceJudge("test", session=session, sleep=sleep).judge(STATE, RUBRIC)
        self.assertIs(result["judge_prediction"], True)
        self.assertEqual(session.post.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        session = Mock()
        session.post.return_value = choice_response(status=401)
        with self.assertRaises(JudgeError):
            JevChoiceJudge("private-key", session=session, sleep=sleep).judge(STATE, RUBRIC)
        self.assertEqual(session.post.call_count, 1)


class HybridTests(unittest.TestCase):
    def test_confidence_boundary_and_confident_false(self):
        for confidence, accepted in [(.699, False), (.70, True), (.701, True)]:
            for prediction in [True, False]:
                with self.subTest(confidence=confidence, prediction=prediction):
                    jev, fallback = judges(confidence, prediction)
                    result = HybridJudge(jev, fallback).judge(STATE, RUBRIC)
                    self.assertEqual(result["escalated"], not accepted)
                    self.assertEqual(result["judge_used"], "jev" if accepted else "llm")
                    self.assertEqual(result["status"], "ok")
                    self.assertEqual(result["jev_prediction"], prediction)
                    self.assertEqual(result["jev_confidence"], confidence)
                    if accepted:
                        fallback.judge.assert_not_called()
                        self.assertIs(result["judge_prediction"], prediction)
                        self.assertEqual(result["judge_reasoning"], "")
                        self.assertEqual(result["escalation_reason"], "")
                    else:
                        fallback.judge.assert_called_once_with(STATE, RUBRIC)
                        self.assertIs(result["judge_prediction"], False)
                        self.assertEqual(result["escalation_reason"], "low_confidence")

    def test_probability_is_not_used_as_native_confidence(self):
        jev, fallback = judges(.6)
        self.assertEqual(jev.judge.return_value["probability_true"], .8)
        result = HybridJudge(jev, fallback).judge(STATE, RUBRIC)
        self.assertEqual(result["judge_used"], "llm")
        fallback.judge.assert_called_once()

    def test_invalid_jev_verdicts_and_confidences_fall_back(self):
        invalid = [None, {}, dict(jev_result(), judge_prediction="true"),
                   dict(jev_result(), judge_prediction=1)]
        invalid.extend(dict(jev_result(), confidence=c) for c in
                       [None, True, "0.9", float("nan"), float("inf"), -.1, 1.1])
        for value in invalid:
            with self.subTest(value=value):
                jev, fallback = judges()
                jev.judge.return_value = value
                result = HybridJudge(jev, fallback).judge(STATE, RUBRIC)
                self.assertEqual(result["status"], "ok")
                self.assertEqual(result["judge_used"], "llm")
                self.assertEqual(result["escalation_reason"], "invalid_jev_output")
                fallback.judge.assert_called_once_with(STATE, RUBRIC)

    def test_jev_errors_fall_back_and_unexpected_exception_text_is_sanitized(self):
        for error in [JudgeError("TypeSafe unavailable"), RuntimeError("SECRET_KEY_IN_EXCEPTION")]:
            with self.subTest(error=type(error).__name__):
                jev, fallback = judges()
                jev.judge.side_effect = error
                result = HybridJudge(jev, fallback).judge(STATE, RUBRIC)
                self.assertEqual(result["judge_used"], "llm")
                self.assertEqual(result["escalation_reason"], "jev_error")
                self.assertTrue(result["jev_error"])
                self.assertNotIn("SECRET_KEY_IN_EXCEPTION", json.dumps(result))
                fallback.judge.assert_called_once()

    def test_fallback_receives_original_label_free_state_without_jev_answer(self):
        source = dict(row(), reviewer_notes="SECRET HUMAN LABEL")
        state = judge_state(source)
        snapshot = copy.deepcopy(state)
        jev, fallback = judges(.2)
        HybridJudge(jev, fallback).judge(state, RUBRIC)
        jev.judge.assert_called_once_with(snapshot, RUBRIC)
        fallback.judge.assert_called_once_with(snapshot, RUBRIC)
        self.assertEqual(state, snapshot)
        self.assertEqual(set(fallback.judge.call_args.args[0]), {"ticket_text", "predicted_category"})

    def test_real_adapters_share_rubric_instructions_criteria_and_evidence(self):
        session, runnable = Mock(), Mock()
        session.post.return_value = choice_response(.4)
        runnable.invoke.return_value = {
            "parsed": SimpleNamespace(label=False, reasoning="No refund intent."),
            "raw": SimpleNamespace(response_metadata={"model_name": "llm-resolved"}),
        }
        hybrid = HybridJudge(JevChoiceJudge("test", session=session), OpenAIJudge(runnable=runnable))
        result = hybrid.judge(STATE, RUBRIC)
        payload = session.post.call_args.kwargs["json"]
        question = payload["questions"]["target_failure"]
        messages = runnable.invoke.call_args.args[0]
        self.assertEqual(messages[0][1],
                         f"{question['instructions']}\nTRUE: {question['criteria']['true']}"
                         f"\nFALSE: {question['criteria']['false']}")
        self.assertEqual(json.loads(messages[1][1]), payload["state"])
        self.assertEqual(result["judge_used"], "llm")
        self.assertIs(result["judge_prediction"], False)

    def test_malformed_wire_response_is_logged_as_invalid_then_escalated(self):
        session = Mock()
        response = choice_response()
        response.json()["answers"]["target_failure"].pop("confidence")
        session.post.return_value = response
        _, fallback = judges()
        result = HybridJudge(JevChoiceJudge("test", session=session), fallback).judge(STATE, RUBRIC)
        self.assertEqual(result["escalation_reason"], "invalid_jev_output")
        self.assertEqual(result["judge_used"], "llm")
        self.assertTrue(result["jev_error"])
        fallback.judge.assert_called_once_with(STATE, RUBRIC)

    def test_failed_or_malformed_fallback_is_unresolved(self):
        cases = [JudgeError("LLM unavailable"), RuntimeError("SECRET_FALLBACK"),
                 None, {}, dict(llm_result(), judge_prediction="false"),
                 dict(llm_result(), judge_prediction=0)]
        for failure in cases:
            with self.subTest(failure=failure):
                jev, fallback = judges(.3)
                if isinstance(failure, Exception):
                    fallback.judge.side_effect = failure
                else:
                    fallback.judge.return_value = failure
                result = HybridJudge(jev, fallback).judge(STATE, RUBRIC)
                self.assertEqual(result["status"], "unresolved")
                self.assertEqual(result["judge_used"], "unresolved")
                self.assertIsNone(result["judge_prediction"])
                self.assertTrue(result["escalated"])
                self.assertTrue(result["llm_error"])
                self.assertEqual(result["jev_confidence"], .3)
                self.assertEqual(result["jev_model"], "jev-resolved")
                self.assertEqual(result["jev_usage"], {"input_tokens": 42, "output_tokens": 6})
                self.assertNotIn("SECRET_FALLBACK", json.dumps(result))
                self.assertGreaterEqual(result["jev_latency_seconds"], 0)
                self.assertGreaterEqual(result["llm_latency_seconds"], 0)

    def test_invalid_thresholds_fail_before_provider_calls(self):
        for threshold in [True, "0.7", None, float("nan"), -.1, 1.1]:
            with self.subTest(threshold=threshold):
                jev, fallback = judges()
                with self.assertRaises(ValueError):
                    HybridJudge(jev, fallback, confidence_threshold=threshold)
                jev.judge.assert_not_called()
                fallback.judge.assert_not_called()

    def test_evaluation_export_preserves_unresolved_row_and_stage_costs(self):
        source = [row(identifier="accepted"), row("Other", "fallback"), row("Other", "unresolved")]
        jev, fallback = judges()
        jev.judge.side_effect = [jev_result(.9), jev_result(.3), jev_result(.2)]
        fallback.judge.side_effect = [llm_result(False), JudgeError("LLM unavailable")]
        judge = HybridJudge(jev, fallback)
        results, metrics = run_evaluation(source, RUBRIC, judge)
        self.assertEqual(metrics["completed"], 2)
        self.assertEqual(metrics["errors"], 1)
        self.assertEqual(metrics["unresolved"], 1)
        self.assertEqual(metrics["agreement"], 1)
        self.assertAlmostEqual(metrics["completion_rate"], 2 / 3)
        self.assertEqual(metrics["escalations"], 2)
        self.assertAlmostEqual(metrics["escalation_rate"], 2 / 3)
        self.assertEqual(metrics["jev_accepted"], 1)
        self.assertGreaterEqual(metrics["latency_seconds"], 0)
        self.assertGreaterEqual(metrics["mean_latency_seconds"], 0)
        self.assertEqual(results[-1]["status"], "unresolved")
        self.assertIsNone(results[-1]["judge_prediction"])
        self.assertIsNone(results[-1]["match"])
        self.assertEqual(results[-1]["escalation_reason"], "low_confidence")
        with tempfile.TemporaryDirectory() as tmp:
            folder = save_run(results, RUBRIC, judge, tmp)
            metadata = json.loads((folder / "run.json").read_text())
            self.assertEqual(metadata["confidence_threshold"], .7)
            self.assertEqual(metadata["jev_requested_model"], "jev-requested")
            self.assertEqual(metadata["llm_requested_model"], "llm-requested")
            self.assertIn("jev-resolved", json.dumps(metadata["resolved_models"]))
            self.assertIn("llm-resolved", json.dumps(metadata["resolved_models"]))
            with (folder / "results.csv").open() as handle:
                saved = list(csv.DictReader(handle))
            self.assertEqual(len(saved), 3)
            self.assertEqual(saved[-1]["status"], "unresolved")
            self.assertEqual(saved[-1]["judge_prediction"], "")
            self.assertEqual(saved[-1]["jev_confidence"], "0.2")
            self.assertEqual(json.loads(saved[-1]["jev_usage"])["input_tokens"], 42)
            self.assertTrue(saved[-1]["llm_error"])
            self.assertEqual(json.loads(saved[1]["llm_usage"])["prompt_tokens"], 30)
            combined_usage = json.loads(saved[1]["usage"])
            self.assertIn("42", json.dumps(combined_usage))
            self.assertIn("30", json.dumps(combined_usage))

    def test_baseline_is_independent_and_comparison_rejects_different_inputs(self):
        source = [row(), row("Other", "two")]
        snapshot = copy.deepcopy(source)
        runnable = Mock()
        runnable.invoke.side_effect = [
            {"parsed": SimpleNamespace(label=label, reasoning="Baseline reason."),
             "raw": SimpleNamespace(response_metadata={"model_name": "baseline-resolved"})}
            for label in [True, False]
        ]
        baseline = OpenAIJudge(model="baseline-requested", runnable=runnable)
        baseline_results, baseline_metrics = run_evaluation(source, RUBRIC, baseline)
        baseline_snapshot = copy.deepcopy(baseline_results)
        self.assertEqual(runnable.invoke.call_count, len(source))
        for call, input_row in zip(runnable.invoke.call_args_list, source):
            self.assertEqual(json.loads(call.args[0][1][1]), judge_state(input_row))
        with tempfile.TemporaryDirectory() as tmp:
            baseline_folder = save_run(baseline_results, RUBRIC, baseline, tmp)
            saved_before = {p.name: p.read_bytes() for p in baseline_folder.iterdir()}
            metadata = json.loads((baseline_folder / "run.json").read_text())
            jev, fallback = judges(.9)
            jev.judge.side_effect = [jev_result(.9, True), jev_result(.9, False)]
            hybrid = HybridJudge(jev, fallback)
            hybrid_results, _ = run_evaluation(source, RUBRIC, hybrid)
            hybrid_folder = save_run(hybrid_results, RUBRIC, hybrid, tmp)
            self.assertNotEqual(hybrid_folder, baseline_folder)
            self.assertEqual(baseline_results, baseline_snapshot)
            self.assertEqual(source, snapshot)
            self.assertEqual({p.name: p.read_bytes() for p in baseline_folder.iterdir()}, saved_before)
            hybrid_metadata = json.loads((hybrid_folder / "run.json").read_text())
            runs = {"llm_only": metadata, "jev_llm": hybrid_metadata}
            comparison = compare_runs(runs, source, RUBRIC)
            self.assertEqual(comparison["llm_only"], baseline_metrics)
            self.assertEqual(comparison["jev_llm"], hybrid_metadata["metrics"])
            changed = copy.deepcopy(source)
            changed[0]["ticket_text"] = "Different ticket"
            with self.assertRaises(ValueError):
                compare_runs(runs, changed, RUBRIC)
            changed_rubric = Rubric(RUBRIC.target, "Changed definition", RUBRIC.positive, RUBRIC.negative)
            with self.assertRaises(ValueError):
                compare_runs(runs, source, changed_rubric)
            self.assertEqual(metadata["rubric"], asdict(RUBRIC))
            # An errored run is still eligible for comparison, with its coverage visible.
            error_metadata = copy.deepcopy(metadata)
            error_metadata["metrics"].update(completed=1, errors=1, completion_rate=.5)
            comparison = compare_runs({"llm_only": error_metadata}, source, RUBRIC)
            self.assertEqual(comparison["llm_only"]["errors"], 1)
            self.assertEqual(comparison["llm_only"]["completion_rate"], .5)


if __name__ == "__main__":
    unittest.main()
