# Build an LLM judge, then try Jev

The classifier chooses a support route. The judge checks a different question:
**Does this classifier mistake belong to the failure category we chose?**

Start by learning how an LLM judge works. Then try Jev or add confidence-based escalation.
All judges see only the ticket and predicted route. We check their answers against your
Step 3 labels afterward. They never see the true route, failure label, or reviewer notes.

For example, if the customer asks for a refund but the classifier chooses `order_status`,
the judge should mark `Refund_Missed` as TRUE. A broken-product complaint without a
refund request should be FALSE. FALSE means a different kind of mistake, not a correct route.

## 1. Build and understand the LLM judge

1. Open notebook **Section 11** from a full clone/download of this repository. Start with
   the demo to learn the steps. It runs independently of the classifier and needs no
   Google or LangSmith credentials.
2. In **11a–11b**, load the examples and write the rubric: the exact target label, its
   definition, and clear TRUE/FALSE rules. Inspect the input the judge will see.
3. In **11c**, define the answer format and create the OpenAI judge. Enter an OpenAI key
   privately when prompted, or set `OPENAI_API_KEY` locally. Try one ticket and check
   whether its TRUE/FALSE answer and short reason follow your rule.
4. In **11d–11e**, run all the examples and review the results. Agreement measures matches
   with your labels. Precision checks its TRUE answers; recall checks how many target
   failures it found. Read a few disagreements, not just the overall numbers.
5. For your evaluation, complete Step 3 in **your own copy** of the workbook. Export it
   as CSV, or download the workbook as XLSX. Keep both your chosen category and other
   failures. Change `INPUT_PATH` and the rubric, then rerun from 11a.

## 2. Optionally try Jev on its own

In **12a–12b**, use the same rows and rubric with Jev. Enter a TypeSafe key privately
when prompted, or set `TYPESAFE_API_KEY` locally. If you only want Jev, run the shared
setup in 11a–11b, then go straight to Section 12. No OpenAI key is needed for that path.

| Type | Use it for | In this project |
| --- | --- | --- |
| **Noul** | A yes/no decision | Sections 12a–12b: does this mistake match our chosen failure category? |
| **Choice** | Pick one of a fixed set of options | Section 12c: the same TRUE/FALSE task, with native confidence for escalation. It can also support multiple failure categories. |
| Score | Rate something on ordered, descriptive levels | An extension: rate the severity of a mistake. You would need severity labels to evaluate it. |

Try one ticket in **12a**, then run all examples and review the results in **12b**.
Jev's Noul returns the probability of TRUE. We use `p >= 0.5` for TRUE and flag
`0.3 < p < 0.7` for review. These are starting settings, not validated thresholds.
Flagged rows still count in the metrics. The probability is not a quality score,
and Jev does not produce a written reason.

## 3. Add confidence-based escalation (optional)

Keep the **Section 11 LLM-only run as your comparison baseline**, then run **12c**.
This option needs both `TYPESAFE_API_KEY` and `OPENAI_API_KEY`, entered privately. It
creates a separate fallback judge using the baseline's `LLM_MODEL` and saves its run
under `jev_llm`; it does not overwrite the baseline or the standalone Noul run.

**Jev call → validate output → confidence gate → LLM fallback → final verdict → logging.**

Jev answers a specific question about **one chosen failure category**, just like the
LLM judge. For `Refund_Missed`: **Does this ticket ask for a refund while the classifier
routes it somewhere other than `refund_request`?** The actual question includes your
category definition and both decision rules:

```python
{
    "type": "choice",
    "instructions": rubric.instructions,  # target category + definition
    "criteria": {"true": rubric.positive, "false": rubric.negative},
}
```

1. **Jev call:** ask a binary Choice with `true` and `false` options, using the same
   ticket, predicted route, and rubric as the LLM-only baseline.
2. **Validate output:** check the Choice answer, probability distribution, and finite
   confidence in `[0, 1]`. Invalid output or a Jev API failure triggers fallback.
3. **Confidence gate:** keep a valid Jev answer when its native `confidence` is **≥0.70**.
   A value **<0.70** triggers fallback; exactly 0.70 stays with Jev.
4. **LLM fallback:** judge the original ticket and route with the same rubric. The
   fallback does not receive Jev's answer or your human reference label.
5. **Final verdict:** use the accepted Jev answer or the validated LLM answer. If
   fallback fails, set `status="unresolved"` and leave the final verdict and match empty.
6. **Logging:** save each stage's result, confidence, model, usage, latency, and errors,
   plus `judge_used`, `escalated`, and `escalation_reason` in the run's CSV and metadata.

Choice supports this same binary task and uses the existing TRUE/FALSE labels.
Its native confidence summarizes the probability distribution; it is not the probability
of TRUE or a guarantee of correctness. Noul has no native `confidence` field, so its
0.3–0.7 review band in 12a–12b is a separate exercise. The 70% gate is a starting policy
to evaluate on held-out examples. See TypeSafe's [Choice](https://docs.typesafe.ai/primitives/choice)
and [confidence](https://docs.typesafe.ai/confidence) documentation.

Review the hybrid's disagreements, escalations, and unresolved cases. Check how often
Jev handled a row, how often fallback ran, and whether the final verdict matched your
label. An escalation is not itself an error; a successful fallback produces a completed
verdict. Compare measured latency and returned token usage; no dollar savings are
claimed without known pricing and comparable measured runs.

## 4. Reflect and save your work

**13a** checks every available run's dataset fingerprint and rubric against each other
and the current rows and rubric. It shows completion coverage, errors, and unresolved
cases separately. Agreement, precision, and recall use completed verdicts only. If a
run is incomplete, resolve those failures before treating the table as a full comparison.
`errors` counts all rows without a final verdict; `unresolved` is the hybrid subset,
so do not add those two counts together.
Compare false alarms, missed target failures, individual decisions, and latency against
the LLM-only baseline. You can also finish with just one judge.

To improve a rule, use a small practice subset, then check separate untouched examples.
With very few rows, describe the result as exploratory; do not claim it proves a better judge.

Each run saves `results.csv` and `run.json` in a new folder under `judge_runs/`.
Import each CSV into a **new tab** in your workbook (File → Import → Upload → Insert new sheet).
Keep the run folders so the model, rubric, threshold, and dataset fingerprint remain available.
Use Step 5 as the results area and Step 6 for your comparison notes. The code exports files;
it does not write directly into Google Sheets or fill the existing Step 5 template.

Follow **13b** to submit your labeled workbook and result CSVs. In Step 6, explain one
decision you reviewed and one rule you would improve. If you ran both judges, also say
which you would use and why. For the hybrid, include the 70% gate, escalation rate,
unresolved count, and one example that used fallback.

## Try the included demo first

Start with `examples/refund_missed_demo.csv` and its matching rubric in Section 11.
These example labels are provided for practice. For your evaluation, use the labels you reviewed and assigned in Step 3.

The workbook's `LLM-J Dataset` tab alone has only 12 positive examples. Running only
those would hide false positives. The loader requires both positive and negative examples,
rejects missing labels and duplicate IDs, and skips the `EX` teaching row.

The workbook and the original notebook use different datasets and routing policies.
For example, the workbook routes an explicit non-delivery refund request to `refund_request`,
while the notebook's generated seed examples use `order_status`. Keep each evaluation's
data, labels, and policy together. Do not compare the demo's judge agreement with the
notebook's 92%/97% classifier accuracy: those measure different things.

## Setup and checks

```bash
pip install -r requirements-judge.txt
python -m unittest discover -s tests -v
```

The Jev adapters post Noul or Choice questions to `https://api.typesafe.ai/v1/systemone`.
They record the resolved model, validate output, and retry transient failures.
Errors and unresolved cases are shown separately from completed judgments; neither becomes FALSE.
The default Jev model is pinned to `jev-1.14.0`; change it deliberately when comparing versions.

Official references: [Noul](https://docs.typesafe.ai/primitives/noul),
[Choice](https://docs.typesafe.ai/primitives/choice),
[Confidence](https://docs.typesafe.ai/confidence),
[Score](https://docs.typesafe.ai/primitives/score),
[API](https://docs.typesafe.ai/api/system-one).
