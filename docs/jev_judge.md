# Compare an LLM judge with Jev

The classifier chooses a support route. The judge checks a different question:
**Does this classifier mistake belong to the failure category we chose?**

Both judges see only the ticket and predicted route. We compare their answers with
your Step 3 labels afterward. They never see the true route, failure label, or reviewer notes.

## Which Jev question type?

| Type | Use it for | In this project |
| --- | --- | --- |
| **Noul** | A yes/no decision | Use this: does this mistake match our chosen failure category? |
| Choice | Pick one of several named options | An extension: choose among all your failure categories, including `other`. |
| Score | Rate something on ordered, descriptive levels | An extension: rate the severity of a mistake. You would need severity labels to evaluate it. |

Jev's Noul returns the probability of TRUE. We use `p >= 0.5` for the comparison,
and flag `0.3 < p < 0.7` for review. These are starting settings, not validated thresholds.
Review-flagged rows still count in the metrics. The probability is not a quality score,
and Jev does not produce a written reason. OpenAI returns a boolean and a short reason.

## Participant exercise

1. Complete Step 3 in **your own copy** of the workbook. Keep both the chosen failure
   category and other failures. Export that tab as CSV, or download the workbook as XLSX.
2. Open notebook **Section 11** from a full clone/download of this repository.
   It runs independently of Sections 2–10. Set `INPUT_PATH` to your export.
3. Write the exact target label, its definition, and clear TRUE/FALSE rules in `Rubric`.
   Use your workbook's routing policy. Do not mix these rows with the notebook's generated dataset.
4. Run Jev with `TYPESAFE_API_KEY`. Run the OpenAI judge with `OPENAI_API_KEY` for comparison.
   The notebook asks privately if a key is missing. Neither Google nor LangSmith credentials
   are needed for Section 11. Jev alone needs no OpenAI key.
5. Compare agreement, precision, recall, false positives, and false negatives. Inspect a
   few mismatches and uncertain Jev rows. If there are API errors, fix them and rerun before comparing judges.
6. Improve one rubric rule or try another model. Use a small practice subset to make that
   change, then check separate untouched examples. With very few rows, describe the result
   as exploratory; do not claim it proves a better judge.

Each run saves `results.csv` and `run.json` in a new folder under `judge_runs/`.
Import each CSV into a **new tab** in your workbook (File → Import → Upload → Insert new sheet).
Keep the run folders so the model, rubric, threshold, and dataset fingerprint remain available.
Use Step 5 as the results area and Step 6 for your comparison notes. The code exports files;
it does not write directly into Google Sheets or fill the existing Step 5 template.

Submit your labeled workbook, both result CSVs (if you ran both judges), and two sentences:
which judge you chose, and one mistake or tradeoff you noticed. A strong result is an
honest comparison, not necessarily higher agreement.

## Try the included demo first

Section 11 defaults to `examples/refund_missed_demo.csv` and the matching rubric JSON.
It contains the workbook's 29 classifier failures. The 12 `Refund_Missed` labels come
from its existing `LLM-J Dataset` tab. The other 17 rows were assigned `Other_failure`
as illustrative reference labels while preparing this integration: they describe
delivery/account problems, product descriptions, feedback, or pre-purchase authenticity
questions without post-purchase refund intent. These additions are **not independently
human-validated labels**. The demo is a wiring check, not a benchmark or participant submission.

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

The Jev adapter posts a Noul question to `https://api.typesafe.ai/v1/systemone`.
It records the resolved model, validates probabilities, and retries transient failures.
Errors are shown separately from completed judgments; an error never becomes FALSE.
The default Jev model is pinned to `jev-1.14.0`; change it deliberately when comparing versions.

Official references: [Noul](https://docs.typesafe.ai/primitives/noul),
[Choice](https://docs.typesafe.ai/primitives/choice),
[Score](https://docs.typesafe.ai/primitives/score),
[API](https://docs.typesafe.ai/api/system-one).
