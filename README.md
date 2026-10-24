# Customer Support Agent Evaluation

This project evaluates an e-commerce customer support routing agent. Given a support ticket, the agent assigns one of five categories:

- `order_status`
- `refund_request`
- `product_issue`
- `account_help`
- `other`

The repository includes an executed evaluation notebook, the human-review workbook, exported results, and a short guide to the OpenTelemetry integration.

## Repository Contents

| File | Description |
| --- | --- |
| [`week4_customer_support_evals.ipynb`](week4_customer_support_evals.ipynb) | Executed notebook containing the classifier, evaluation workflow, prompt revision, results, and trace instrumentation. |
| [`Week 4_ AI Evals (E-Commerce Customer Support Agent).xlsx`](Week%204_%20AI%20Evals%20%28E-Commerce%20Customer%20Support%20Agent%29.xlsx) | Evaluation workbook for reviewing failures, grouping error patterns, revising prompts, and optionally aligning an LLM judge. |
| [`results_v1.csv`](results_v1.csv) | Baseline predictions and evaluation results. |
| [`results_v2.csv`](results_v2.csv) | Improved-prompt predictions and evaluation results. |
| [`docs/opentelemetry_integration_one_pager.md`](docs/opentelemetry_integration_one_pager.md) | Overview of the OpenTelemetry and LangSmith tracing setup. |
| [`judge_evaluation.py`](judge_evaluation.py) | Binary OpenAI and Jev judges, input checks, metrics, and CSV exports for notebook Section 11. |
| [`docs/jev_judge.md`](docs/jev_judge.md) | Short participant exercise, setup, and Noul / Choice / Score guide. |

## Results

| Run | Accuracy | Correct predictions |
| --- | ---: | ---: |
| Baseline prompt | 92% | 92 / 100 |
| Improved prompt | 97% | 97 / 100 |
| Change | +5 percentage points | +5 |

The revised prompt improved routing accuracy while preserving the individual prediction records needed for failure analysis.

## Evaluation Workflow

The notebook and workbook support the following review process:

1. Run the baseline classifier and compare predictions with the expected categories.
2. Annotate incorrect predictions and group recurring failure patterns.
3. Revise the classifier prompt to address a selected failure pattern.
4. Rerun the evaluation and compare the new results with the baseline.
5. Optionally compare OpenAI and Jev judges against your failure labels in notebook Section 11.

## Jev and LLM judges (optional)

Section 11 now implements both judges. It runs on saved, labeled failures without rerunning
the classifier. Jev uses **Noul**, its yes/no question type, to detect one chosen failure
category. OpenAI answers the same question with a boolean and a short reason.

Start with the included demo, then use your Step 3 CSV/XLSX and your own rubric.
Set `TYPESAFE_API_KEY` for Jev and `OPENAI_API_KEY` for the OpenAI comparison; the notebook
can ask for them privately. Install the section's dependencies with
`pip install -r requirements-judge.txt`. Each run exports a CSV to import as a new workbook tab.

The workbook and generated notebook data use different routing policies; keep each dataset
with its own labels. These example labels are provided for practice. For your evaluation, use the labels you reviewed and assigned in Step 3.
See the [participant guide](docs/jev_judge.md) for the exercise and submission steps.

## OpenTelemetry and LangSmith

Each ticket classification is wrapped in an OpenTelemetry span. The span records the ticket ID, prompt version, expected category, predicted category, correctness, and model reasoning. LangSmith receives these traces so individual failures can be inspected alongside the aggregate evaluation results.

See [`docs/opentelemetry_integration_one_pager.md`](docs/opentelemetry_integration_one_pager.md) for implementation details and troubleshooting guidance.

## Run the Notebook

Create a virtual environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install jupyter langgraph "langsmith[otel]>=0.4.25" langchain-openai langchain-core pandas scikit-learn pydantic tqdm opentelemetry-sdk opentelemetry-exporter-otlp
```

Set your API keys and tracing configuration locally:

```bash
export OPENAI_API_KEY="your-openai-key"
export LANGSMITH_API_KEY="your-langsmith-key"
export LANGSMITH_TRACING=true
export LANGSMITH_PROJECT=customer-support-evals
export LANGSMITH_ENDPOINT=https://api.smith.langchain.com
export LANGSMITH_OTEL_ENABLED=true
```

Then launch the notebook:

```bash
jupyter notebook week4_customer_support_evals.ipynb
```

Do not commit API keys, `.env` files, or other secrets.

## References

- [Customer Support Agent Evaluation solution kit](https://docs.google.com/document/d/1zsxSbgZxsLVNVeOx3n95w555xAykzqZM9ewc9a4rTtk/edit?tab=t.0)
- [Evaluation spreadsheet](https://docs.google.com/spreadsheets/d/1DQXHydc-zx2fPh9flEwUKXyIMERs153CBvbLzUqqZWw/edit)
- [Submission form](https://forms.gle/ziAMAGXSvzyWzqAU7)
