# ple-harness

Dev-run harness for the Kaggle **Physical Engineering's Last Exam (PLE) Agent
Harness** competition ($400,000; deadline 5 Jan 2027 23:59 UTC).

This repo is a fork of the kit's `sample_submissions/minimal/` (one model, three
tools, one loop) wired to a free model for dev-set evaluation. The local dev
machine cannot run Docker containers (its kernel blocks `setns`), so the dev
set is evaluated on **GitHub Actions** runners, which have Docker.

## Layout

- `serve.py`, `taskbridge.py`, `requirements.txt` — the minimal harness.
- `submission.yaml` — declares `openai-compatible/nvidia/nemotron-3.5-super-vl:free`
  via OpenRouter for dev runs. **Not submittable as-is**: the competition only
  accepts `google-cloud/<id>` models. The final submission switches `model_api`
  to a sponsored model (planned: `google-cloud/gemini-3.8-flash`).
- `.github/workflows/eval.yml` — runs `eval/run.py` on the pinned kit
  (`google-research/ple` @ `8320f04`) for selected dev tasks, one job per task.

## Running the eval (GitHub Actions)

Repo secrets required (Settings → Secrets and variables → Actions):

| Secret | What it is |
|---|---|
| `HF_TOKEN` | Hugging Face read token of the account granted access to `google/physical-engineerings-last-exam` |
| `OPENROUTER_API_KEY` | OpenRouter API key (free tier) — model calls and the dev judge |

Then: Actions → **PLE dev-set eval** → Run workflow → `tasks` (comma-separated
task IDs; empty = all 29 dev tasks). Each job runs one task (up to 6 h) and
uploads its `runs/` output as an artifact named `ple-run-<task>`.

## Local runs

Local runs need Docker, `uv`, and Harbor 0.20.0 — see the kit's
`eval/README.md`. With the kit checked out beside this repo:

```bash
uv run python ../ple/eval/run.py . --env-file local.env --tasks wind-diaphragm-shear
```

`local.env` (kept outside this repo) holds `HF_TOKEN` and `MODEL_API_KEY`.

## License

Apache-2.0 (see `LICENSE`). `serve.py` / `taskbridge.py` are Google's
sample-submission code under the same license.
