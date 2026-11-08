# Twitter Sentiment Analysis

A sentiment-analysis project for classifying short text as negative, neutral, or positive. It includes classical ML, a PyTorch BiLSTM, and a fine-tuned Hugging Face transformer, with a Streamlit dashboard, SQLite prediction history, data collectors, and a FastAPI application.

## Features

- Train baseline TF-IDF classifiers, a BiLSTM, or the configured transformer from `src/train_baseline.py`, `src/train_lstm.py`, and `src/train_bert.py`.
- Collect text from CSV files, public Reddit search results, and public YouTube comments with `src/collect.py`.
- Analyze a single text or uploaded CSV in the dashboard. Batch analysis can process up to 5,000 rows, skip empty text, and download results.
- Explore saved predictions in the Insights tab with source/model/sentiment filters, sentiment counts over time, and top keywords.
- Review recent predictions in the History section. Saved records are stored in `data/app.db`.
- Use the dashboard's optional live X API tab, model comparison, urgency/reply guidance, and LIME-based explanation.

[SCREENSHOT: dashboard home]

[SCREENSHOT: batch analysis results]

[SCREENSHOT: insights tab]

## Architecture

```text
TweetEval / local labeled CSV -> training scripts -> saved models in models/

CSV / Reddit / YouTube -> src/collect.py -> src/predict.py -> SQLite (data/app.db)
Dashboard single-text / batch prediction -> src/predict.py -> SQLite
Dashboard History / Insights <- SQLite
API client -> FastAPI routes -> src/predict.py -> JSON response
```

`src/predict.py` selects an available saved backend (`baseline`, `lstm`, or `bert`). The dashboard reads saved prediction rows for History and Insights. The API route definitions are present, but the current `api/main.py` has an import-time error described under [Limitations](#limitations).

## Results

These are the values currently recorded in `models/results/comparison.csv`. Accuracy and macro-F1 are shown as stored, without additional rounding.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| bert | 0.7025 | 0.7048 |
| baseline_logreg | 0.5882 | 0.5848 |
| baseline_linear_svm | 0.5713 | 0.5664 |
| lstm | 0.5691 | 0.5542 |
| baseline_naive_bayes | 0.5342 | 0.4334 |

These are saved evaluation results; retraining can produce different values depending on the data and model configuration.

## Quick Start

Run commands from the repository root. In Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Train the baseline on the bundled sample CSV for a local smoke run:

```powershell
python -m src.train_baseline --source sample
```

For the default TweetEval training source, run `python -m src.train_baseline`; it loads TweetEval through Hugging Face and requires network access. The other training entry points are `python -m src.train_lstm` and `python -m src.train_bert`. Each accepts `--source`, `--sample_size`, and model-specific options; see `--help` for the exact arguments. Refresh the saved comparison with:

```powershell
python -m src.evaluate
```

Run the dashboard:

```powershell
streamlit run app/dashboard.py
```

The dashboard requires at least one trained model artifact. Its default local URL is `http://localhost:8501`.

The API launch command is:

```powershell
uvicorn api.main:app --reload
```

However, `api.main` currently fails during import because module-level code refers to undefined `request` and `result` values. The API command and routes below describe the intended interface; the startup defect must be fixed before the API can be used. FastAPI's default documentation route is `/docs` once the app imports successfully.

## Collecting Data

All collection examples require a trained model unless `--no-predict` is used. CSV collection reads the configured text column (default `text`) and stores predictions in SQLite:

```powershell
python -m src.collect --source csv --file data/raw/Datasets/sample_tweets.csv --limit 50 --model baseline
```

YouTube collection requires a `YOUTUBE_API_KEY` in the process environment. This example fetches, predicts, saves to history, and also exports the fetched rows:

```powershell
$env:YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY"
python -m src.collect --source youtube --video "https://youtu.be/TiKuJzi8qpI" --limit 50 --model baseline --export-csv data/processed/youtube_comments.csv
```

To fetch and export without predicting or saving to the database, add `--no-predict`:

```powershell
python -m src.collect --source youtube --video "https://youtu.be/TiKuJzi8qpI" --limit 50 --export-csv data/processed/youtube_comments.csv --no-predict
```

Reddit collection uses `REDDIT_CLIENT_ID` and `REDDIT_CLIENT_SECRET` environment variables:

```powershell
$env:REDDIT_CLIENT_ID = "YOUR_REDDIT_CLIENT_ID"
$env:REDDIT_CLIENT_SECRET = "YOUR_REDDIT_CLIENT_SECRET"
python -m src.collect --source reddit --query "flight delay" --limit 50 --model baseline
```

The optional dashboard Live (X API) tab uses `X_BEARER_TOKEN`. `.env.example` documents that variable, but the code reads it from the process environment; PowerShell can set it with `$env:X_BEARER_TOKEN = "YOUR_X_BEARER_TOKEN"`.

## API Endpoints

The routes defined in `api/main.py` are:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Return service status and available model names. |
| `POST` | `/predict` | Predict one text. Request fields: `text` (1-1,000 characters) and optional `model` (defaults to `baseline`). |
| `POST` | `/predict_batch` | Predict a list of 1-500 texts. Request fields: `texts` and optional `model`. |

Prediction responses are produced by the selected backend and contain sentiment, confidence, and class probabilities. These routes cannot currently be reached until the import-time error noted above is fixed.

## Testing

Activate the virtual environment and run the test suite:

```powershell
python -m pytest -q tests
```

The tests cover preprocessing, model/data pipeline behavior, database operations, collection sources, urgency, explanations, and insight calculations. The bundled sample dataset supports the local pipeline smoke test.

## Project Structure

```text
api/                 FastAPI application: main.py
app/                 Streamlit dashboard and theme: dashboard.py, theme.py
data/
  processed/         airline.csv, analyzed.csv, analyzed_full.csv, issues.csv
  raw/Datasets/      sample_tweets.csv (bundled sample)
src/
  sources/           base.py, csv_source.py, reddit_source.py, youtube_source.py
  batch.py           Batch prediction helper
  collect.py         Fetch, predict, save, and optionally export posts
  db.py              SQLite history
  evaluate.py        Evaluation metrics and model comparison
  predict.py         Unified model prediction interface
  train_baseline.py  TF-IDF and classical classifiers
  train_lstm.py      BiLSTM training
  train_bert.py      Transformer training
  insights.py        Insight aggregations and existing issue/trend utilities
  preprocessing.py   Text preprocessing
models/              Model artifacts, checkpoints, and results/
tests/               pytest suite
Dockerfile           Container image configuration
requirements.txt     Python dependencies
```

Other source modules include data loading/collection, issue and urgency analysis, explanations, replies, artifact checks, external evaluation, and result reporting. Tests are organized by pipeline, source, database, preprocessing, insights, explanation, and urgency behavior.

## Ethics and Privacy

- Use public posts only when collecting from Reddit or YouTube, and respect each service's terms, quotas, and applicable law.
- The source adapters and SQLite schema do not store author usernames. The database does store post text and may store the original post date; YouTube CSV exports also include comment and video identifiers.
- Public text can still contain personal or sensitive information. Do not upload or retain material without appropriate permission; review exported files before sharing them.
- `.gitignore` excludes raw collection files under `data/raw`. The bundled `data/raw/Datasets/sample_tweets.csv` is an explicit tracked exception for local tests. Keep credentials and other raw/private data out of commits.
- Sentiment predictions are estimates. Review model outputs before using them to make decisions about people.

## Limitations

- The API currently fails at import because `api/main.py` executes a prediction-save block at module scope using undefined `request` and `result` variables.
- TweetEval is the default training source and requires network access. `sentiment140` is loaded as a two-class dataset, unlike the project's usual three sentiment labels.
- Reddit and YouTube collection require credentials; live X API access also depends on a valid bearer token and current API availability.
- The dashboard can only select model backends whose artifacts are present. Transformer training may require substantial download time and compute.
- The Insights keyword counts are a simple English-token frequency measure, not topic or phrase analysis. SQLite is a local application database, not a multi-user database service.

## Future Work

- Fix and test API startup, move prediction persistence into the appropriate route, and add API-level tests.
- Add user authentication and operational safeguards before exposing the API or dashboard outside a trusted local environment.
- Add multilingual and phrase-level insights, plus evaluation across consistent datasets and label sets.
- Add data-retention controls and optional redaction for uploaded or collected text.
- Add dashboard/API integration tests and deployment guidance for production environments.
