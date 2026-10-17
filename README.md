# Twitter/X Sentiment Analysis (Major Project)

Classifies tweets as **positive / neutral / negative** using three model levels and compares them:

| Level | Model | File |
|---|---|---|
| 1 | TF-IDF + Logistic Regression / Naive Bayes / Linear SVM | `src/train_baseline.py` |
| 2 | BiLSTM (PyTorch) | `src/train_lstm.py` |
| 3 | Fine-tuned Twitter-RoBERTa (Hugging Face) | `src/train_bert.py` |

Served through a **FastAPI** backend and a **Streamlit** dashboard.

## Architecture

```
Data (TweetEval / Sentiment140 / CSV / X API)
        -> preprocessing.py (clean_tweet / normalize_for_transformer)
        -> train_baseline | train_lstm | train_bert
        -> evaluate.py (accuracy, precision, recall, F1, confusion matrix, comparison table)
        -> models/  ->  predict.py (unified predictor)
                         |-> api/main.py       (REST API)
                         |-> app/dashboard.py  (Streamlit UI)
```

## Setup

```bash
py -m venv .venv
.\.venv\Scripts\Activate.ps1    # Windows PowerShell
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick smoke test (no internet, uses the bundled 90-tweet sample)

```bash
python -m pytest -q tests
python -m src.train_baseline --source sample
streamlit run app/dashboard.py
```

## Real training (run all commands from the project root)

```bash
# Level 1: fast, CPU is fine (downloads TweetEval automatically)
python -m src.train_baseline

# Level 2: BiLSTM (CPU ok, GPU faster)
python -m src.train_lstm --epochs 6

# Level 3: RoBERTa fine-tuning. Use a GPU (Google Colab free tier works).
python -m src.train_bert --epochs 3

# Comparison table of all trained models
python -m src.evaluate
```

Tip: while developing, add `--sample_size 5000` to train quickly on a subset.

**Using Sentiment140 instead** (binary): download from Kaggle, save as `data/raw/sentiment140.csv`, then
`python -m src.train_baseline --source sentiment140 --sample_size 200000`.
(Train all models with the same source so the comparison is fair.)

**Your own data:** a CSV with columns `text,label` (labels: negative/neutral/positive), pass `--source yourfile.csv`.

## Run the services

```bash
uvicorn api.main:app --reload          # http://localhost:8000/docs
streamlit run app/dashboard.py         # http://localhost:8501
```

API example:
```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"text": "I love this phone!", "model": "baseline"}'
```

## Live tweets (optional)

Copy `.env.example` to `.env`, add your X API bearer token, export it (`export X_BEARER_TOKEN=...`), then use the
"Live (X API)" tab. X API access and pricing change often, so check developer.x.com. If you have no access, use the CSV upload tab.

## Docker

```bash
docker build -t tweet-sentiment .
docker run -p 8501:8501 tweet-sentiment
```
(Train the models first so `models/` contains files, or mount it with `-v $(pwd)/models:/app/models`.)

## Project structure

```
data/          raw & processed data, sample_tweets.csv
src/           preprocessing, data_loader, training scripts, evaluate, predict, data_collection
api/           FastAPI app
app/           Streamlit dashboard
models/        saved models and results/ (metrics JSON, confusion matrices)
tests/         pytest tests
```

## Ideas to extend (for extra marks)

- Hinglish / multilingual support (e.g., `cardiffnlp/twitter-xlm-roberta-base-sentiment`)
- Sarcasm detection, emoji-aware features
- Topic modeling with BERTopic/LDA to explain *why* sentiment is positive/negative
- Hyperparameter tuning and cross-validation
- Explainability with LIME/SHAP

## Evidence and evaluation

Review sentiment errors and export examples for annotation:
```bash
python -m src.error_analysis --backend baseline --n 60
```

Write bootstrap model comparisons and confidence intervals:
```bash
python -m src.results_report
```

Evaluate a saved model on a separate labeled CSV without retraining:
```bash
python -m src.external_eval --input reviews.csv --text-col text --label-col label --label-mode text --backend baseline
```

Explain an individual prediction with LIME:
```bash
python -m src.explain "The flight was delayed and the staff were rude" --backend baseline
```
