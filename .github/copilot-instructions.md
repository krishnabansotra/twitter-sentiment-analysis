# Project context
Customer Feedback Intelligence Platform (Python 3.12). Code is in src/, api/, app/.
- Data: data/processed/airline.csv (text, label, created_at)
- Reuse src/preprocessing.py, src/evaluate.py, src/predict.py. Do not rewrite them.
- Every new module goes in src/ with a test in tests/.
- Use scikit-learn and Hugging Face. No paid APIs.
- Evaluate with macro-F1 and save results to models/results/.
- Keep code simple, with short docstrings (student project).