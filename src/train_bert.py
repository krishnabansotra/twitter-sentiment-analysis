"""Level 3: Fine-tune a Twitter-RoBERTa transformer. Run on a GPU (Google Colab is fine)."""
import argparse
import os
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          Trainer, TrainingArguments, DataCollatorWithPadding)

from src.data_loader import load_data, LABELS, ID2LABEL, LABEL2ID, ROOT
from src.preprocessing import normalize_for_transformer
from src.evaluate import evaluate

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"   # or "bert-base-uncased", "distilbert-base-uncased"


def main(source, sample_size, epochs, batch_size, model_name):
    train, val, test = load_data(source, sample_size)
    to_ds = lambda df: Dataset.from_dict({
        "text": [normalize_for_transformer(t) for t in df["text"]],
        "label": [LABEL2ID[l] for l in df["label"]]})
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = lambda b: tok(b["text"], truncation=True, max_length=64)
    tr, va, te = [to_ds(d).map(enc, batched=True) for d in (train, val, test)]

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True)

    def metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return {"accuracy": accuracy_score(p.label_ids, preds),
                "f1_macro": f1_score(p.label_ids, preds, average="macro")}

    out_dir = os.path.join(ROOT, "models", "bert")
    args = TrainingArguments(
        output_dir=os.path.join(ROOT, "models", "bert_checkpoints"),
        num_train_epochs=epochs, per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64, learning_rate=2e-5, weight_decay=0.01,
        eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="f1_macro", fp16=False, report_to="none", save_total_limit=1)
    trainer = Trainer(model=model, args=args, train_dataset=tr, eval_dataset=va,
                      data_collator=DataCollatorWithPadding(tok), compute_metrics=metrics)
    trainer.train()

    preds = np.argmax(trainer.predict(te).predictions, axis=1)
    evaluate([ID2LABEL[i] for i in te["label"]], [ID2LABEL[i] for i in preds], "bert", LABELS)

    trainer.save_model(out_dir); tok.save_pretrained(out_dir)
    print(f"Saved fine-tuned model to {out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="tweet_eval")
    ap.add_argument("--sample_size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--model_name", default=MODEL_NAME)
    a = ap.parse_args()
    main(a.source, a.sample_size, a.epochs, a.batch_size, a.model_name)
