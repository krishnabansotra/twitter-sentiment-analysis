"""Level 2: BiLSTM (PyTorch)."""
import argparse
import json
import os
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_loader import load_data, LABELS, LABEL2ID, ROOT
from src.preprocessing import clean_tweet
from src.evaluate import evaluate

MAX_LEN, MAX_VOCAB = 40, 30000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, emb=128, hidden=128, n_classes=3, dropout=0.4):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden * 2, n_classes)

    def forward(self, x):
        out, _ = self.lstm(self.emb(x))
        pooled = out.max(dim=1).values          # max-pool over time
        return self.fc(self.drop(pooled))


def build_vocab(texts):
    counts = Counter(w for t in texts for w in t.split())
    vocab = {"<pad>": 0, "<unk>": 1}
    for w, _ in counts.most_common(MAX_VOCAB - 2):
        vocab[w] = len(vocab)
    return vocab


def encode(texts, vocab):
    arr = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, 1) for w in t.split()][:MAX_LEN]
        arr[i, :len(ids)] = ids
    return torch.tensor(arr)


def main(source, sample_size, epochs, batch_size):
    train, val, test = load_data(source, sample_size)
    for d in (train, val, test):
        d["clean"] = d["text"].apply(clean_tweet)
    labels = sorted(set(train["label"]), key=lambda l: LABEL2ID.get(l, 99))
    l2i = {l: i for i, l in enumerate(labels)}

    vocab = build_vocab(train["clean"])
    mk = lambda df: TensorDataset(encode(df["clean"], vocab),
                                  torch.tensor(df["label"].map(l2i).values))
    tr_dl = DataLoader(mk(train), batch_size=batch_size, shuffle=True)
    va_dl = DataLoader(mk(val), batch_size=256)
    te_dl = DataLoader(mk(test), batch_size=256)

    model = BiLSTM(len(vocab), n_classes=len(labels)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    def run_eval(dl):
        model.eval(); preds, golds = [], []
        with torch.no_grad():
            for x, y in dl:
                preds += model(x.to(DEVICE)).argmax(1).cpu().tolist()
                golds += y.tolist()
        return golds, preds

    best_acc, best_state = 0, None
    for ep in range(1, epochs + 1):
        model.train(); total = 0
        for x, y in tr_dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            total += loss.item()
        g, p = run_eval(va_dl)
        acc = float(np.mean(np.array(g) == np.array(p)))
        print(f"Epoch {ep}/{epochs}  loss={total/len(tr_dl):.4f}  val_acc={acc:.4f}")
        if acc >= best_acc:
            best_acc, best_state = acc, {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    g, p = run_eval(te_dl)
    evaluate([labels[i] for i in g], [labels[i] for i in p], "lstm", labels)

    out = os.path.join(ROOT, "models")
    torch.save({"state": best_state, "n_classes": len(labels), "labels": labels},
               os.path.join(out, "lstm_model.pt"))
    with open(os.path.join(out, "lstm_vocab.json"), "w") as f:
        json.dump(vocab, f)
    print("Saved LSTM to models/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="tweet_eval")
    ap.add_argument("--sample_size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=64)
    a = ap.parse_args()
    main(a.source, a.sample_size, a.epochs, a.batch_size)
