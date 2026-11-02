"""LSTM + linear classifier with a sklearn-style fit/predict interface on raw text."""
import copy
import re
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

PAD, UNK = 0, 1
_TOKEN_RE = re.compile(r"[a-z0-9']+|[^\sa-z0-9]")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(str(text).lower())


class LSTMNet(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, bidirectional, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD)
        self.lstm = nn.LSTM(embed_dim, hidden_size, num_layers=num_layers, batch_first=True,
                            bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * (2 if bidirectional else 1), 2)
        self.bidirectional = bidirectional

    def forward(self, x, lengths):
        packed = pack_padded_sequence(self.drop(self.embed(x)), lengths.cpu(),
                                      batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        h = torch.cat([h[-2], h[-1]], dim=1) if self.bidirectional else h[-1]
        return self.fc(self.drop(h))


class LSTMClassifier:
    """Tokenize -> embedding -> LSTM -> Linear. ``fit`` prints validation metrics every epoch."""

    uses_validation = True  # main.py passes val_data to fit

    def __init__(self, embedding=None, lstm=None, train=None):
        e, l, t = embedding or {}, lstm or {}, train or {}
        self.max_vocab = e.get("max_vocab", 20000)
        self.min_freq = e.get("min_freq", 2)
        self.max_len = e.get("max_len", 200)
        self.embed_dim = e.get("embed_dim", 100)
        self.hidden_size = l.get("hidden_size", 128)
        self.num_layers = l.get("num_layers", 1)
        self.bidirectional = l.get("bidirectional", True)
        self.dropout = l.get("dropout", 0.3)
        self.epochs = t.get("epochs", 10)
        self.batch_size = t.get("batch_size", 64)
        self.lr = t.get("lr", 1e-3)
        self.weight_decay = t.get("weight_decay", 0.0)
        self.grad_clip = t.get("grad_clip", 5.0)
        self.class_weight = t.get("class_weight", None)  # None | "balanced"
        self.keep_best = t.get("keep_best", True)  # restore epoch with best val macro-F1
        self.seed = t.get("seed", 42)
        self.device = t.get("device", "auto")

    # ---- data ----
    def _build_vocab(self, texts):
        counts = Counter(tok for t in texts for tok in tokenize(t))
        words = [w for w, c in counts.most_common(self.max_vocab - 2) if c >= self.min_freq]
        self.vocab_ = {w: i + 2 for i, w in enumerate(words)}

    def _encode(self, texts):
        seqs = [[self.vocab_.get(tok, UNK) for tok in tokenize(t)][: self.max_len] or [UNK]
                for t in texts]
        lengths = torch.tensor([len(s) for s in seqs])
        x = torch.zeros(len(seqs), int(lengths.max()), dtype=torch.long)
        for i, s in enumerate(seqs):
            x[i, : len(s)] = torch.tensor(s)
        return x, lengths

    def _batches(self, x, lengths, y=None, shuffle=False):
        idx = torch.randperm(len(x)) if shuffle else torch.arange(len(x))
        for i in range(0, len(x), self.batch_size):
            b = idx[i: i + self.batch_size]
            n = int(lengths[b].max())
            yield (x[b, :n].to(self.dev_), lengths[b],
                   None if y is None else y[b].to(self.dev_))

    # ---- sklearn-style API ----
    def fit(self, texts, labels, val_data=None):
        torch.manual_seed(self.seed)
        self.dev_ = torch.device("cuda" if self.device == "auto" and torch.cuda.is_available()
                                 else "cpu" if self.device == "auto" else self.device)
        texts, y_np = list(texts), np.asarray(labels)
        self._build_vocab(texts)
        x, lengths = self._encode(texts)
        y = torch.tensor(y_np, dtype=torch.long)

        self.net_ = LSTMNet(len(self.vocab_) + 2, self.embed_dim, self.hidden_size,
                            self.num_layers, self.bidirectional, self.dropout).to(self.dev_)
        weight = None
        if self.class_weight == "balanced":
            cnt = np.bincount(y_np, minlength=2)
            weight = torch.tensor(len(y_np) / (2 * cnt), dtype=torch.float, device=self.dev_)
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        print(f"Device: {self.dev_} | vocab={len(self.vocab_) + 2} | epochs={self.epochs}")
        best_f1, best_state, best_epoch = -1.0, None, 0
        for epoch in range(1, self.epochs + 1):
            self.net_.train()
            total = 0.0
            for xb, lb, yb in self._batches(x, lengths, y, shuffle=True):
                opt.zero_grad()
                loss = loss_fn(self.net_(xb, lb), yb)
                loss.backward()
                if self.grad_clip:
                    nn.utils.clip_grad_norm_(self.net_.parameters(), self.grad_clip)
                opt.step()
                total += loss.item() * len(yb)
            line = f"Epoch {epoch:>3}/{self.epochs} | train_loss={total / len(y):.4f}"
            if val_data is not None:
                vt, vy = val_data
                pred = self.predict(vt)
                acc = accuracy_score(vy, pred)
                f1 = f1_score(vy, pred, average="macro")
                line += f" | val_acc={acc:.4f} val_macro_f1={f1:.4f}"
                if f1 > best_f1:
                    best_f1, best_epoch = f1, epoch
                    best_state = copy.deepcopy(self.net_.state_dict())
            print(line)
        if val_data is not None and self.keep_best and best_state is not None:
            self.net_.load_state_dict(best_state)
            print(f"Restored best epoch {best_epoch} (val macro-F1={best_f1:.4f})")
        return self

    @torch.no_grad()
    def predict(self, texts):
        self.net_.eval()
        x, lengths = self._encode(list(texts))
        out = [self.net_(xb, lb).argmax(1).cpu() for xb, lb, _ in self._batches(x, lengths)]
        return torch.cat(out).numpy()
