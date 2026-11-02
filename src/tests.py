import unittest

import pandas as pd

from evaluate import evaluate
from model import build_model
from prepare_data import LABEL_COL, TEXT_COL, split_data


def _toy_df():
    pos = [f"great product number {i}" for i in range(30)]
    neg = [f"terrible broken item {i}" for i in range(10)]
    df = pd.DataFrame({TEXT_COL: pos + neg + [pos[0]],
                       LABEL_COL: [1] * 30 + [0] * 10 + [1]})
    return df


class SmokeTests(unittest.TestCase):
    def test_split_dedups_and_stratifies(self):
        train_df, val_df = split_data(_toy_df(), val_size=0.2, seed=0)
        self.assertEqual(len(train_df) + len(val_df), 40)
        self.assertFalse(set(train_df[TEXT_COL]) & set(val_df[TEXT_COL]))
        self.assertEqual(int((val_df[LABEL_COL] == 0).sum()), 2)

    def test_model_fit_and_evaluate(self):
        train_df, val_df = split_data(_toy_df(), val_size=0.2, seed=0)
        model = build_model("tfidf_svm", {"tfidf": {"ngram_range": [1, 2]}, "svm": {"C": 1.0}})
        model.fit(train_df[TEXT_COL], train_df[LABEL_COL])
        metrics = evaluate(model, val_df[TEXT_COL], val_df[LABEL_COL])
        for key in ("accuracy", "macro_f1", "report", "confusion_matrix"):
            self.assertIn(key, metrics)

    def test_lstm_fit_and_evaluate(self):
        train_df, val_df = split_data(_toy_df(), val_size=0.2, seed=0)
        model = build_model("lstm", {"embedding": {"min_freq": 1, "embed_dim": 8},
                                     "lstm": {"hidden_size": 8}, "train": {"epochs": 2, "device": "cpu"}})
        model.fit(train_df[TEXT_COL], train_df[LABEL_COL],
                  val_data=(val_df[TEXT_COL], val_df[LABEL_COL]))
        metrics = evaluate(model, val_df[TEXT_COL], val_df[LABEL_COL])
        self.assertIn("macro_f1", metrics)

    def test_unknown_model(self):
        with self.assertRaises(ValueError):
            build_model("nope")


if __name__ == "__main__":
    unittest.main()
