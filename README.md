# Mini Project: Sentiment Analysis of Product Reviews

Binary sentiment classification (0 = negative, 1 = positive) of product reviews. The current baseline is TF-IDF features + a linear SVM. The code is structured so other models can be added and selected from the command line.

## Setup

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/). All dependencies are managed with `uv`; do not use `pip` directly.

```
uv sync
```

## Data

| File | Rows | Columns |
|---|---|---|
| `data/train.json` | 7401 | `reviews` (text), `sentiments` (0/1) |
| `data/test.json` | 1851 | `reviews` (text, no labels) |

The training set is imbalanced (about 85% positive) and contains 216 duplicate reviews. Duplicates are removed before the train/validation split. Because of the imbalance, look at macro-F1 and negative-class recall as well as accuracy.

## Usage

Run from the repository root:

```
uv run python src/main.py --model tfidf_svm
```

This trains the model on the training split, evaluates it on the validation split, prints the results, and saves outputs.

| Option | Description |
|---|---|
| `--model NAME` | Model to train (required). Currently: `tfidf_svm`. |
| `--config PATH` | TOML config to use. Default: `configs/<model>.toml`. |
| `--set KEY=VALUE` | Override a config value, e.g. `--set model.svm.C=0.5`. Repeatable. Values are parsed as Python literals, so strings need inner quotes: `--set model.svm.class_weight='"balanced"'`. |
| `--predict-test` | Also write predictions for the test set. |

Outputs (gitignored):
- `model/<name>.joblib`: the fitted model (feature extraction included)
- `results/<name>_val_metrics.json`: validation accuracy, macro-F1, per-class report, confusion matrix, and the parameters used
- `results/<name>_test_predictions.csv`: only with `--predict-test`

## Tuning parameters

Edit `configs/tfidf_svm.toml`, or override on the command line.

```toml
[data]
val_size = 0.2
seed = 42

[model.tfidf]          # passed to sklearn TfidfVectorizer
ngram_range = [1, 2]
sublinear_tf = true
min_df = 1

[model.svm]            # passed to sklearn LinearSVC
C = 1.0
max_iter = 10000
# class_weight = "balanced"
```

Any keyword argument accepted by the underlying scikit-learn estimator can be added to its section.

## Project structure

```
configs/             Model/data configuration (TOML), one file per model
data/                train.json, test.json
src/
  main.py            CLI entry point: config -> split -> train -> evaluate -> save
  prepare_data.py    Data loading and the deduplicated, stratified train/val split
  model.py           Model registry (MODELS) and model builders
  evaluate.py        Validation metrics: compute, print, save
  tests.py           Smoke tests
```

## Adding a new model

Every model takes raw review text and exposes `fit(texts, labels)` and `predict(texts)`, with feature extraction inside the model.

1. Write a `build_<name>(params: dict)` function in `src/model.py` that returns such an estimator.
2. Register it in the `MODELS` dict.
3. Add `configs/<name>.toml` with its parameters.
4. Run `uv run python src/main.py --model <name>`.

## Tests

```
cd src
uv run python -m unittest tests
```

(Run from `src/` because the modules import each other directly.)

## Baseline results

TF-IDF (unigrams + bigrams, sublinear tf) + LinearSVC, 80/20 stratified split, seed 42, 1437 validation samples:

| Config | Accuracy | Macro-F1 | Negative recall |
|---|---|---|---|
| `C=1.0` (default) | 0.924 | 0.817 | 0.54 |
| `C=0.5` | 0.916 | 0.786 | 0.47 |
| `C=0.1`, `class_weight="balanced"` | 0.905 | 0.819 | 0.74 |

Negative reviews are the hard class, so `class_weight` and `C` are the first things to tune.
