# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

NTU IE4483 mini project (Option 1): binary sentiment classification of product reviews (0 = negative, 1 = positive). Course materials (assignment PDF, TF-IDF example) live in the gitignored `project1/` and `tfidf_features_example.ipynb`. Other model families (embeddings, RNN, BERT) are allowed and expected to be added later.

## Commands

Python 3.13. Always use `uv` (`uv add`, `uv sync`, `uv run`); never call `pip` directly. Run everything from the repo root.

```
uv sync
uv run python src/main.py --model tfidf_svm                          # train + evaluate on validation split + save
uv run python src/main.py --model tfidf_svm --set model.svm.C=0.5    # override any config value (repeatable)
uv run python src/main.py --model tfidf_svm --predict-test           # also write test-set predictions
cd src && uv run python -m unittest tests                            # smoke tests (flat imports, run from src/)
```

## Architecture

Flat modules in `src/` that import each other directly (`from model import ...`), so scripts are run as `python src/<file>.py`.

- `prepare_data.py`: `load_train/load_test`, `split_data` (drops duplicate reviews *before* a stratified split, to avoid train/val leakage). Defines `TEXT_COL="reviews"`, `LABEL_COL="sentiments"`, `PROJECT_ROOT`.
- `model.py`: `MODELS` registry mapping a name to a `build_*(params)` function. Every model takes **raw text** (`fit(texts, labels)` / `predict(texts)`); feature extraction (e.g. TF-IDF) lives inside the model pipeline. To add a model: write a builder, register it in `MODELS`, add `configs/<name>.toml`.
- `evaluate.py`: `evaluate` (accuracy, macro-F1, per-class report, confusion matrix), `print_report`, `save_report`.
- `main.py`: CLI. Loads `configs/<model>.toml` (or `--config`), applies `--set a.b.c=value` overrides, splits, trains, evaluates on validation, saves `model/<name>.joblib` and `results/<name>_val_metrics.json`.
- `configs/*.toml`: `[data]` (paths, val_size, seed) and `[model.*]` sections whose keys are passed as kwargs to the underlying estimators.

`model/*.joblib` and `results/` are gitignored outputs.

## Data notes

`data/train.json`: 7401 rows, ~85% positive (heavily imbalanced, so watch macro-F1 and negative-class recall, not just accuracy), 216 duplicate reviews. `data/test.json`: 1851 unlabeled rows. Baseline (tfidf_svm defaults): val accuracy 0.924, macro-F1 0.817.
