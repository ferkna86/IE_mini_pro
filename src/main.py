"""Train a model, evaluate it on the validation split, and save it.

Usage (from repo root):
    uv run python src/main.py --model tfidf_svm
    uv run python src/main.py --model tfidf_svm --set model.svm.C=0.5 --predict-test
"""
import argparse
import ast
import tomllib

import joblib
import pandas as pd

from evaluate import evaluate, print_report, save_report
from model import MODELS, build_model
from prepare_data import (
    LABEL_COL,
    PROJECT_ROOT,
    TEXT_COL,
    load_test,
    load_train,
    split_data,
)


def load_config(path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def apply_override(cfg: dict, item: str) -> None:
    """Apply 'a.b.c=value' to the nested config dict (value parsed as a Python literal)."""
    key, sep, raw = item.partition("=")
    if not sep:
        raise ValueError(f"--set expects key=value, got '{item}'")
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        value = raw
    node = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(MODELS), help="model to train")
    p.add_argument("--config", help="TOML config (default: configs/<model>.toml)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. model.svm.C=0.5 (repeatable)")
    p.add_argument("--predict-test", action="store_true",
                   help="also write predictions for the test set")
    return p.parse_args()


def main():
    args = parse_args()
    config_path = args.config or PROJECT_ROOT / "configs" / f"{args.model}.toml"
    cfg = load_config(config_path)
    for item in args.set:
        apply_override(cfg, item)

    data_cfg = cfg.get("data", {})
    params = cfg.get("model", {})

    df = load_train(data_cfg.get("train_path", "data/train.json"))
    train_df, val_df = split_data(df, data_cfg.get("val_size", 0.2), data_cfg.get("seed", 42))
    print(f"Model: {args.model} | train={len(train_df)} val={len(val_df)}")
    print(f"Params: { {k: v for k, v in params.items() if k != 'name'} }")

    model = build_model(args.model, params)
    if getattr(model, "uses_validation", False):  # iterative models report val metrics per epoch
        model.fit(train_df[TEXT_COL], train_df[LABEL_COL],
                  val_data=(val_df[TEXT_COL], val_df[LABEL_COL]))
    else:
        model.fit(train_df[TEXT_COL], train_df[LABEL_COL])

    metrics = evaluate(model, val_df[TEXT_COL], val_df[LABEL_COL])
    print_report(metrics)

    results_dir = PROJECT_ROOT / "results"
    save_report({"model": args.model, "params": params, **metrics},
                results_dir / f"{args.model}_val_metrics.json")

    model_dir = PROJECT_ROOT / "model"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / f"{args.model}.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    if args.predict_test:
        test_df = load_test(data_cfg.get("test_path", "data/test.json"))
        out = pd.DataFrame({TEXT_COL: test_df[TEXT_COL],
                            LABEL_COL: model.predict(test_df[TEXT_COL])})
        out_path = results_dir / f"{args.model}_test_predictions.csv"
        out.to_csv(out_path, index=False)
        print(f"Test predictions saved to: {out_path}")


if __name__ == "__main__":
    main()
