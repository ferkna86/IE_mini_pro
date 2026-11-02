import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


def evaluate(model, texts, labels) -> dict:
    """Apply a fitted model to a labelled set (e.g. validation) and score it."""
    preds = model.predict(texts)
    return {
        "n_samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "report": classification_report(
            labels, preds, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


def print_report(metrics: dict, title: str = "Validation") -> None:
    print(f"\n=== {title} results ({metrics['n_samples']} samples) ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print("Per class (0 = negative, 1 = positive):")
    for cls in ("0", "1"):
        r = metrics["report"].get(cls)
        if r:
            print(
                f"  {cls}: precision={r['precision']:.4f} "
                f"recall={r['recall']:.4f} f1={r['f1-score']:.4f} "
                f"support={int(r['support'])}"
            )
    print("Confusion matrix (rows = true, cols = predicted):")
    for row in metrics["confusion_matrix"]:
        print("  ", row)


def save_report(metrics: dict, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def misclassified(model, texts, labels) -> "pd.DataFrame":
    """Return the rows the model got wrong, with true/predicted label and error type."""
    preds = model.predict(texts)
    df = pd.DataFrame({
        "review": list(texts),
        "true": list(labels),
        "predicted": list(preds),
    })
    if hasattr(model, "decision_function"):
        df["score"] = model.decision_function(texts)
    df = df[df["true"] != df["predicted"]].copy()
    df["error_type"] = df["predicted"].map({1: "false_positive", 0: "false_negative"})
    return df.sort_values(["error_type"]).reset_index(names="val_row")


def full_params(model) -> dict:
    """Every hyper-parameter actually used (explicit config values and sklearn defaults)."""
    steps = getattr(model, "steps", [("model", model)])
    return {name: est.get_params() for name, est in steps}


def save_summary_md(path, *, model_name, metrics, config, effective_params, sizes, n_errors) -> None:
    """Write a human-readable results + hyper-parameter document (Markdown)."""
    rep = metrics["report"]
    (tn, fp), (fn, tp) = metrics["confusion_matrix"]
    L = [f"# {model_name}: validation results", ""]
    L += ["## Overall", "", "| Metric | Value |", "|---|---|",
          f"| Accuracy | {metrics['accuracy']:.4f} |",
          f"| Macro F1 | {metrics['macro_f1']:.4f} |",
          f"| Validation samples | {metrics['n_samples']} |",
          f"| Misclassified | {n_errors} |", ""]
    L += ["## Per class", "", "| Class | Precision | Recall | F1 | Support |", "|---|---|---|---|---|"]
    for cls, label in (("0", "0 (negative)"), ("1", "1 (positive)")):
        r = rep.get(cls)
        if r:
            L.append(f"| {label} | {r['precision']:.4f} | {r['recall']:.4f} | "
                     f"{r['f1-score']:.4f} | {int(r['support'])} |")
    L += ["", "## Confusion matrix", "", "| | Pred 0 | Pred 1 |", "|---|---|---|",
          f"| **True 0** | {tn} | {fp} |", f"| **True 1** | {fn} | {tp} |", ""]
    L += ["## Data", "", f"- Train samples: {sizes['train']}", f"- Validation samples: {sizes['val']}"]
    for k, v in config.get("data", {}).items():
        L.append(f"- `{k}` = `{v}`")
    L += ["", "## Hyper-parameters (from config / CLI overrides)", "", "```json",
          json.dumps(config.get("model", {}), indent=2), "```", "",
          "## Effective parameters (including sklearn defaults)", ""]
    for step, params in effective_params.items():
        L += [f"### {step}", "", "| Parameter | Value |", "|---|---|"]
        L += [f"| `{k}` | `{v}` |" for k, v in sorted(params.items())]
        L.append("")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L), encoding="utf-8")
