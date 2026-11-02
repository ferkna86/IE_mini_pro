import json
from pathlib import Path

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
