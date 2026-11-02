"""Model registry.

Every model is an estimator with sklearn-style ``fit(texts, labels)`` and
``predict(texts)`` that takes RAW review text, so feature extraction lives
inside the model. To add a model, write a ``build_*`` function taking a params
dict and register it in ``MODELS``.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_tfidf_svm(params: dict):
    tfidf = dict(params.get("tfidf", {}))
    svm = dict(params.get("svm", {}))
    if "ngram_range" in tfidf:
        tfidf["ngram_range"] = tuple(tfidf["ngram_range"])
    return Pipeline([
        ("tfidf", TfidfVectorizer(**tfidf)),
        ("svm", LinearSVC(**svm)),
    ])


def build_lstm(params: dict):
    from lstm_model import LSTMClassifier  # lazy: torch is only needed for this model
    return LSTMClassifier(params.get("embedding"), params.get("lstm"), params.get("train"))


MODELS = {
    "tfidf_svm": build_tfidf_svm,
    "lstm": build_lstm,
}


def build_model(name: str, params: dict | None = None):
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(MODELS)}")
    return MODELS[name](params or {})
