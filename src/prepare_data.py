from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEXT_COL = "reviews"
LABEL_COL = "sentiments"


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_train(path="data/train.json") -> pd.DataFrame:
    return pd.read_json(_resolve(path))


def load_test(path="data/test.json") -> pd.DataFrame:
    return pd.read_json(_resolve(path))


def split_data(df: pd.DataFrame, val_size: float = 0.2, seed: int = 42):
    """Drop duplicate reviews, then make a stratified train/validation split.

    Deduplicating first keeps identical reviews from landing on both sides.
    """
    df = df.drop_duplicates(subset=[TEXT_COL]).reset_index(drop=True)
    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        random_state=seed,
        stratify=df[LABEL_COL],
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)
