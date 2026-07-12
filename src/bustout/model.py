"""Early-warning model for bust-out: gradient boosting over the trajectory features.

The model is feature-agnostic (it takes a list of feature columns), so the same code serves
whatever feature set the panel supports. Two things are specific to this problem:

  time split      the panel is split by statement date, training on earlier months and
                  testing on later ones, because a monitoring model only ever has the past.

  scoreable rows  months after an account has already busted (its charge-off tail) are
                  dropped from training and evaluation: the account is gone and there is no
                  action left to take, so scoring those months measures nothing useful.
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


def scoreable(panel: pd.DataFrame) -> pd.Series:
    """Rows a monitoring model would actually score: everything except post-bust months."""
    m2b = panel["months_to_bust"]
    return m2b.isna() | (m2b >= 0)


def time_split(panel: pd.DataFrame, frac: float = 0.55):
    """Split account-months chronologically by statement date into (train, test, cutoff)."""
    dates = panel["statement_date"].sort_values(kind="mergesort").to_numpy()
    cutoff = pd.Timestamp(dates[int(len(dates) * frac)])
    train = panel[panel["statement_date"] < cutoff]
    test = panel[panel["statement_date"] >= cutoff]
    return train, test, cutoff


def train_model(train: pd.DataFrame, feature_cols: list[str],
                label: str = "will_bustout", seed: int = 0) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_depth=4,
        learning_rate=0.06,
        max_iter=400,
        l2_regularization=1.0,
        min_samples_leaf=40,
        class_weight="balanced",
        early_stopping=True,
        validation_fraction=0.15,
        random_state=seed,
    )
    model.fit(train[feature_cols], train[label])
    return model


def score_model(model: HistGradientBoostingClassifier, df: pd.DataFrame,
                feature_cols: list[str]) -> pd.Series:
    prob = model.predict_proba(df[feature_cols])[:, 1]
    return pd.Series(prob, index=df.index, name="bustout_prob")
