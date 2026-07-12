"""Leakage-safe longitudinal features for bust-out detection.

Every feature for an account at month t uses only that account's statements up to and
including t, never a later month, because a monitoring model scores an account with the
history it has so far. The forward label (does the account bust within the horizon) is the
only thing allowed to look ahead, and it is attached separately in `panel_data.add_labels`.

The features are built to separate a bust-out ramp from genuine distress, which look alike
at a single month. The difference is in the path: a bust-out climbs fast off a pristine,
full-paying history, often right after a limit increase, and then draws cash and stops
paying; distress climbs slowly, pays the minimum for a long time, and its spend falls as
the room runs out. So the features lean on slopes, jumps above a prior peak, payment
streaks and their breaks, limit growth, and cash-draw share, not on levels alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "utilization",
    "util_slope_3m",
    "util_accel",
    "util_jump_over_prior_peak",
    "payment_ratio",
    "payment_ratio_drop",
    "full_pay_streak_prior",
    "min_pay_streak",
    "cash_advance_share",
    "spend_share_limit",
    "spend_slope_3m",
    "limit_growth_ratio",
    "limit_growth_6m",
    "months_since_limit_up",
    "util_after_limit_up",
    "dpd",
    "dpd_rising",
    "util_vol_6m",
    "months_on_book",
]


def _streak(flag: pd.Series, account: pd.Series) -> pd.Series:
    """Trailing count of consecutive True ending at each row, per account (0 if current False)."""
    block = (~flag).groupby(account).cumsum()
    streak = flag.groupby([account, block]).cumcount() + 1
    return streak.where(flag, 0)


def build_features(panel: pd.DataFrame):
    """Return (panel_with_features, feature_cols): leakage-safe trajectory features."""
    df = panel.sort_values(["account_id", "month_index"]).reset_index(drop=True)
    acct = df["account_id"]
    g = df.groupby("account_id", sort=False)

    util = df["utilization"]
    prev_bal = g["balance"].shift(1)
    payment_ratio = (df["payments"] / (prev_bal + 1.0)).clip(0, 2)
    paid_full = payment_ratio >= 0.9
    near_min = (df["payments"] <= 1.2 * df["min_payment_due"]) & (prev_bal > 0)

    util_max_prior = g["utilization"].cummax().groupby(acct).shift(1)
    limit_up = (df["credit_limit"] - g["credit_limit"].shift(1)) > 1.0
    first_limit = g["credit_limit"].transform("first")

    # months since the last limit increase, and where utilisation sat just before it
    up_month = df["month_index"].where(limit_up)
    last_up_month = up_month.groupby(acct).ffill()
    months_since_up = (df["month_index"] - last_up_month)

    spend_share = df["purchases"] / df["credit_limit"]

    feats = {
        "utilization": util,
        "util_slope_3m": util - g["utilization"].shift(3),
        "util_accel": util - 2 * g["utilization"].shift(1) + g["utilization"].shift(2),
        "util_jump_over_prior_peak": util - util_max_prior,
        "payment_ratio": payment_ratio,
        "full_pay_streak_prior": _streak(paid_full, acct).groupby(acct).shift(1),
        "min_pay_streak": _streak(near_min, acct),
        "cash_advance_share": df["cash_advance"] / (df["purchases"] + df["cash_advance"] + 1.0),
        "spend_share_limit": spend_share,
        "spend_slope_3m": spend_share - g["purchases"].shift(3) / df["credit_limit"],
        "limit_growth_ratio": df["credit_limit"] / first_limit,
        "limit_growth_6m": df["credit_limit"] / g["credit_limit"].shift(6),
        "months_since_limit_up": months_since_up,
        "util_after_limit_up": util * limit_up.astype(float),
        "dpd": df["dpd"].astype(float),
        "dpd_rising": (df["dpd"] - g["dpd"].shift(1) > 0).astype(float),
        "util_vol_6m": g["utilization"].transform(
            lambda s: s.rolling(6, min_periods=2).std()),
        "months_on_book": df["month_index"].astype(float),
    }
    # payment_ratio_drop: how far payment fell from the account's prior full-pay behaviour
    prior_pay_mean = (g["payments"].cumsum() - df["payments"]) / g.cumcount().replace(0, np.nan)
    prior_ratio = (prior_pay_mean / (prev_bal + 1.0)).clip(0, 2)
    feats["payment_ratio_drop"] = (prior_ratio - payment_ratio).clip(lower=0).fillna(0)

    feat_df = pd.DataFrame(feats, index=df.index)

    fills = {
        "util_slope_3m": 0.0, "util_accel": 0.0, "util_jump_over_prior_peak": 0.0,
        "payment_ratio": 1.0, "full_pay_streak_prior": 0.0,
        "spend_slope_3m": 0.0, "limit_growth_6m": 1.0,
        "months_since_limit_up": df["month_index"].astype(float),
        "util_vol_6m": 0.0,
    }
    for col, val in fills.items():
        feat_df[col] = feat_df[col].fillna(val)
    feat_df = feat_df.fillna(0.0)

    out = pd.concat([df.drop(columns=[c for c in feat_df.columns if c in df.columns]),
                     feat_df], axis=1)
    return out, list(feat_df.columns)
