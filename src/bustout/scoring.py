"""Turn a bust-out score into a monthly decision and an exposure-ranked freeze queue.

The action is deliberately coarse: monitor, reduce the line, or freeze it. The queue ranks
by exposure at risk, the undrawn line a bust-out would draw at the max-out, because the
point of acting early is to cut that exposure before it is taken. Reasons are written from
the trajectory, and the one that separates a bust-out from genuine distress is a clean,
full-paying history that has just broken.
"""

from __future__ import annotations

import pandas as pd

FREEZE_P = 0.50
REDUCE_P = 0.20


def exposure_at_risk(df: pd.DataFrame) -> pd.Series:
    """Undrawn line a bust-out would draw at the max-out."""
    return (df["credit_limit"] - df["balance"]).clip(lower=0.0)


def decide(prob: pd.Series) -> pd.Series:
    action = pd.Series("monitor", index=prob.index)
    action[prob >= REDUCE_P] = "reduce_line"
    action[prob >= FREEZE_P] = "freeze"
    return action


def _reasons(row: pd.Series) -> str:
    r = []
    if row.get("util_slope_3m", 0) > 0.15:
        r.append("utilisation climbing fast")
    if row.get("util_jump_over_prior_peak", 0) > 0.15:
        r.append("far above its usual level")
    if row.get("full_pay_streak_prior", 0) >= 2 and row.get("payment_ratio", 1) < 0.5:
        r.append("clean payment history just broke")
    elif row.get("payment_ratio_drop", 0) > 0.2:
        r.append("payments dropped off")
    if row.get("cash_advance_share", 0) > 0.10:
        r.append("cash draws on the line")
    if row.get("limit_growth_ratio", 1) > 1.4 and row.get("months_since_limit_up", 99) <= 3:
        r.append("line grew fast and recently")
    if row.get("dpd_rising", 0) > 0.5 and row.get("dpd", 0) > 0:
        r.append("falling behind on payments")
    return "; ".join(r[:3]) if r else "elevated model score"


def build_queue(df: pd.DataFrame, prob: pd.Series, top: int | None = 25) -> pd.DataFrame:
    """Rank scored account-months by expected loss, with an action and reasons."""
    q = df.copy()
    q["bustout_prob"] = prob.to_numpy()
    q["exposure_at_risk"] = exposure_at_risk(q).to_numpy()
    q["expected_loss"] = (q["bustout_prob"] * q["exposure_at_risk"]).to_numpy()
    q["action"] = decide(q["bustout_prob"]).to_numpy()
    q["reasons"] = q.apply(_reasons, axis=1)
    cols = ["account_id", "month_index", "statement_date", "credit_limit", "balance",
            "utilization", "bustout_prob", "exposure_at_risk", "expected_loss",
            "action", "reasons"]
    cols = [c for c in cols if c in q.columns]
    out = q.sort_values("expected_loss", ascending=False)[cols].reset_index(drop=True)
    return out.head(top) if top else out
