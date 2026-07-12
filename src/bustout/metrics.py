"""Evaluation for bust-out detection.

Four things are worth measuring, and the last is the one that matters most here:

  pr_auc              ranking quality against a rare label, versus the base rate and versus
                      point-in-time baselines (rank by utilisation, rank by delinquency).

  lead_time           how many months before the max-out the model first flags an account,
                      because the value is in acting during the ramp, not at the draw.

  exposure_at_budget  the share of bust-out exposure caught if a team works the top slice of
                      the ranked queue each month.

  separation          of the accounts flagged, how many are genuine bust-out and how many
                      are honest distress. A point-in-time queue fills with distress; a
                      trajectory model should not. This is reported as bust-out precision and
                      the distress share of flags, model against baseline at a matched budget.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def pr_auc(y_true, score) -> float:
    return float(average_precision_score(y_true, score))


def base_rate(y_true) -> float:
    return float(np.mean(y_true))


def lead_time(scored: pd.DataFrame, threshold: float, label: str = "will_bustout") -> dict:
    """Per bust-out account, the earliest pre-bust month flagged; months of warning it buys."""
    bust = scored[scored["account_type"] == "bustout"]
    leads, caught_before, caught_any = [], 0, 0
    n = 0
    for _, g in bust.groupby("account_id"):
        pre = g[g["months_to_bust"] >= 0]
        if pre.empty:
            continue  # this account busted before the test window
        n += 1
        flagged = pre[pre["bustout_prob"] >= threshold]
        if flagged.empty:
            continue
        earliest = float(flagged["months_to_bust"].max())  # largest months-to-bust = earliest
        caught_any += 1
        if earliest >= 1:
            caught_before += 1
            leads.append(earliest)
    return {
        "accounts": n,
        "caught_before_bust_pct": caught_before / n if n else 0.0,
        "caught_by_bust_pct": caught_any / n if n else 0.0,
        "median_lead_months": float(np.median(leads)) if leads else 0.0,
    }


def exposure_at_budget(scored: pd.DataFrame, budgets=(0.01, 0.02, 0.05),
                       label: str = "will_bustout") -> pd.DataFrame:
    """Share of bust-out exposure and accounts caught by working the top of the queue."""
    s = scored.sort_values("expected_loss", ascending=False).reset_index(drop=True)
    pos = s[label] == 1
    total_exp = s.loc[pos, "exposure_at_risk"].sum()
    total_accts = s.loc[pos, "account_id"].nunique()
    rows = []
    for b in budgets:
        k = max(1, int(len(s) * b))
        head = s.head(k)
        hp = head[label] == 1
        rows.append({
            "budget": b,
            "alerts": k,
            "exposure_caught_pct": head.loc[hp, "exposure_at_risk"].sum() / total_exp
            if total_exp else 0.0,
            "accounts_caught_pct": head.loc[hp, "account_id"].nunique() / total_accts
            if total_accts else 0.0,
        })
    return pd.DataFrame(rows)


def separation(scored: pd.DataFrame, budget: float = 0.01, label: str = "will_bustout") -> dict:
    """Top-budget flags: bust-out precision and distress share, model against baselines."""
    n = len(scored)
    k = max(1, int(n * budget))
    out = {"budget": budget, "alerts": k}
    for name, col in [("model", "bustout_prob"),
                      ("by_utilisation", "utilization"),
                      ("by_delinquency", "dpd")]:
        head = scored.sort_values(col, ascending=False).head(k)
        out[name] = {
            "bustout_precision": float((head[label] == 1).mean()),
            "distress_share": float((head["account_type"] == "distress").mean()),
        }
    return out
