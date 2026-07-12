"""Account-month panel data for bust-out detection: a schema-faithful mock, and a loader.

No lender publishes labelled bust-out data, so the mock is the primary source and the
write-ups say so. It simulates a credit line month by month for four kinds of account,
because the point of the project is telling them apart:

  good       moderate use, pays well, occasional limit increase, never busts
  revolver   carries a balance at higher utilisation but pays reliably (a confound)
  distress   genuine hardship: utilisation climbs gradually, payments taper, spend falls,
             delinquency follows, but the account keeps paying something and never spikes
  bustout    cultivated to look excellent (full payments, low utilisation, fast limit
             growth), then maxes the whole line, draws cash, and stops paying abruptly

The distress archetype is the one that matters. A model that only learns "high utilisation
and missed payments" flags distress and bust-out alike; the separable signal is in the
trajectory, and the generator is built to make that difference real rather than trivial.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REF_DATE = pd.Timestamp("2023-01-01")
TYPES = ("good", "revolver", "distress", "bustout")
_MIX = (0.68, 0.10, 0.14, 0.08)

PANEL_COLS = ["account_id", "month_index", "statement_date", "account_type",
              "credit_limit", "purchases", "cash_advance", "payments", "balance",
              "min_payment_due", "dpd", "utilization", "months_to_bust"]


def _min_due(bal: float) -> float:
    return round(max(25.0, 0.02 * bal), 2)


def simulate_account(aid: int, atype: str, rng, max_months: int = 24) -> list[dict]:
    open_m = int(rng.integers(0, 12))
    limit = float(rng.choice([2000, 3000, 5000, 7500, 10000]))
    bal = 0.0
    dpd = 0
    since_cli = 0

    if atype == "bustout":
        cult = int(rng.integers(4, 8))
        ramp_len = int(rng.integers(3, 5))
        bust_m = cult + ramp_len
        life = min(max_months, bust_m + int(rng.integers(3, 5)))
    elif atype == "distress":
        onset = int(rng.integers(6, 12))
        life = min(max_months, onset + int(rng.integers(6, 12)))
        peak_u = float(rng.uniform(0.90, 0.99))
        base_u = float(rng.uniform(0.20, 0.35))
        bust_m = None
    else:
        onset = None
        life = max_months
        bust_m = None

    good_base = float(rng.uniform(0.12, 0.30))
    rev_base = float(rng.uniform(0.45, 0.70))
    cult_base = float(rng.uniform(0.05, 0.20))

    rows = []
    for k in range(life):
        prev_bal = bal
        cash = 0.0
        did_cli = False
        min_due = _min_due(prev_bal)

        # payment behaviour and this month's target utilisation, by archetype
        if atype == "good":
            payment = prev_bal if rng.random() < 0.6 else prev_bal * rng.uniform(0.5, 0.95)
            target_u = float(np.clip(good_base + rng.normal(0, 0.03), 0.03, 0.5))
            if since_cli >= rng.integers(8, 13) and prev_bal / limit < 0.5:
                limit *= 1.30
                since_cli = 0
                did_cli = True
        elif atype == "revolver":
            payment = max(min_due, prev_bal * rng.uniform(0.20, 0.55))
            target_u = float(np.clip(rev_base + rng.normal(0, 0.04), 0.3, 0.8))
            if since_cli >= rng.integers(12, 16) and prev_bal / limit < 0.6:
                limit *= 1.25
                since_cli = 0
                did_cli = True
        elif atype == "distress":
            if k < onset:
                payment = prev_bal if rng.random() < 0.4 else prev_bal * rng.uniform(0.5, 0.9)
                target_u = float(np.clip(base_u + rng.normal(0, 0.03), 0.05, 0.5))
            else:
                step = (k - onset) / max(life - onset - 1, 1)
                target_u = float(np.clip(base_u + (peak_u - base_u) * step, 0.05, 0.99))
                pay_frac = max(0.0, 0.22 * (1 - step))  # payments taper below the minimum
                payment = prev_bal * pay_frac
        else:  # bustout
            if k < cult:
                payment = prev_bal  # pristine: pays in full every cultivation month
                target_u = float(np.clip(cult_base + 0.015 * k + rng.normal(0, 0.02), 0.03, 0.4))
                if since_cli >= int(rng.integers(3, 5)) and prev_bal / limit < 0.4:
                    limit *= rng.uniform(1.4, 1.7)  # fast limit growth on clean behaviour
                    since_cli = 0
                    did_cli = True
            elif k < bust_m:  # ramp: utilisation creeps up, payments shift off full
                j = k - cult
                step = (j + 1) / ramp_len
                target_u = float(np.clip(cult_base + (0.85 - cult_base) * step
                                         + rng.normal(0, 0.03), 0.05, 0.95))
                payment = prev_bal * max(0.0, 1.0 - 0.85 * step)
                if j == 0 and rng.random() < 0.5 and prev_bal / limit < 0.5:
                    limit *= rng.uniform(1.3, 1.6)  # grabs more limit right before the spree
                    since_cli = 0
                    did_cli = True
            elif k == bust_m:
                cash = rng.uniform(0.20, 0.50) * limit
                payment = 0.0 if rng.random() < 0.7 else prev_bal  # sometimes a kite, then re-max
                target_u = 1.0
            else:
                payment = 0.0
                target_u = float(prev_bal / limit)

        target_bal = target_u * limit
        purchases = max(0.0, target_bal - prev_bal + payment - cash)
        bal = float(np.clip(prev_bal + purchases + cash - payment, 0.0, limit))

        if payment + 1e-6 >= min_due or prev_bal <= 0:
            dpd = 0
        else:
            dpd = min(dpd + 30, 120)

        since_cli += 1
        rows.append({
            "account_id": f"A{aid:05d}",
            "month_index": k,
            "statement_date": REF_DATE + pd.DateOffset(months=open_m + k),
            "account_type": atype,
            "credit_limit": round(limit, 2),
            "purchases": round(float(purchases), 2),
            "cash_advance": round(float(cash), 2),
            "payments": round(float(payment), 2),
            "balance": round(float(bal), 2),
            "min_payment_due": min_due,
            "dpd": int(dpd),
            "utilization": round(float(bal / limit), 4),
            "months_to_bust": (bust_m - k) if bust_m is not None else np.nan,
            "_cli": int(did_cli),
        })
    return rows


def mock_panel(n_accounts: int = 1200, seed: int = 7, max_months: int = 24) -> pd.DataFrame:
    """Build a schema-faithful account-month panel across the four archetypes."""
    rng = np.random.default_rng(seed)
    types = rng.choice(TYPES, size=n_accounts, p=_MIX)
    rows = []
    for aid, atype in enumerate(types):
        rows.extend(simulate_account(aid, atype, rng, max_months=max_months))
    df = pd.DataFrame(rows)
    return df.sort_values(["account_id", "month_index"]).reset_index(drop=True)


def add_labels(panel: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """Forward label: 1 if the account busts out within `horizon` months of this statement."""
    out = panel.copy()
    m2b = out["months_to_bust"]
    out["will_bustout"] = ((m2b >= 0) & (m2b <= horizon)).fillna(False).astype(int)
    return out


def load_panel(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    df = pd.read_csv(path, nrows=nrows, parse_dates=["statement_date"])
    return df.sort_values(["account_id", "month_index"]).reset_index(drop=True)


def write_mock_panel(out_dir: str | Path, **kwargs) -> Path:
    """Write a mock panel CSV, dropping the account-level labels a live feed would not have."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "mock_panel.csv"
    drop = ["account_type", "months_to_bust", "_cli"]
    mock_panel(**kwargs).drop(columns=drop).to_csv(path, index=False)
    return path
