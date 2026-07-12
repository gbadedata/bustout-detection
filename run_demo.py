"""Bust-out detection demo: the panel, the label, and the trajectory signal.

Runs on a real account-month panel at data/panel.csv if present, else on the mock.

    python run_demo.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bustout import features, panel_data


def load() -> pd.DataFrame:
    real = Path("data/panel.csv")
    if real.exists():
        print("Loading account-month panel from data/panel.csv ...")
        return panel_data.load_panel(real)
    print("No data/panel.csv found; using the schema-faithful mock.\n")
    return panel_data.mock_panel(seed=7)


def main() -> None:
    panel = panel_data.add_labels(load(), horizon=3)
    accts = panel["account_id"].nunique()
    print(f"  {len(panel):,} account-months | {accts} accounts | "
          f"statements {panel['statement_date'].min().date()} to "
          f"{panel['statement_date'].max().date()}")
    if "account_type" in panel.columns:
        mix = panel.drop_duplicates("account_id")["account_type"].value_counts()
        print("  accounts by type: " + ", ".join(f"{k} {v}" for k, v in mix.items()))
    print(f"  bust-out within 3 months (label rate): {panel['will_bustout'].mean():.2%}\n")

    feat, cols = features.build_features(panel)
    print(f"Leakage-safe trajectory features: {len(cols)} (no account's future is used).\n")

    if "account_type" in panel.columns:
        ramp = feat[feat["will_bustout"] == 1]
        distress = feat[(feat["account_type"] == "distress") & (feat["dpd"] > 0)]
        good = feat[feat["account_type"] == "good"]
        print("The confound and the signal (mean over the relevant account-months):")
        print(f"  {'feature':26s} {'bust ramp':>10s} {'distress':>9s} {'good':>7s}")
        show = ["utilization", "util_slope_3m", "util_jump_over_prior_peak",
                "full_pay_streak_prior", "min_pay_streak", "cash_advance_share",
                "spend_slope_3m", "limit_growth_ratio", "months_since_limit_up"]
        for c in show:
            print(f"  {c:26s} {ramp[c].mean():>10.3f} {distress[c].mean():>9.3f} "
                  f"{good[c].mean():>7.3f}")
        print("\nUtilisation alone points the wrong way: at the terminal month a distress "
              "account\nsits higher than a bust-out ramp. The separation is in the path, a "
              "pristine\nfull-paying history that breaks, cash draws, and fast limit growth, "
              "which is what\nthe model and the cost-ranked freeze queue are built on next.")


if __name__ == "__main__":
    main()
