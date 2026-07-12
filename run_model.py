"""Bust-out detection: train the early-warning model and report the evaluation.

Runs on data/panel.csv if present, else on the schema-faithful mock.

    python run_model.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from bustout import features, metrics, model, panel_data, scoring

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def load() -> pd.DataFrame:
    real = Path("data/panel.csv")
    if real.exists():
        print("Loading account-month panel from data/panel.csv ...")
        return panel_data.load_panel(real)
    print("No data/panel.csv found; using the schema-faithful mock.\n")
    return panel_data.mock_panel(seed=7)


def main() -> None:
    panel = panel_data.add_labels(load(), horizon=3)
    feat, cols = features.build_features(panel)

    train, test, cutoff = model.time_split(feat, frac=0.55)
    train = train[model.scoreable(train)]
    test = test[model.scoreable(test)].copy()
    print(f"Time split at {cutoff.date()}: train {len(train):,} months, test {len(test):,} "
          f"months (post-bust charge-off months excluded).")
    print(f"Test label rate (bust within 3 months): {test['will_bustout'].mean():.2%}\n")

    clf = model.train_model(train, cols, seed=0)
    test["bustout_prob"] = model.score_model(clf, test, cols).to_numpy()
    test["exposure_at_risk"] = scoring.exposure_at_risk(test).to_numpy()
    test["expected_loss"] = (test["bustout_prob"] * test["exposure_at_risk"]).to_numpy()

    y = test["will_bustout"]
    print("Ranking quality (average precision):")
    print(f"  base rate                 {metrics.base_rate(y):.3f}")
    print(f"  by utilisation (baseline) {metrics.pr_auc(y, test['utilization']):.3f}")
    print(f"  by delinquency (baseline) {metrics.pr_auc(y, test['dpd']):.3f}")
    print(f"  trajectory model          {metrics.pr_auc(y, test['bustout_prob']):.3f}\n")

    print("Early detection (how many months before the max-out the model flags an account):")
    for thr, name in [(scoring.REDUCE_P, "reduce-line threshold"),
                      (scoring.FREEZE_P, "freeze threshold")]:
        lt = metrics.lead_time(test, threshold=thr)
        print(f"  at the {name} ({thr:.2f}): "
              f"{lt['caught_by_bust_pct']:.0%} of bust-outs caught, "
              f"{lt['caught_before_bust_pct']:.0%} before the draw, "
              f"median lead {lt['median_lead_months']:.0f} months "
              f"(over {lt['accounts']} test accounts)")
    print()

    print("Exposure caught by working the top of the queue each month:")
    eb = metrics.exposure_at_budget(test)
    for _, r in eb.iterrows():
        print(f"  top {r['budget']:.0%} ({int(r['alerts'])} alerts): "
              f"{r['exposure_caught_pct']:.0%} of bust-out exposure, "
              f"{r['accounts_caught_pct']:.0%} of bust-out accounts")
    print()

    sep = metrics.separation(test, budget=0.01)
    print(f"Separation from distress, top {sep['budget']:.0%} of the queue ({sep['alerts']} "
          f"flags):")
    print(f"  {'queue':22s} {'bust-out precision':>18s} {'distress share':>15s}")
    for name in ["model", "by_utilisation", "by_delinquency"]:
        print(f"  {name:22s} {sep[name]['bustout_precision']:>18.0%} "
              f"{sep[name]['distress_share']:>15.0%}")
    print("\n  A queue ranked on utilisation or delinquency fills with honest distress; the\n"
          "  trajectory model flags bust-outs and largely leaves distress alone.\n")

    q = scoring.build_queue(test, test["bustout_prob"], top=8)
    show = ["account_id", "month_index", "utilization", "bustout_prob", "exposure_at_risk",
            "action", "reasons"]
    print("Top of the freeze queue (by expected loss):")
    print(q[show].to_string(index=False))


if __name__ == "__main__":
    main()
