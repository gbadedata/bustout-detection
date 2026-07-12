"""Tests for the bust-out panel and feature layer.

The load-bearing test is leakage-safety: features computed on a month-prefix of the panel
must match the features on the full panel for those same rows, so no account is scored with
knowledge of a later statement. The rest check that the archetypes behave as intended and
that the trajectory features separate a bust-out ramp from genuine distress, which is the
whole point of the project.
"""

import numpy as np

from bustout import features, metrics, model, panel_data, scoring


def test_features_use_no_future_info():
    panel = panel_data.mock_panel(n_accounts=200, seed=3)
    full, cols = features.build_features(panel)
    trunc = panel[panel["month_index"] <= 10]
    part, _ = features.build_features(trunc)
    key = ["account_id", "month_index"]
    merged = full.merge(part[key + cols], on=key, suffixes=("", "_p"))
    for c in cols:
        assert np.allclose(merged[c].to_numpy(), merged[f"{c}_p"].to_numpy(),
                           equal_nan=True), f"future info leaked in {c}"


def test_label_is_the_prebust_window():
    panel = panel_data.add_labels(panel_data.mock_panel(n_accounts=300, seed=4), horizon=3)
    pos = panel[panel["will_bustout"] == 1]
    assert (pos["account_type"] == "bustout").all()
    assert pos["months_to_bust"].between(0, 3).all()
    bust = panel[panel["account_type"] == "bustout"]
    assert bust.groupby("account_id")["will_bustout"].max().eq(1).all()


def test_bustout_accounts_reach_maxout_then_delinquency():
    panel = panel_data.mock_panel(n_accounts=300, seed=5)
    bust = panel[panel["account_type"] == "bustout"]
    peak = bust.groupby("account_id")["utilization"].max()
    assert (peak >= 0.98).mean() > 0.95
    # the terminal balance is drawn to the limit and payment stops
    last = bust.sort_values("month_index").groupby("account_id").tail(1)
    assert (last["dpd"] > 0).mean() > 0.8


def test_all_archetypes_present():
    panel = panel_data.mock_panel(n_accounts=400, seed=6)
    assert set(panel["account_type"]) == {"good", "revolver", "distress", "bustout"}


def test_trajectory_separates_bustout_from_distress():
    panel = panel_data.add_labels(panel_data.mock_panel(n_accounts=800, seed=6), horizon=3)
    feat, _ = features.build_features(panel)
    ramp = feat[feat["will_bustout"] == 1]
    distress = feat[(feat["account_type"] == "distress") & (feat["dpd"] > 0)]
    # bust-out was pristine, grew its limit fast, and draws cash; distress pays minimums
    assert ramp["full_pay_streak_prior"].mean() > distress["full_pay_streak_prior"].mean()
    assert ramp["cash_advance_share"].mean() > distress["cash_advance_share"].mean()
    assert ramp["limit_growth_ratio"].mean() > distress["limit_growth_ratio"].mean()
    assert distress["min_pay_streak"].mean() > ramp["min_pay_streak"].mean()


def _fit_and_score(seed=8, n=900):
    panel = panel_data.add_labels(panel_data.mock_panel(n_accounts=n, seed=seed), horizon=3)
    feat, cols = features.build_features(panel)
    train, test, _ = model.time_split(feat, frac=0.55)
    train = train[model.scoreable(train)]
    test = test[model.scoreable(test)].copy()
    clf = model.train_model(train, cols, seed=0)
    prob = model.score_model(clf, test, cols)
    test["bustout_prob"] = prob.to_numpy()
    test["exposure_at_risk"] = scoring.exposure_at_risk(test).to_numpy()
    test["expected_loss"] = (test["bustout_prob"] * test["exposure_at_risk"]).to_numpy()
    return test, prob


def test_scoreable_excludes_post_bust_only():
    panel = panel_data.add_labels(panel_data.mock_panel(n_accounts=300, seed=5))
    m = model.scoreable(panel)
    assert (panel.loc[~m, "months_to_bust"] < 0).all()
    assert m[panel["account_type"] != "bustout"].all()


def test_time_split_is_chronological():
    panel = panel_data.add_labels(panel_data.mock_panel(n_accounts=200, seed=3))
    feat, _ = features.build_features(panel)
    train, test, _ = model.time_split(feat, frac=0.55)
    assert train["statement_date"].max() < test["statement_date"].min()


def test_model_beats_point_in_time_baselines():
    test, _ = _fit_and_score()
    y = test["will_bustout"]
    assert metrics.pr_auc(y, test["bustout_prob"]) > 0.7
    assert metrics.pr_auc(y, test["bustout_prob"]) > metrics.pr_auc(y, test["utilization"]) + 0.2


def test_queue_leaves_distress_alone_and_ranks_by_exposure():
    from bustout import scoring as sc
    test, prob = _fit_and_score()
    sep = metrics.separation(test, budget=0.01)
    assert sep["model"]["distress_share"] <= sep["by_utilisation"]["distress_share"]
    q = sc.build_queue(test, prob, top=10)
    assert q["expected_loss"].is_monotonic_decreasing
    assert (q["reasons"].str.len() > 0).all()
