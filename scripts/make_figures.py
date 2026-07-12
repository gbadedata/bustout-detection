"""Generate the figures used in the README, from the mock pipeline.

    python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bustout import features, metrics, model, panel_data, scoring

BLUE, RED, PURPLE, GREY, GREEN = "#4C72B0", "#C44E52", "#8172B3", "#B0B0B0", "#55A868"
OUT = Path("docs/img")


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _pipeline():
    panel = panel_data.add_labels(panel_data.mock_panel(seed=7), horizon=3)
    feat, cols = features.build_features(panel)
    train, test, _ = model.time_split(feat, frac=0.55)
    train = train[model.scoreable(train)]
    test = test[model.scoreable(test)].copy()
    clf = model.train_model(train, cols, seed=0)
    test["bustout_prob"] = model.score_model(clf, test, cols).to_numpy()
    test["exposure_at_risk"] = scoring.exposure_at_risk(test).to_numpy()
    test["expected_loss"] = (test["bustout_prob"] * test["exposure_at_risk"]).to_numpy()
    return panel, test


def fig_trajectory(panel):
    """Same endpoint, different path: a bust-out against genuine distress and a good account."""
    def pick(atype, cond):
        for _aid, g in panel[panel["account_type"] == atype].groupby("account_id"):
            if cond(g):
                return g.sort_values("month_index")
        return None

    bust = pick("bustout", lambda g: (g["months_to_bust"] == 0).any() and len(g) >= 10)
    dist = pick("distress", lambda g: g["utilization"].max() > 0.9 and len(g) >= 14)
    good = pick("good", lambda g: len(g) >= 14)
    bust_m = int(bust.loc[bust["months_to_bust"] == 0, "month_index"].iloc[0])

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    series = [("Utilisation", "utilization", (0, 1.05)),
              ("Payment / prior balance", None, (0, 1.3)),
              ("Credit limit", "credit_limit", None)]
    for ax, (title, col, ylim) in zip(axes, series, strict=True):
        for g, c, lab in [(good, GREY, "good"), (dist, BLUE, "distress"), (bust, RED, "bust-out")]:
            if col is None:
                prev = g["balance"].shift(1)
                yv = (g["payments"] / (prev + 1)).clip(0, 1.3)
            else:
                yv = g[col]
            ax.plot(g["month_index"], yv, color=c, linewidth=2.2, label=lab, zorder=3)
        ax.axvline(bust_m, color=RED, linestyle=":", linewidth=1.3, zorder=2)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("month on book")
        if ylim:
            ax.set_ylim(*ylim)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle("The confound: distress and bust-out reach the same place by different paths",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "trajectory_confound.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_separation(test):
    """Precision and distress share of the top 1% of the queue, model vs point-in-time."""
    sep = metrics.separation(test, budget=0.01)
    names = ["model", "by_utilisation", "by_delinquency"]
    labels = ["trajectory\nmodel", "ranked by\nutilisation", "ranked by\ndelinquency"]
    prec = [sep[n]["bustout_precision"] * 100 for n in names]
    dist = [sep[n]["distress_share"] * 100 for n in names]

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(x - w / 2, prec, w, color=GREEN, label="bust-out precision", zorder=3)
    ax.bar(x + w / 2, dist, w, color=RED, label="honest distress wrongly flagged", zorder=3)
    for xi, p, d in zip(x, prec, dist, strict=True):
        ax.text(xi - w / 2, p + 1.5, f"{p:.0f}%", ha="center", fontsize=9)
        ax.text(xi + w / 2, d + 1.5, f"{d:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("share of the flagged accounts (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Who fills the queue: bust-outs, or honest customers in distress", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "queue_separation.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_lead_time(test, threshold=0.5):
    """Cumulative share of bust-out accounts flagged, by months before the draw."""
    bust = test[test["account_type"] == "bustout"]
    earliest = []
    for _, g in bust.groupby("account_id"):
        pre = g[g["months_to_bust"] >= 0]
        if pre.empty:
            continue
        f = pre[pre["bustout_prob"] >= threshold]
        earliest.append(float(f["months_to_bust"].max()) if not f.empty else -1)
    earliest = np.array(earliest)
    n = len(earliest)
    leads = [3, 2, 1, 0]
    frac = [(earliest >= L).mean() * 100 for L in leads]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(range(len(leads)), frac, color=PURPLE, linewidth=2.4, marker="o",
            markersize=7, zorder=3)
    for i, f in enumerate(frac):
        ax.text(i, f + 2, f"{f:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(range(len(leads)))
    ax.set_xticklabels([f"{L} months\nbefore draw" if L else "at the\ndraw" for L in leads])
    ax.set_ylabel("bust-out accounts flagged (%)")
    ax.set_ylim(0, 105)
    ax.set_title(f"Early detection: caught before the max-out (over {n} test accounts)",
                 fontsize=12)
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "lead_time.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_exposure_budget(test):
    """Exposure caught against review budget, model queue vs a utilisation-ranked queue."""
    pos = test["will_bustout"] == 1
    total_exp = test.loc[pos, "exposure_at_risk"].sum()
    budgets = np.linspace(0.002, 0.06, 30)

    def curve(rank_col):
        s = test.sort_values(rank_col, ascending=False).reset_index(drop=True)
        hp = (s["will_bustout"] == 1).to_numpy()
        exp = s["exposure_at_risk"].to_numpy()
        out = []
        for b in budgets:
            k = max(1, int(len(s) * b))
            out.append(exp[:k][hp[:k]].sum() / total_exp * 100)
        return out

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(budgets * 100, curve("expected_loss"), color=GREEN, linewidth=2.4,
            label="trajectory model queue", zorder=3)
    ax.plot(budgets * 100, curve("utilization"), color=GREY, linewidth=2.2,
            linestyle="--", label="ranked by utilisation", zorder=3)
    ax.set_xlabel("review budget (share of account-months worked)")
    ax.set_ylabel("bust-out exposure caught (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Exposure caught for the effort spent", fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    _style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "exposure_budget.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel, test = _pipeline()
    fig_trajectory(panel)
    fig_separation(test)
    fig_lead_time(test)
    fig_exposure_budget(test)
    print(f"wrote figures to {OUT}/")


if __name__ == "__main__":
    main()
