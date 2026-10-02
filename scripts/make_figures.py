"""Render the two figures the post argues from, straight out of results.json.

Nothing here is illustrative: every bar and point is read from data/results.json at
build time, so the figure cannot drift from the measurement it depicts.

Usage: python scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "assets"
OUT.mkdir(exist_ok=True)

results = json.loads((REPO / "data" / "results.json").read_text(encoding="utf-8"))
rows = {r["model"]: r for r in results["holdout_table"]}

INK = "#1a1c22"
MUTED = "#787c84"
BLUE = "#8bb2de"
RED = "#e53927"
FOREST = "#3d5f58"
PAPER = "#f7f7f2"

FOLK = "FOLK BELIEF: loudest-first (leaky features)"
TRIVIAL = "always-alive (trivial)"
N_LABELS = "single best feature: n_labels"
LOGISTIC = "logistic, all triage-time features"
TABPFN = "TabPFN v2, triage-time features only"

plt.rcParams.update(
    {
        "figure.facecolor": INK,
        "axes.facecolor": INK,
        "text.color": PAPER,
        "axes.labelcolor": PAPER,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": "#3a3f4a",
        "font.family": "DejaVu Sans",
    }
)


def percent(value, _pos):
    return f"{value:.0%}"


def precision_figure():
    labels = [
        "loudest-first\n(the rule)",
        "pick at\nrandom",
        "n_labels\nalone",
        "logistic\n46 features",
        "TabPFN v2",
    ]
    values = [rows[k]["precision_at_100"] for k in (FOLK, TRIVIAL, N_LABELS, LOGISTIC, TABPFN)]
    colours = [RED, RED, BLUE, BLUE, FOREST]

    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=110)
    bars = ax.bar(labels, values, color=colours, width=0.62)

    baseline = rows[TRIVIAL]["precision_at_100"]
    ax.axhline(baseline, color=MUTED, linestyle="--", linewidth=1.2)
    ax.text(4.45, baseline + 0.014, "random baseline", color=MUTED, fontsize=10, ha="right")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.016,
            f"{value:.3f}",
            ha="center",
            color=PAPER,
            fontsize=13,
            fontweight="bold",
        )

    ax.set_ylim(0, 0.78)
    ax.set_ylabel("precision@100")
    ax.set_title(
        "Duds found in your top 100 issues, held-out future period",
        color=PAPER,
        fontsize=15,
        fontweight="bold",
        loc="left",
        pad=30,
    )
    ax.text(
        0,
        1.015,
        f"{results['corpus']['n_closed']:,} closed issues"
        f"  |  base rate {rows[TABPFN]['base_rate']:.3f}"
        "  |  600 test issues",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10.5,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path = OUT / "precision-at-100.png"
    fig.savefig(path, facecolor=INK)
    plt.close(fig)
    return path


def body_length_figure():
    curve = results["h3_body_length_curve"]
    labels = [
        "0 (empty)" if row["body_len_min"] == 0 else f"{row['body_len_min']}\u2013{row['body_len_max']}"
        for row in curve
    ]
    rates = [row["dead_rate"] for row in curve]
    counts = [row["n"] for row in curve]

    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=110)
    ax.plot(labels, rates, color=BLUE, marker="o", markersize=9, linewidth=2.4)

    for label, rate, count in zip(labels, rates, counts):
        ax.annotate(
            f"{rate:.1%}",
            (label, rate),
            textcoords="offset points",
            xytext=(0, 13),
            ha="center",
            color=PAPER,
            fontsize=11.5,
            fontweight="bold",
        )
        ax.annotate(
            f"n={count:,}",
            (label, rate),
            textcoords="offset points",
            xytext=(0, -22),
            ha="center",
            color=MUTED,
            fontsize=9.5,
        )

    ratio = curve[2]["dead_rate"] / curve[0]["dead_rate"]
    ax.set_title(
        "An empty issue body is the strongest signal, and it is free",
        color=PAPER,
        fontsize=15,
        fontweight="bold",
        loc="left",
        pad=30,
    )
    ax.text(
        0,
        1.015,
        f"{ratio:.1f}x less likely to ever ship than a 201\u2013500 character issue"
        "  |  no model, no GPU",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10.5,
    )
    ax.set_ylabel("closed without shipping a fix")
    ax.yaxis.set_major_formatter(percent)
    ax.set_ylim(0, max(rates) * 1.35)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    path = OUT / "body-length.png"
    fig.savefig(path, facecolor=INK)
    plt.close(fig)
    return path


if __name__ == "__main__":
    for built in (precision_figure(), body_length_figure()):
        print(f"{built.name}  {built.stat().st_size / 1024:.0f} KB")