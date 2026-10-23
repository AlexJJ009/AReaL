"""Capability coverage of the tau2 training set vs the test split (dumbbell plot).

Input is the `counts.json` written by the coverage study (pueue 685, Codex):
`_artifacts/AReaL/codex-sao-math/tau2-train-test-coverage-20261022/out/counts.json`,
copied to `docs/research/tau2/figures/coverage_counts.json`. For every capability
tag the plot shows the share of tasks carrying it in the current training set,
the test split and one candidate pool; the right-hand columns give task counts.

    python docs/research/tau2/scripts/figures/plot_coverage.py
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Palette from figures/user_sim_retail.tex (same as plot_results.py).
C_TRAIN, C_POOL, C_TEST = "#2A78D6", "#8E8C84", "#EB6834"
INK, MUTED, GRID = "#222222", "#6B6B6B", "#E6E6E6"

GROUP_ORDER = {
    "airline": ["Outcome", "Refusal", "Information", "Write", "Write subtype",
                "Payment", "Complexity"],
    "retail": ["Outcome", "Refusal", "Information", "Write", "Payment",
               "Complexity"],
    "telecom": ["Issue", "Account action", "Fault", "Complexity", "Persona"],
}
POOL = {
    "airline": "pool_areal_v1",
    "retail": "pool_areal_v1",
    "telecom": "pool_telecom_full",
}
POOL_LABEL = {
    "pool_areal_v1": "Pool: AReaL pack v1",
    "pool_telecom_full": "Pool: telecom_full",
}
TITLE = {"airline": "Airline", "retail": "Retail", "telecom": "Telecom"}

# A row is a gap when test needs the capability much more often than train has it.
GAP_PP = 15.0


def share(sets, key, tag):
    n = sets[key]["n"]
    return 100.0 * sets[key]["counts"].get(tag, 0) / n if n else float("nan")


def ordered_rows(domain, info):
    sets = info["sets"]
    groups = {}
    for t in info["tags"]:
        groups.setdefault(t["group"], []).append(t)
    order = GROUP_ORDER[domain] + [g for g in groups if g not in GROUP_ORDER[domain]]
    rows = []
    for g in order:
        if g not in groups:
            continue
        tags = sorted(
            groups[g],
            key=lambda t: share(sets, "current_train", t["tag"])
            - share(sets, "test", t["tag"]),
        )
        rows.append((g, tags))
    return rows


def plot_domain(domain, info, out_dir):
    sets = info["sets"]
    pool = POOL[domain]
    rows = ordered_rows(domain, info)
    n_rows = sum(len(t) for _, t in rows) + len(rows)
    fig, ax = plt.subplots(figsize=(7.4, 0.24 * n_rows + 1.3))

    y, ticks, labels, gap_flags = 0, [], [], []
    for group, tags in rows:
        ax.text(-0.02, y, group, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=7.5, fontweight="bold", color=MUTED)
        y += 1
        for t in tags:
            tag = t["tag"]
            s_tr, s_te, s_po = (share(sets, k, tag) for k in ("current_train", "test", pool))
            ax.plot([s_tr, s_te], [y, y], color="#C9C9C9", lw=1.6, zorder=1)
            ax.scatter(s_po, y, marker="D", s=16, facecolor="white",
                       edgecolor=C_POOL, lw=1.0, zorder=2)
            # Train is drawn larger so it stays visible when both shares coincide.
            ax.scatter(s_tr, y, s=40, color=C_TRAIN, zorder=3, edgecolor="white", lw=0.5)
            ax.scatter(s_te, y, s=20, color=C_TEST, zorder=4, edgecolor="white", lw=0.5)
            counts = [sets[k]["counts"].get(tag, 0) for k in ("current_train", "test", pool)]
            for x, c, color in zip((1.05, 1.18, 1.32), counts, (C_TRAIN, C_TEST, C_POOL)):
                ax.text(x, y, str(c), transform=ax.get_yaxis_transform(), ha="center",
                        va="center", fontsize=6.8,
                        color=INK if c else color, fontweight="normal" if c else "bold")
            ticks.append(y)
            labels.append(t["label"])
            gap_flags.append(s_te - s_tr >= GAP_PP or (counts[0] == 0 and counts[1] > 0))
            y += 1

    n_tr, n_te, n_po = (sets[k]["n"] for k in ("current_train", "test", pool))
    for x, head in zip((1.05, 1.18, 1.32), (f"Train\nn={n_tr}", f"Test\nn={n_te}",
                                            f"Pool\nn={n_po}")):
        ax.text(x, -0.9, head, transform=ax.get_yaxis_transform(), ha="center",
                va="bottom", fontsize=6.8, color=MUTED)

    ax.set_yticks(ticks, labels, fontsize=7.5)
    for tick, gap in zip(ax.get_yticklabels(), gap_flags):
        if gap:
            tick.set_color(C_TEST)
            tick.set_fontweight("bold")
    ax.set_ylim(y - 0.4, -1.0)
    ax.set_xlim(-2, 102)
    ax.set_xlabel("Share of tasks requiring the capability (%)", fontsize=8)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=C_TRAIN, label="Train (run 3)"),
        plt.Line2D([], [], marker="o", ls="", color=C_TEST, label="Test"),
        plt.Line2D([], [], marker="D", ls="", markerfacecolor="white",
                   markeredgecolor=C_POOL, label=POOL_LABEL[pool]),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3,
              frameon=False, fontsize=7.5, handletextpad=0.3, columnspacing=1.2)
    ax.set_title(f"{TITLE[domain]}: capability coverage", fontsize=9.5,
                 fontweight="bold", loc="left", pad=24)
    fig.text(0.01, 0.005,
             f"Orange labels: test share exceeds train share by ≥{GAP_PP:.0f} pp, or "
             "absent from train. Tags co-occur; shares do not sum to 100%.",
             fontsize=6.5, color=MUTED)
    fig.subplots_adjust(left=0.34, right=0.76, top=1 - 0.75 / fig.get_figheight(),
                        bottom=0.55 / fig.get_figheight())
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"coverage_{domain}.{ext}", dpi=220)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="docs/research/tau2/figures/coverage_counts.json")
    ap.add_argument("--out-dir", default="docs/research/tau2/figures")
    args = ap.parse_args()
    with open(args.counts) as f:
        counts = json.load(f)
    out_dir = Path(args.out_dir)
    for domain, info in counts["domains"].items():
        plot_domain(domain, info, out_dir)


if __name__ == "__main__":
    main()
