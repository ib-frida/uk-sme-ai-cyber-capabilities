"""
make_figures.py
===============
Generates every figure in Chapter 5, named to match the \\includegraphics calls
in dissertation.tex and ordered by research question.

    RQ1   fig_bics_sector.pdf      adoption by sector, against the six-sector mean
    RQ1   fig_bics_mix.pdf         composition by technology type, suppression shown
    RQ2   fig_csbs_profile.pdf     capability and breach by size band, with 95% CIs
    RQ3   fig_gap.pdf              the naive gap G_s, with the near-balance band
    RQ3   fig_cluster.pdf          the AI-cyber capability plane
    RQ3   fig_ai_governance.pdf    V_s against C_s with the differential annotated

Design decisions, and why:

  * Categorical hues are assigned in fixed slot order and never cycled. Slots 1-4
    (blue, orange, aqua, yellow) validate for the adjacent pairlist that grouped
    bars use: worst adjacent CVD separation 9.1 (protan), worst normal-vision
    separation 22.9, both clear of the required floors.
  * Aqua and yellow fall below 3:1 contrast against the chart surface, so the
    relief rule applies: every mark in those hues carries a visible direct label.
    Identity is never colour-alone.
  * Hatching is applied on the composition chart as the print/greyscale channel,
    so the figure survives being photocopied and read by a colourblind examiner.
  * Signed quantities use the diverging pair (blue positive, red negative) with a
    neutral zero line. Magnitudes use a single hue.
  * One axis per chart. No dual-scale figure appears anywhere in this
    dissertation.
  * Grid and axis are recessive hairlines; all text sits in ink colours rather
    than in a series colour.

Usage:
    python3 make_figures.py

Author: Frida Alejandra Ibarra Zavaleta
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import load_data as ld
import stats_core as sc

mpl.use("Agg")

SEED = 42
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
FIGS = HERE.parent / "figures"
FIGS.mkdir(exist_ok=True)

BICS_PATH = DATA / "wave147dsitaiadoptionrequest.xlsx"
CSBS_PATH = DATA / "DSIT_CSBS_2025-2026_tables.xlsx"

# --- palette (light surface) ------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # categorical slots
POS, NEG, ZERO = "#2a78d6", "#d03b3b", "#f0efec"             # diverging pair

BANDS = ld.SIZE_LABELS

# Sector names are long enough to collide on a 7-inch axis; these are the
# shortest forms that remain unambiguous within the six DSIT priority sectors.
SHORT = {
    "Life Sciences": "Life\nSciences",
    "Creative Industries": "Creative\nIndustries",
    "Advanced Manufacturing": "Advanced\nManufacturing",
    "Digital and Technologies": "Digital and\nTechnologies",
    "Professional and Business Services": "Professional and\nBusiness Svcs",
    "Defence": "Defence",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "serif",
    "font.serif": ["Palatino", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.frameon": False,
    "legend.fontsize": 8.5,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.06,
    "pdf.fonttype": 42,
})


def _clean(ax, ygrid=True):
    """Recessive chrome: no top/right spine, hairline grid behind the marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    if ygrid:
        ax.set_axisbelow(True)
        ax.yaxis.grid(True)
        ax.xaxis.grid(False)


def _save(fig, name):
    path = FIGS / name
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {path.name}")


# ===========================================================================
# RQ1
# ===========================================================================

def fig_bics_sector(bics):
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    rates = bics["ai_use_rate"].to_numpy() * 100
    labels = [SHORT[str(s)] for s in bics["sector"]]
    mean6 = rates.mean()

    x = np.arange(len(rates))
    ax.bar(x, rates, width=0.62, color=S1, edgecolor=SURFACE, linewidth=1.2, zorder=3)

    # Every bar carries its value, set inside the bar. Placing the labels inside
    # rather than above keeps them clear of the six-sector mean line, which two
    # of the six bars sit within a couple of points of.
    for xi, v in zip(x, rates):
        ax.text(xi, v - 1.6, f"{v:.1f}", ha="center", va="top",
                fontsize=8.5, color="#ffffff", zorder=5)

    ax.axhline(mean6, color=INK_2, lw=1.0, ls=(0, (4, 3)), zorder=4)
    ax.text(len(rates) - 0.45, mean6 + 1.2, f"six-sector mean {mean6:.1f}%",
            ha="right", va="bottom", fontsize=8, color=INK_2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Using at least one AI technology (%)")
    ax.set_ylim(0, 60)
    _clean(ax)
    _save(fig, "fig_bics_sector.pdf")


def fig_bics_mix(bics):
    fig, ax = plt.subplots(figsize=(7.9, 3.9))
    tech = [("ai_llm_rate", "Text generation (LLM)", S1, ""),
            ("ai_ml_data_rate", "ML for data processing", S2, "///"),
            ("ai_image_ml_rate", "ML for image processing", S3, "\\\\\\"),
            ("ai_robotics_rate", "Robotics", S4, "xxx")]

    x = np.arange(len(bics))
    w = 0.20
    for k, (col, lab, colour, hatch) in enumerate(tech):
        vals = bics[col].to_numpy(dtype=float) * 100
        offs = x + (k - 1.5) * w
        drawn = np.where(np.isnan(vals), 0, vals)
        ax.bar(offs, drawn, width=w * 0.88, label=lab, color=colour,
               edgecolor=SURFACE, linewidth=1.0, hatch=hatch, zorder=3)
        for xi, v in zip(offs, vals):
            if np.isnan(v):
                # suppressed for disclosure control: marked, never imputed
                ax.text(xi, 0.7, "c", ha="center", va="bottom", fontsize=8,
                        color=MUTED, style="italic")
            else:
                ax.text(xi, v + 0.5, f"{v:.1f}", ha="center", va="bottom",
                        fontsize=6.8, color=INK_2)

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[str(s)] for s in bics["sector"]], fontsize=8)
    ax.set_ylabel("Businesses using each technology (%)")
    ax.set_ylim(0, 34)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    _clean(ax)
    _save(fig, "fig_bics_mix.pdf")


# ===========================================================================
# RQ2
# ===========================================================================

def fig_csbs_profile(ind):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    series = [("A", "AI capability $A_s$", S1, ""),
              ("C", "Cyber capability $C_s$", S2, "///"),
              ("R", "Reported breaches $R_s$", S3, "...")]

    x = np.arange(4)
    w = 0.25
    for k, (key, lab, colour, hatch) in enumerate(series):
        vals = ind[key].to_numpy() * 100
        offs = x + (k - 1) * w

        if key == "C":
            # C_s is a composite of three correlated components; the error bar
            # shows the widest defensible interval (perfect-correlation bound)
            errs = [sc.Z_95 * sc.composite_se_bounds(
                [r["T"], r["B"], r["P"]], r["n_all"])[1] * 100
                for _, r in ind.iterrows()]
        else:
            errs = [sc.moe(p, n) * 100 for p, n in zip(ind[key], ind["n_all"])]

        ax.bar(offs, vals, width=w * 0.86, label=lab, color=colour,
               edgecolor=SURFACE, linewidth=1.1, hatch=hatch, zorder=3)
        ax.errorbar(offs, vals, yerr=errs, fmt="none", ecolor=INK_2,
                    elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=4)
        for xi, v, e in zip(offs, vals, errs):
            ax.text(xi, v + e + 1.6, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK_2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n(n={int(n)})" for b, n in zip(BANDS, ind["n_all"])])
    ax.set_ylabel("Organisations (%)")
    ax.set_ylim(0, 100)
    ax.legend(ncol=3, loc="upper left")
    _clean(ax)
    ax.text(0.0, -0.30,
            "Error bars are 95% intervals on the effective base. The interval on $C_s$ is the "
            "widest defensible bound,\nsince the covariance between its three components is not "
            "recoverable from published tables.",
            transform=ax.transAxes, fontsize=7, color=MUTED, va="top")
    _save(fig, "fig_csbs_profile.pdf")


# ===========================================================================
# RQ3
# ===========================================================================

def fig_gap(ind):
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    vals = ind["G"].to_numpy() * 100
    x = np.arange(4)

    ax.axhspan(-5, 5, color=ZERO, zorder=1)
    ax.text(3.30, 0, "near balance\n$|G_s| \\leq 5$", ha="right", va="center",
            fontsize=7.5, color=MUTED)

    colours = [NEG if v < 0 else POS for v in vals]
    ax.bar(x, vals, width=0.5, color=colours, edgecolor=SURFACE,
           linewidth=1.2, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v - 1.6, f"{v:+.1f}", ha="center", va="top",
                fontsize=8.5, color=INK)

    ax.axhline(0, color=AXIS, lw=1.0, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(BANDS)
    ax.set_ylabel("$G_s = A_s - C_s$ (percentage points)")
    ax.set_ylim(-44, 12)
    _clean(ax)
    ax.text(0.0, -0.22,
            "$G_s$ sets an adoption measure against a governance measure and is not "
            "construct-valid;\nit is shown because the published tables invite it. "
            "See Figure~\\ref{fig:aigov} for the corrected comparison.".replace("~\\ref{fig:aigov}", " 5.6"),
            transform=ax.transAxes, fontsize=7, color=MUTED, va="top")
    _save(fig, "fig_gap.pdf")


def fig_cluster(ind):
    fig, ax = plt.subplots(figsize=(5.8, 5.0))

    X = ind[["T", "B", "A", "C", "R"]].to_numpy(dtype=float)
    labels = KMeans(n_clusters=2, n_init=20, random_state=SEED).fit(
        StandardScaler().fit_transform(X)).labels_

    a = ind["A"].to_numpy() * 100
    c = ind["C"].to_numpy() * 100
    r = ind["R"].to_numpy() * 100

    lim = (5, 95)
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_aspect("equal")          # so the A_s = C_s diagonal really is 45 degrees

    ax.plot(lim, lim, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.text(84, 87, "$A_s = C_s$", fontsize=8, color=MUTED, rotation=45,
            rotation_mode="anchor", ha="center", va="bottom")

    cols = {0: S1, 1: S2}
    # offsets chosen per point so no label overlaps a marker or the diagonal
    offsets = {0: (34, -6), 1: (34, -4), 2: (-40, 6), 3: (36, -2)}
    aligns = {0: "left", 1: "left", 2: "right", 3: "left"}
    for i in range(4):
        ax.scatter(a[i], c[i], s=r[i] * 7.5, color=cols[labels[i]],
                   edgecolor=SURFACE, linewidth=1.6, zorder=4, alpha=0.92)
        ax.annotate(f"{BANDS[i]}\n$R_s$ = {r[i]:.1f}%",
                    (a[i], c[i]), textcoords="offset points",
                    xytext=offsets[i], ha=aligns[i], va="center",
                    fontsize=8, color=INK, zorder=5)

    handles = [plt.Line2D([], [], marker="o", ls="", markersize=8,
                          markerfacecolor=cols[k], markeredgecolor=SURFACE,
                          label=lab)
               for k, lab in [(0, "Low capability"), (1, "High capability, cyber-led")]]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, 1.0))

    ax.set_xlabel("AI capability $A_s$ (%)")
    ax.set_ylabel("Composite cyber capability $C_s$ (%)")
    _clean(ax)
    ax.xaxis.grid(True)
    ax.text(0.0, -0.16,
            "Every band sits above the diagonal. Marker area is proportional to the reported "
            "breach rate;\ncluster membership from $k$-means, $k = 2$, four observations, "
            "exploratory only.",
            transform=ax.transAxes, fontsize=7, color=MUTED, va="top")
    _save(fig, "fig_cluster.pdf")


def fig_ai_governance(ind):
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(4)
    w = 0.32

    v = ind["V"].to_numpy() * 100
    c = ind["C"].to_numpy() * 100
    d = ind["D"].to_numpy() * 100

    ax.bar(x - w / 2, v, width=w * 0.9, label="AI-specific cyber practices $V_s$",
           color=S1, edgecolor=SURFACE, linewidth=1.1, zorder=3)
    ax.bar(x + w / 2, c, width=w * 0.9, label="General cyber capability $C_s$",
           color=S2, edgecolor=SURFACE, linewidth=1.1, hatch="///", zorder=3)

    errs_v = [sc.moe(p, n) * 100 for p, n in zip(ind["V"], ind["n_ai"])]
    errs_c = [sc.Z_95 * sc.composite_se_bounds([r["T"], r["B"], r["P"]],
                                               r["n_all"])[1] * 100
              for _, r in ind.iterrows()]
    ax.errorbar(x - w / 2, v, yerr=errs_v, fmt="none", ecolor=INK_2,
                elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=4)
    ax.errorbar(x + w / 2, c, yerr=errs_c, fmt="none", ecolor=INK_2,
                elinewidth=0.9, capsize=2.5, capthick=0.9, zorder=4)
    errs_v = np.asarray(errs_v); errs_c = np.asarray(errs_c)

    for xi, vv, cc, dd, ev, ec in zip(x, v, c, d, errs_v, errs_c):
        top = max(vv + ev, cc + ec) + 6.5
        ax.annotate("", xy=(xi - w / 2, vv + ev + 1), xytext=(xi + w / 2, cc + ec + 1),
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
        ax.text(xi, top, f"$D_s$ = {dd:+.1f} pp", ha="center", va="bottom",
                fontsize=8.5, color=NEG if dd < 0 else POS)
        ax.text(xi - w / 2, vv + ev + 1.4, f"{vv:.1f}", ha="center", va="bottom",
                fontsize=7.5, color=INK_2)
        ax.text(xi + w / 2, cc + ec + 1.4, f"{cc:.1f}", ha="center", va="bottom",
                fontsize=7.5, color=INK_2)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{b}\n$n_{{AI}}$={int(na)}, $n_{{all}}$={int(nl)}"
                        for b, na, nl in zip(BANDS, ind["n_ai"], ind["n_all"])],
                       fontsize=8)
    ax.set_ylabel("Organisations (%)")
    ax.set_ylim(0, 108)
    ax.legend(ncol=2, loc="upper left")
    _clean(ax)
    ax.text(0.0, -0.30,
            "$V_s$ is based on organisations using or considering AI; $C_s$ on all "
            "organisations. The two bases differ,\nso $D_s$ is a differential between "
            "populations rather than a within-firm difference.",
            transform=ax.transAxes, fontsize=7, color=MUTED, va="top")
    _save(fig, "fig_ai_governance.pdf")


# ===========================================================================

def main():
    print("Generating figures ...")
    bics = ld.load_bics_ai_usage(str(BICS_PATH), wave="Wave 147")
    ind = ld.build_indicator_frame(str(CSBS_PATH))

    fig_bics_sector(bics)
    fig_bics_mix(bics)
    fig_csbs_profile(ind)
    fig_gap(ind)
    fig_cluster(ind)
    fig_ai_governance(ind)
    print(f"\nAll figures written to {FIGS.relative_to(HERE.parent)}/")


if __name__ == "__main__":
    main()
