"""
model_selection.py
==================
Empirical justification for the estimator choices made in this study.

Chapter 3 of the dissertation is required to justify why each method was chosen
over the alternatives a reader might reasonably expect instead. This module
produces that justification as evidence rather than assertion. Nothing here is
argued from authority: every candidate estimator is implemented, evaluated on
the rates and effective bases that actually occur in these data, and the
comparison is written to disk so that the tables in Sections 3.4, 3.6 and 3.7
are generated rather than transcribed.

Outputs (all written to out/):
    model_selection.json            every result below, machine-readable
    ms_interval_widths.csv          candidate interval widths at four anchors
    ms_coverage.csv                 Monte Carlo coverage of each candidate
    ms_multiplicity.csv             rejections under four multiplicity rules
    ms_kmeans_sweep.csv             inertia and silhouette across k
    ms_composite_weighting.csv      sensitivity of C_s to component weights

Author: Frida Alejandra Ibarra Zavaleta
"""

from __future__ import annotations

import json
import os
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import load_data as ld
import stats_core as sc

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
CSBS = os.path.join(DATA, "DSIT_CSBS_2025-2026_tables.xlsx")

os.makedirs(OUT, exist_ok=True)

SEED = 42
N_SIM = 20000

INDICATORS = [("T", "Training"), ("B", "Board responsibility"),
              ("P", "Formal policy"), ("A", "AI adoption"), ("R", "Any breach")]


# ---------------------------------------------------------------------------
# 1. Interval estimator selection
# ---------------------------------------------------------------------------

def interval_anchors(ind: pd.DataFrame) -> list[dict]:
    """Four points chosen to span the range of (rate, base) this study meets.

    The anchors are not illustrative examples invented for the purpose. Each is
    an estimate the analysis actually reports, chosen so that the comparison
    covers a low rate on a large base, a mid rate on a large base, a mid rate on
    the smallest base in the study, and a high rate near the upper boundary.
    """
    m = ind.set_index("size_band")
    t28 = ld.load_csbs_table(CSBS, "Table28")
    ransom = t28.rate_series(
        "Your organisation’s devices being targeted with ransomware")
    outside = t28.rate_series(
        "Unauthorised accessing of files or networks by people outside "
        "your organisation (other than staff or students)")
    return [
        {"label": "AI adoption, medium", "symbol": "$A_{\\mathrm{Medium}}$",
         "p": float(m.loc["Medium", "A"]), "n": float(m.loc["Medium", "n_all"]),
         "why": "mid rate, mid base"},
        {"label": "AI practices, large", "symbol": "$V_{\\mathrm{Large}}$",
         "p": float(m.loc["Large", "V"]), "n": float(m.loc["Large", "n_ai"]),
         "why": "mid rate, smallest base in the study"},
        {"label": "Formal policy, large", "symbol": "$P_{\\mathrm{Large}}$",
         "p": float(m.loc["Large", "P"]), "n": float(m.loc["Large", "n_all"]),
         "why": "high rate near the upper boundary"},
        {"label": "Ransomware, micro", "symbol": "---",
         "p": float(ransom[0]), "n": float(m.loc["Micro", "n_all"]),
         "why": "very low rate, largest base"},
        {"label": "External hacking, large", "symbol": "---",
         "p": float(outside[3]), "n": float(m.loc["Large", "n_all"]),
         "why": "very low rate, small base: the boundary case"},
    ]


def run_interval_comparison(anchors: list[dict]) -> pd.DataFrame:
    rows = []
    for a in anchors:
        comp = sc.interval_comparison(a["p"], a["n"])
        for method, r in comp.items():
            rows.append({
                "anchor": a["label"], "symbol": a["symbol"],
                "p_pct": round(a["p"] * 100, 1), "n_eff": int(round(a["n"])),
                "method": method,
                "low_pct": round(r["low"] * 100, 2),
                "high_pct": round(r["high"] * 100, 2),
                "width_pp": round(r["width_pp"], 2),
                "admissible": r["admissible"],
            })
    return pd.DataFrame(rows)


def run_coverage(anchors: list[dict]) -> pd.DataFrame:
    """Monte Carlo coverage. This is the decisive evidence for the choice."""
    rows = []
    for a in anchors:
        cov = sc.coverage_simulation(a["p"], a["n"], n_sim=N_SIM, seed=SEED)
        for method, r in cov.items():
            rows.append({
                "anchor": a["label"], "p_pct": round(a["p"] * 100, 1),
                "n_eff": int(round(a["n"])), "method": method,
                "coverage_pct": round(r["coverage"] * 100, 2),
                "shortfall_pp": round(r["coverage_error_pp"], 2),
                "mean_width_pp": round(r["mean_width_pp"], 2),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 2. Multiplicity procedure selection
# ---------------------------------------------------------------------------

def build_contrast_family(ind: pd.DataFrame) -> pd.DataFrame:
    """The thirty pairwise size-band contrasts that form the test family."""
    tests = []
    for key, name in INDICATORS:
        for i, j in combinations(range(4), 2):
            a, b = ind.iloc[i], ind.iloc[j]
            tests.append(sc.two_proportion_test(
                a[key], a["n_all"], b[key], b["n_all"],
                label=f"{name}: {a['size_band']} vs {b['size_band']}").as_dict())
    return pd.DataFrame(tests)


def run_multiplicity(t: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    comp = sc.multiplicity_comparison(t["p_value"].tolist())
    summary = pd.DataFrame([
        {"procedure": "Uncorrected", "controls": "Nothing",
         "rejections": comp["uncorrected"]},
        {"procedure": "Bonferroni", "controls": "FWER (single-step)",
         "rejections": comp["bonferroni"]},
        {"procedure": "Holm-Bonferroni", "controls": "FWER (step-down)",
         "rejections": comp["holm"]},
        {"procedure": "Benjamini-Hochberg", "controls": "FDR",
         "rejections": comp["benjamini_hochberg"]},
    ])
    summary["of_n_tests"] = comp["n_tests"]
    detail = t.copy()
    detail["p_holm"] = [x["p_holm"] for x in comp["detail"]["holm"]]
    detail["reject_holm"] = [x["reject_holm"] for x in comp["detail"]["holm"]]
    detail["p_bonferroni"] = [x["p_adj"] for x in comp["detail"]["bonferroni"]]
    detail["reject_bonferroni"] = [x["reject"] for x in comp["detail"]["bonferroni"]]
    detail["p_bh"] = [x["p_adj"] for x in comp["detail"]["bh"]]
    detail["reject_bh"] = [x["reject"] for x in comp["detail"]["bh"]]
    comp.pop("detail")
    return comp, summary, detail


# ---------------------------------------------------------------------------
# 3. Hyperparameter selection for the exploratory partition
# ---------------------------------------------------------------------------

def run_kmeans_selection(ind: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    from sklearn.preprocessing import StandardScaler

    X = ind[["T", "B", "A", "C", "R"]].to_numpy(float)
    Xs = StandardScaler().fit_transform(X)

    sweep = sc.kmeans_sweep(Xs, k_values=(2, 3), seed=SEED)
    stability = sc.partition_stability(Xs, k=2, seed=SEED)

    # Does standardisation matter? If the partition is identical on raw
    # features, the preprocessing choice is not doing the work.
    raw = sc.kmeans_sweep(X, k_values=(2,), seed=SEED)
    scaling_effect = {
        "labels_standardised": sweep[0]["labels"],
        "labels_raw": raw[0]["labels"],
        "identical": bool(np.array_equal(np.array(sweep[0]["labels"]),
                                         np.array(raw[0]["labels"])) or
                          np.array_equal(1 - np.array(sweep[0]["labels"]),
                                         np.array(raw[0]["labels"]))),
    }

    # Seed sensitivity: 50 seeds, does the partition ever move?
    from sklearn.cluster import KMeans
    base = np.array(sweep[0]["labels"])
    moved = 0
    for s in range(50):
        lab = KMeans(n_clusters=2, n_init=20, random_state=s).fit(Xs).labels_
        if not (np.array_equal(lab, base) or np.array_equal(1 - lab, base)):
            moved += 1
    stability["seeds_tested"] = 50
    stability["seeds_giving_different_partition"] = int(moved)

    return pd.DataFrame([{k: v for k, v in d.items() if k != "labels"}
                         for d in sweep]), stability, scaling_effect


# ---------------------------------------------------------------------------
# 4. Composite weighting: the one genuine tuning parameter
# ---------------------------------------------------------------------------

def run_composite_weighting(ind: pd.DataFrame) -> pd.DataFrame:
    """C_s under equal weights and under every leave-one-out alternative.

    Equal weighting is a modelling decision, not a fitted parameter, and it is
    the only place in the analysis where a free choice materially shapes a
    reported quantity. It is therefore varied systematically.
    """
    schemes = {
        "Equal (T,B,P)": {"T": 1 / 3, "B": 1 / 3, "P": 1 / 3},
        "Omit T": {"B": 0.5, "P": 0.5},
        "Omit B": {"T": 0.5, "P": 0.5},
        "Omit P": {"T": 0.5, "B": 0.5},
        "Govern-weighted": {"T": 0.2, "B": 0.4, "P": 0.4},
        "Protect-weighted": {"T": 0.5, "B": 0.25, "P": 0.25},
    }
    rows = []
    for name, w in schemes.items():
        c = sum(ind[k] * v for k, v in w.items())
        row = {"scheme": name}
        for b, cv, gv, dv in zip(ind["size_band"], c, ind["A"] - c, ind["V"] - c):
            row[f"C_{b}"] = round(float(cv * 100), 2)
            row[f"G_{b}"] = round(float(gv * 100), 2)
            row[f"D_{b}"] = round(float(dv * 100), 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    gcols = [c for c in df.columns if c.startswith("G_")]
    dcols = [c for c in df.columns if c.startswith("D_")]
    df.attrs["G_sign_stable"] = bool((df[gcols] < 0).all().all())
    df.attrs["D_sign_stable"] = bool((df[dcols] < 0).all().all())
    return df


# ---------------------------------------------------------------------------
# 5. Evaluation metrics actually attained
# ---------------------------------------------------------------------------

def run_precision_budget(ind: pd.DataFrame) -> pd.DataFrame:
    """What precision the design buys, band by band.

    A study working from published tables cannot choose its sample size, so the
    honest equivalent of a power calculation is to report the smallest
    difference the attained bases can resolve.
    """
    rows = []
    for _, r in ind.iterrows():
        # Minimum detectable difference at 80% power, alpha = 0.05, two-sided,
        # against the same band's composite rate, using the standard
        # two-proportion approximation.
        for base_col, base_name, p in [("n_all", "All organisations", r["C"]),
                                       ("n_ai", "AI-engaged", r["V"])]:
            n = r[base_col]
            z_a, z_b = sc.Z_95, 0.8416
            mdd = (z_a + z_b) * np.sqrt(2 * p * (1 - p) / n)
            rows.append({
                "size_band": r["size_band"], "base": base_name,
                "n_eff": int(n),
                "rate_pct": round(float(p) * 100, 1),
                "moe_pp": round(sc.moe(p, n) * 100, 2),
                "min_detectable_diff_pp": round(float(mdd) * 100, 1),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    ind = ld.build_indicator_frame(CSBS)

    anchors = interval_anchors(ind)
    widths = run_interval_comparison(anchors)
    coverage = run_coverage(anchors)

    family = build_contrast_family(ind)
    mult, mult_summary, mult_detail = run_multiplicity(family)

    sweep, stability, scaling = run_kmeans_selection(ind)
    weighting = run_composite_weighting(ind)
    precision = run_precision_budget(ind)

    widths.to_csv(os.path.join(OUT, "ms_interval_widths.csv"), index=False)
    coverage.to_csv(os.path.join(OUT, "ms_coverage.csv"), index=False)
    mult_summary.to_csv(os.path.join(OUT, "ms_multiplicity.csv"), index=False)
    mult_detail.to_csv(os.path.join(OUT, "ms_multiplicity_detail.csv"), index=False)
    sweep.to_csv(os.path.join(OUT, "ms_kmeans_sweep.csv"), index=False)
    weighting.to_csv(os.path.join(OUT, "ms_composite_weighting.csv"), index=False)
    precision.to_csv(os.path.join(OUT, "ms_precision_budget.csv"), index=False)

    results = {
        "seed": SEED,
        "n_simulations": N_SIM,
        "anchors": anchors,
        "interval_widths": widths.to_dict(orient="records"),
        "coverage": coverage.to_dict(orient="records"),
        "multiplicity": mult,
        "multiplicity_summary": mult_summary.to_dict(orient="records"),
        "kmeans_sweep": sweep.to_dict(orient="records"),
        "partition_stability": stability,
        "scaling_effect": scaling,
        "composite_weighting": weighting.to_dict(orient="records"),
        "composite_weighting_G_sign_stable": weighting.attrs["G_sign_stable"],
        "composite_weighting_D_sign_stable": weighting.attrs["D_sign_stable"],
        "precision_budget": precision.to_dict(orient="records"),
    }
    with open(os.path.join(OUT, "model_selection.json"), "w") as fh:
        json.dump(results, fh, indent=2, default=float)

    # ---- console summary -------------------------------------------------
    print("=" * 72)
    print("MODEL SELECTION EVIDENCE")
    print("=" * 72)
    print("\n1. Interval estimator: Monte Carlo coverage of a nominal 95% interval")
    piv = coverage.pivot(index="method", columns="anchor", values="coverage_pct")
    print(piv.round(2).to_string())
    print("\n   Mean width (pp):")
    print(coverage.pivot(index="method", columns="anchor",
                         values="mean_width_pp").round(2).to_string())
    print("\n   Admissibility (interval inside [0,1]):")
    print(widths.groupby("method")["admissible"].all().to_string())

    print("\n2. Multiplicity procedure over the 30-contrast family")
    print(mult_summary.to_string(index=False))
    print(f"   FWER under a global null, uncorrected: "
          f"{mult['fwer_under_global_null']*100:.1f}%")

    print("\n3. Partition hyperparameters")
    print(sweep.to_string(index=False))
    print(f"   Leave-one-feature-out stability: "
          f"{stability['n_stable']}/{stability['n_features']}")
    print(f"   Seeds giving a different partition: "
          f"{stability['seeds_giving_different_partition']}/{stability['seeds_tested']}")
    print(f"   Standardised and raw partitions identical: {scaling['identical']}")

    print("\n4. Composite weighting sensitivity")
    print(weighting[["scheme"] + [c for c in weighting.columns
                                  if c.startswith("D_")]].to_string(index=False))
    print(f"   Sign of G_s stable across schemes: {weighting.attrs['G_sign_stable']}")
    print(f"   Sign of D_s stable across schemes: {weighting.attrs['D_sign_stable']}")

    print("\n5. Attained precision")
    print(precision.to_string(index=False))
    print("\nWritten to out/model_selection.json and seven CSVs.")
    return results


if __name__ == "__main__":
    main()
