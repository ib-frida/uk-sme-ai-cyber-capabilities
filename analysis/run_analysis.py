"""
run_analysis.py
===============
The full quantitative analysis, organised to match the three research
questions of the final dissertation.

  RQ1  Sectoral patterns of AI adoption                       (BICS Wave 147)
  RQ2  Cybersecurity capability, governance, breach exposure  (CSBS 2025/26)
  RQ3  AI-specific vs general governance, like-for-like       (CSBS 2025/26)

A note on what this script does and does not do. Both sources publish
population-weighted aggregate rates for four size bands and six sectors. There
are no record-level data. Every result below is therefore a statement about
band-level or sector-level rates, and the script is written to make that
boundary hard to cross by accident: nothing here produces a firm-level
estimate, and the inferential machinery is applied only where the published
tables supply an effective base size to apply it to.

Outputs (written to ../out):
    indicators.csv             every indicator with its base and Wilson CI
    rq1_sector.csv             sector adoption and composition
    rq2_pairwise_tests.csv     all size-band contrasts, Holm-adjusted
    rq3_differential.csv       D_s with conservative CIs and tests
    results.json               machine-readable summary of every statistic
    verification_report.md     recomputed values vs those in the dissertation

Usage:
    python3 run_analysis.py

Author: Frida Alejandra Ibarra Zavaleta
"""

from __future__ import annotations

import json
import warnings
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import load_data as ld
import stats_core as sc

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUT = HERE.parent / "out"
OUT.mkdir(exist_ok=True)

BICS_PATH = DATA / "wave147dsitaiadoptionrequest.xlsx"
CSBS_PATH = DATA / "DSIT_CSBS_2025-2026_tables.xlsx"

BANDS = ld.SIZE_LABELS
results: dict = {}


def pct(x, nd=1):
    return None if not np.isfinite(x) else round(float(x) * 100, nd)


def rule(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ===========================================================================
# RQ1 — Sectoral patterns of AI adoption (BICS Wave 147)
# ===========================================================================

def run_rq1() -> pd.DataFrame:
    rule("RQ1  Sectoral patterns of AI adoption  (BICS Wave 147)")

    bics = ld.load_bics_ai_usage(str(BICS_PATH), wave="Wave 147")
    all_biz = bics.attrs["all_business_rate"]
    mean6 = bics["ai_use_rate"].mean()

    bics["deviation_pp"] = (bics["ai_use_rate"] - mean6) * 100
    bics["margin_over_all_business_pp"] = (bics["ai_use_rate"] - all_biz) * 100

    print(f"\nAll-business adoption rate: {pct(all_biz)}%")
    print(f"Six-sector mean:            {pct(mean6)}%")
    print(f"Spread (max - min):         "
          f"{pct(bics['ai_use_rate'].max() - bics['ai_use_rate'].min())} pp\n")

    show = bics[["sector", "ai_use_rate", "deviation_pp",
                 "margin_over_all_business_pp"]].copy()
    show["ai_use_rate"] = (show["ai_use_rate"] * 100).round(1)
    show[["deviation_pp", "margin_over_all_business_pp"]] = \
        show[["deviation_pp", "margin_over_all_business_pp"]].round(1)
    print(show.to_string(index=False))

    # --- composition -------------------------------------------------------
    tech = ["ai_llm_rate", "ai_ml_data_rate", "ai_image_ml_rate", "ai_robotics_rate"]
    comp = bics[["sector"] + tech].copy()

    top, bottom = "Creative Industries", "Defence"
    ratios = {}
    for t in tech:
        a = float(comp.loc[comp["sector"] == top, t].iloc[0])
        b = float(comp.loc[comp["sector"] == bottom, t].iloc[0])
        ratios[t] = None if not (np.isfinite(a) and np.isfinite(b) and b > 0) else round(a / b, 2)

    n_suppressed = int(comp[tech].isna().sum().sum())
    print(f"\nSuppressed technology cells: {n_suppressed} of {comp[tech].size}")
    print(f"Creative Industries : Defence ratios -> {ratios}")

    # --- sensitivity: does the ranking depend on the derived headline? ------
    # The headline is 1 - (not currently using). An alternative reading is the
    # maximum of the named technology rates, which is a lower bound on adoption
    # of any *listed* technology. If the sector ordering is preserved under
    # both, the Defence result is not an artefact of the derivation.
    bics["alt_lower_bound"] = bics[tech].max(axis=1)
    rank_headline = bics["ai_use_rate"].rank(ascending=False)
    rank_alt = bics["alt_lower_bound"].rank(ascending=False)
    rank_agreement = float(np.corrcoef(rank_headline, rank_alt)[0, 1])
    defence_lowest_both = bool(
        bics.loc[bics["sector"] == "Defence", "ai_use_rate"].iloc[0] == bics["ai_use_rate"].min()
        and bics.loc[bics["sector"] == "Defence", "alt_lower_bound"].iloc[0] == bics["alt_lower_bound"].min()
    )
    print(f"\nSensitivity: rank correlation headline vs alternative measure = {rank_agreement:.3f}")
    print(f"Defence lowest under both measures: {defence_lowest_both}")

    print("\nINFERENCE NOTE: the BICS ad hoc release publishes no base sizes, so "
          "\nno confidence interval or significance test is available for any RQ1 "
          "\ncomparison. Every figure above is descriptive.")

    results["rq1"] = {
        "all_business_rate_pct": pct(all_biz),
        "six_sector_mean_pct": pct(mean6),
        "spread_pp": pct(bics["ai_use_rate"].max() - bics["ai_use_rate"].min()),
        "by_sector": [
            {"sector": str(r.sector),
             "ai_use_pct": pct(r.ai_use_rate),
             "deviation_pp": round(r.deviation_pp, 1),
             "llm_pct": pct(r.ai_llm_rate),
             "ml_data_pct": pct(r.ai_ml_data_rate),
             "image_pct": pct(r.ai_image_ml_rate),
             "robotics_pct": pct(r.ai_robotics_rate)}
            for r in bics.itertuples()
        ],
        "creative_defence_ratios": ratios,
        "suppressed_cells": n_suppressed,
        "sensitivity_rank_agreement": round(rank_agreement, 3),
        "defence_lowest_under_both_measures": defence_lowest_both,
        "inference_available": False,
        "inference_note": "No base sizes published in the BICS ad hoc release.",
    }

    bics.to_csv(OUT / "rq1_sector.csv", index=False)
    return bics


# ===========================================================================
# RQ2 — Cyber capability, governance and breach exposure (CSBS)
# ===========================================================================

def run_rq2(ind: pd.DataFrame) -> pd.DataFrame:
    rule("RQ2  Cybersecurity capability by firm size  (CSBS 2025/2026)")

    print("\nEffective base sizes (all-organisation questions): "
          f"{[int(n) for n in ind['n_all']]}")
    print(f"Respondent counts:                                 "
          f"{[int(n) for n in ind['respondents_all']]}")

    # --- levels with Wilson intervals --------------------------------------
    print("\nIndicator levels with 95% Wilson intervals (percentage points)\n")
    rows = []
    for _, r in ind.iterrows():
        for key, name in [("T", "Training"), ("B", "Board resp."),
                          ("P", "Formal policy"), ("A", "AI adoption"),
                          ("R", "Any breach")]:
            lo, hi = sc.wilson_ci(r[key], r["n_all"])
            rows.append({"size_band": r["size_band"], "indicator": name,
                         "rate_pct": pct(r[key]),
                         "ci_low_pct": pct(lo), "ci_high_pct": pct(hi),
                         "moe_pp": round(sc.moe(r[key], r["n_all"]) * 100, 2),
                         "n_effective": int(r["n_all"])})
    levels = pd.DataFrame(rows)
    print(levels.pivot(index="indicator", columns="size_band",
                       values="rate_pct")[BANDS].to_string())

    # --- composite with correlation bounds ---------------------------------
    print("\nComposite cyber capability C_s, with bounding standard errors")
    print("(components share respondents; the covariance is unrecoverable from")
    print(" published tables, so both bounds are reported)\n")
    comp_rows = []
    for _, r in ind.iterrows():
        se_indep, se_perfect = sc.composite_se_bounds([r["T"], r["B"], r["P"]], r["n_all"])
        comp_rows.append({
            "size_band": r["size_band"],
            "C_pct": pct(r["C"], 2),
            "se_lower_pp": round(se_indep * 100, 2),
            "se_upper_pp": round(se_perfect * 100, 2),
            "ci_widest_low_pct": pct(r["C"] - sc.Z_95 * se_perfect, 1),
            "ci_widest_high_pct": pct(r["C"] + sc.Z_95 * se_perfect, 1),
        })
    comp = pd.DataFrame(comp_rows)
    print(comp.to_string(index=False))

    # --- pairwise size-band contrasts, Holm-adjusted -----------------------
    print("\nPairwise size-band contrasts (two-proportion z on effective bases)\n")
    tests = []
    for key, name in [("T", "Training"), ("B", "Board resp."),
                      ("P", "Formal policy"), ("A", "AI adoption"),
                      ("R", "Any breach")]:
        for i, j in combinations(range(4), 2):
            a, b = ind.iloc[i], ind.iloc[j]
            t = sc.two_proportion_test(
                a[key], a["n_all"], b[key], b["n_all"],
                label=f"{name}: {a['size_band']} vs {b['size_band']}")
            tests.append(t.as_dict())
    tdf = pd.DataFrame(tests)
    adj = sc.holm_bonferroni(tdf["p_value"].tolist())
    tdf["p_holm"] = [a["p_holm"] for a in adj]
    tdf["reject_holm"] = [a["reject_holm"] for a in adj]

    n_raw = int(tdf["significant"].sum())
    n_holm = int(tdf["reject_holm"].sum())
    print(f"{len(tdf)} contrasts. Significant at raw alpha=0.05: {n_raw}. "
          f"Surviving Holm-Bonferroni: {n_holm}.")

    key_rows = tdf[tdf["label"].str.contains("Micro vs Small|Medium vs Large|Micro vs Medium")]
    print("\nThe contrasts that carry interpretive weight:\n")
    print(key_rows[["label", "diff_pp", "ci_low_pp", "ci_high_pp",
                    "z", "p_value", "p_holm", "cohens_h", "h_magnitude"]]
          .round({"diff_pp": 1, "ci_low_pp": 1, "ci_high_pp": 1, "z": 2,
                  "p_value": 4, "p_holm": 4, "cohens_h": 2}).to_string(index=False))

    # --- the effect-size asymmetry that drives Chapter 3's argument --------
    print("\nEffect-size asymmetry, micro vs medium: purchasable vs practised controls\n")
    t15 = ld.load_csbs_table(str(CSBS_PATH), "Table15")
    asym = []
    for label, kind in [
        ("Up-to-date malware protection", "purchasable"),
        ("Firewalls that cover your entire IT network, as well as individual devices", "purchasable"),
        ("Backing up data securely via a cloud service", "purchasable"),
        ("Any monitoring of user activity", "practised"),
        ("Separate WiFi networks for staff and for visitors", "practised"),
    ]:
        try:
            s = t15.rate_series(label)
        except KeyError:
            continue
        t = sc.two_proportion_test(s[0], ind["n_all"][0], s[2], ind["n_all"][2], label=label)
        asym.append({"control": label[:52], "kind": kind,
                     "micro_pct": pct(s[0]), "medium_pct": pct(s[2]),
                     "cohens_h": round(t.cohens_h, 2), "magnitude": t.h_magnitude})
    # training is the practised control the dissertation leans on hardest
    t_train = sc.two_proportion_test(ind["T"][0], ind["n_all"][0],
                                     ind["T"][2], ind["n_all"][2], label="Staff training")
    asym.append({"control": "Cyber security training (Table 19)", "kind": "practised",
                 "micro_pct": pct(ind["T"][0]), "medium_pct": pct(ind["T"][2]),
                 "cohens_h": round(t_train.cohens_h, 2), "magnitude": t_train.h_magnitude})
    asym_df = pd.DataFrame(asym)
    print(asym_df.to_string(index=False))

    # --- the detection confound, quantified --------------------------------
    print("\nDetection confound: monitoring capability against breach reporting\n")
    monitoring = t15.rate_series("Any monitoring of user activity")
    t12 = ld.load_csbs_table(str(CSBS_PATH), "Table12")
    sec_tools = t12.rate_series(
        "Used specific tools designed for security monitoring, such as Intrusion Detection Systems")
    sp_mon = sc.spearman_exact(monitoring, ind["R"].tolist())
    sp_tools = sc.spearman_exact(sec_tools, ind["R"].tolist())
    sp_cap = sc.spearman_exact(ind["C"].tolist(), ind["R"].tolist())
    print(f"  monitoring vs breach rate      rho = {sp_mon['rho']:.2f}  "
          f"exact p = {sp_mon['p_exact_two_sided']:.3f}")
    print(f"  security tools vs breach rate  rho = {sp_tools['rho']:.2f}  "
          f"exact p = {sp_tools['p_exact_two_sided']:.3f}")
    print(f"  composite C_s vs breach rate   rho = {sp_cap['rho']:.2f}  "
          f"exact p = {sp_cap['p_exact_two_sided']:.3f}")
    print(f"\n  Smallest attainable two-sided p at n=4 is "
          f"{sp_cap['min_attainable_p']:.3f}. None of these can reach p<0.05,")
    print("  and none is offered as evidence of association. They describe ordering.")
    print("\n  The substantive point stands without a p-value: capability and")
    print("  reported breaches rise together, which is the wrong sign for a")
    print("  protective effect and the right sign for a detection effect. R_s is")
    print("  therefore not usable as an outcome variable.")

    levels.to_csv(OUT / "rq2_levels.csv", index=False)
    tdf.to_csv(OUT / "rq2_pairwise_tests.csv", index=False)
    asym_df.to_csv(OUT / "rq2_effect_size_asymmetry.csv", index=False)

    results["rq2"] = {
        "effective_bases": [int(n) for n in ind["n_all"]],
        "respondents": [int(n) for n in ind["respondents_all"]],
        "levels": levels.to_dict("records"),
        "composite": comp.to_dict("records"),
        "n_contrasts": len(tdf),
        "n_significant_raw": n_raw,
        "n_significant_holm": n_holm,
        "key_contrasts": key_rows[["label", "diff_pp", "ci_low_pp", "ci_high_pp",
                                   "z", "p_value", "p_holm", "cohens_h",
                                   "h_magnitude"]].round(4).to_dict("records"),
        "effect_size_asymmetry": asym_df.to_dict("records"),
        "spearman": {"monitoring_vs_breach": sp_mon,
                     "security_tools_vs_breach": sp_tools,
                     "composite_vs_breach": sp_cap},
    }
    return tdf


# ===========================================================================
# RQ3 — AI-specific governance differential
# ===========================================================================

def run_rq3(ind: pd.DataFrame) -> pd.DataFrame:
    rule("RQ3  AI-specific vs general cyber governance  (CSBS 2025/2026)")

    # --- the naive gap, reported and then set aside ------------------------
    print("\nStep 1. The naive capability gap G_s = A_s - C_s\n")
    gap = pd.DataFrame({
        "size_band": ind["size_band"],
        "A_pct": [pct(v) for v in ind["A"]],
        "C_pct": [pct(v) for v in ind["C"]],
        "G_pp": [round(v * 100, 1) for v in ind["G"]],
        "label": ["Near balance" if abs(v * 100) <= 5
                  else ("Cyber ahead" if v < 0 else "AI ahead") for v in ind["G"]],
    })
    print(gap.to_string(index=False))
    print("\n  G_s is not construct-valid: A_s is an adoption measure and C_s is a")
    print("  governance measure. It is reported because the published tables invite")
    print("  the comparison, and because showing why it fails is part of the result.")

    # --- the like-for-like differential ------------------------------------
    print("\nStep 2. The construct-valid differential D_s = V_s - C_s\n")
    rows = []
    for _, r in ind.iterrows():
        lo, hi = sc.differential_ci(r["V"], r["n_ai"], r["C"], r["n_all"])
        test = sc.two_proportion_test(r["V"], r["n_ai"], r["C"], r["n_all"],
                                      label=f"D: {r['size_band']}")
        u_lo, u_hi = sc.wilson_ci(r["U"], r["n_ai"])
        rows.append({
            "size_band": r["size_band"],
            "V_pct": pct(r["V"]), "n_ai": int(r["n_ai"]),
            "C_pct": pct(r["C"]), "n_all": int(r["n_all"]),
            "D_pp": round(r["D"] * 100, 1),
            "D_ci_low_pp": round(lo, 1), "D_ci_high_pp": round(hi, 1),
            "z": round(test.z, 2), "p_value": round(test.p_value, 4),
            "cohens_h": round(test.cohens_h, 2), "h_magnitude": test.h_magnitude,
            "U_pct": pct(r["U"]),
            "U_ci_low_pct": pct(u_lo), "U_ci_high_pct": pct(u_hi),
        })
    d = pd.DataFrame(rows)
    print(d[["size_band", "V_pct", "n_ai", "C_pct", "D_pp",
             "D_ci_low_pp", "D_ci_high_pp", "p_value", "cohens_h"]].to_string(index=False))

    print("\n  The CI is computed under an independence assumption. V_s and C_s are")
    print("  measured on overlapping populations, and positive correlation between")
    print("  overlapping samples reduces the variance of a difference, so this")
    print("  interval is conservative: it cannot overstate the precision of D_s.")

    # --- is the differential stable above micro? ---------------------------
    print("\nStep 3. Is the -13 to -15pp differential stable, or is that an artefact")
    print("        of small AI-engaged bases?\n")
    print(f"  AI-engaged effective bases: {[int(n) for n in ind['n_ai']]}")
    sub = d[d["size_band"] != "Micro"]
    span = sub["D_pp"].max() - sub["D_pp"].min()
    widest_ci = float((sub["D_ci_high_pp"] - sub["D_ci_low_pp"]).max())
    print(f"  Spread of D_s across small/medium/large: {span:.1f} pp")
    print(f"  Widest 95% CI on any of those D_s:       {widest_ci:.1f} pp")
    stable_claim_supported = bool(span < widest_ci)
    print(f"\n  The observed spread ({span:.1f} pp) is "
          f"{'smaller' if stable_claim_supported else 'larger'} than the "
          f"uncertainty on a single\n  estimate ({widest_ci:.1f} pp). Describing the "
          "differential as stable across these\n  three bands is therefore "
          f"{'defensible: the data cannot distinguish them' if stable_claim_supported else 'not supported'}.")

    # explicit test of whether the small-band and large-band D differ
    d_small = ind.loc[ind["size_band"] == "Small"].iloc[0]
    d_large = ind.loc[ind["size_band"] == "Large"].iloc[0]
    se_small = np.sqrt(d_small["V"] * (1 - d_small["V"]) / d_small["n_ai"]
                       + d_small["C"] * (1 - d_small["C"]) / d_small["n_all"])
    se_large = np.sqrt(d_large["V"] * (1 - d_large["V"]) / d_large["n_ai"]
                       + d_large["C"] * (1 - d_large["C"]) / d_large["n_all"])
    z_dd = (d_small["D"] - d_large["D"]) / np.sqrt(se_small**2 + se_large**2)
    from scipy import stats as _st
    p_dd = 2 * (1 - _st.norm.cdf(abs(z_dd)))
    print(f"\n  Difference-in-differences, small vs large: "
          f"{(d_small['D'] - d_large['D'])*100:+.1f} pp, z = {z_dd:.2f}, p = {p_dd:.3f}")
    print("  The differential does not narrow as general governance improves, which")
    print("  is the second component of H1.")

    # --- micro band: why the smaller differential is not good news ---------
    micro = ind.iloc[0]
    print(f"\nStep 4. The micro band. D = {micro['D']*100:+.1f} pp looks better than the")
    print(f"        others, but V = {pct(micro['V'])}% is the lowest of any band and")
    print(f"        U = {pct(micro['U'])}% of AI-engaged micro firms have no AI risk")
    print("        process. The small differential reflects a low C_s, not better")
    print("        coverage of AI risk.")

    # --- robustness of the composite ---------------------------------------
    print("\nStep 5. Robustness of C_s to composite construction\n")
    rob_rows = []
    for name, comps in [("Full (T,B,P)", ["T", "B", "P"]),
                        ("Omit T", ["B", "P"]),
                        ("Omit B", ["T", "P"]),
                        ("Omit P", ["T", "B"])]:
        c_alt = ind[comps].mean(axis=1)
        rob_rows.append({
            "variant": name,
            **{b: round(float(g * 100), 2) for b, g in zip(BANDS, ind["A"] - c_alt)},
            **{f"D_{b}": round(float(x * 100), 2) for b, x in zip(BANDS, ind["V"] - c_alt)},
        })
    rob = pd.DataFrame(rob_rows)
    print("  G_s under leave-one-out composites:")
    print(rob[["variant"] + BANDS].to_string(index=False))
    print("\n  D_s under leave-one-out composites:")
    print(rob[["variant"] + [f"D_{b}" for b in BANDS]].to_string(index=False))

    g_signs_stable = bool((rob[BANDS] < 0).all().all())
    d_signs_stable = bool((rob[[f"D_{b}" for b in BANDS]] < 0).all().all())
    micro_label_flips = bool(abs(rob.loc[rob["variant"] == "Omit T", "Micro"].iloc[0]) > 5)
    print(f"\n  Sign of G_s negative in every variant: {g_signs_stable}")
    print(f"  Sign of D_s negative in every variant: {d_signs_stable}")
    print(f"  Micro 'near balance' label flips when training is dropped: {micro_label_flips}")
    print("  The substantive conclusions do not depend on the weighting; the")
    print("  boundary label for one band does, and is reported as an artefact.")

    # --- clustering --------------------------------------------------------
    print("\nStep 6. Exploratory clustering (k = 2, seed 42)\n")
    X = ind[["T", "B", "A", "C", "R"]].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=2, n_init=20, random_state=SEED).fit(Xs)
    labels = km.labels_
    sil = float(silhouette_score(Xs, labels)) if len(set(labels)) > 1 else np.nan
    print("  " + "  ".join(f"{b}={l}" for b, l in zip(BANDS, labels)))
    print(f"  Silhouette: {sil:.3f}")

    # leave-one-feature-out stability
    stable = 0
    for f in range(X.shape[1]):
        Xf = np.delete(Xs, f, axis=1)
        lf = KMeans(n_clusters=2, n_init=20, random_state=SEED).fit(Xf).labels_
        same = (np.array_equal(lf, labels) or np.array_equal(1 - lf, labels))
        stable += int(same)
    print(f"  Partition unchanged under leave-one-feature-out: {stable}/{X.shape[1]}")
    print("\n  With n = 4 observations and k = 2, any partition separates the data")
    print("  along the dominant axis of variation, which here is firm size itself.")
    print("  This cannot demonstrate a typology and is not treated as doing so. It")
    print("  supports one narrow claim: the five indicators group the bands")
    print("  consistently when considered jointly rather than one at a time.")

    d.to_csv(OUT / "rq3_differential.csv", index=False)
    rob.to_csv(OUT / "rq3_robustness.csv", index=False)

    results["rq3"] = {
        "naive_gap": gap.to_dict("records"),
        "differential": d.to_dict("records"),
        "ai_engaged_bases": [int(n) for n in ind["n_ai"]],
        "stability": {
            "spread_above_micro_pp": round(float(span), 1),
            "widest_ci_width_pp": round(widest_ci, 1),
            "stable_claim_supported": stable_claim_supported,
            "did_small_vs_large_pp": round(float((d_small["D"] - d_large["D"]) * 100), 1),
            "did_z": round(float(z_dd), 2),
            "did_p": round(float(p_dd), 3),
        },
        "robustness": rob.to_dict("records"),
        "robustness_flags": {
            "G_negative_all_variants": g_signs_stable,
            "D_negative_all_variants": d_signs_stable,
            "micro_label_flips_without_training": micro_label_flips,
        },
        "clustering": {
            "labels": {b: int(l) for b, l in zip(BANDS, labels)},
            "silhouette": round(sil, 3),
            "leave_one_feature_out_stable": f"{stable}/{X.shape[1]}",
            "seed": SEED,
        },
    }
    return d


# ===========================================================================
# Verification against the values quoted in the dissertation
# ===========================================================================

# (description, recomputed key, value printed in the dissertation, tolerance pp)
def verify(ind: pd.DataFrame, bics: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    rule("VERIFICATION  recomputed values against the dissertation text")

    def band(b, col):
        return float(ind.loc[ind["size_band"] == b, col].iloc[0]) * 100

    def sect(s):
        return float(bics.loc[bics["sector"] == s, "ai_use_rate"].iloc[0]) * 100

    checks = [
        ("Abstract / RQ1: Defence adoption", sect("Defence"), 35.2),
        ("Abstract / RQ1: Life Sciences adoption", sect("Life Sciences"), 51.2),
        ("RQ1: six-sector mean", float(bics["ai_use_rate"].mean()) * 100, 45.5),
        ("RQ1: Defence LLM rate", float(bics.loc[bics["sector"] == "Defence", "ai_llm_rate"].iloc[0]) * 100, 7.8),
        ("RQ1: Creative LLM rate", float(bics.loc[bics["sector"] == "Creative Industries", "ai_llm_rate"].iloc[0]) * 100, 30.1),
        ("RQ2: micro training", band("Micro", "T"), 14.4),
        ("RQ2: large training", band("Large", "T"), 84.4),
        ("RQ2: micro board responsibility", band("Micro", "B"), 28.8),
        ("RQ2: large board responsibility", band("Large", "B"), 68.0),
        ("RQ2: micro policy", band("Micro", "P"), 31.2),
        ("RQ2: large policy", band("Large", "P"), 89.1),
        ("RQ2: micro composite C", band("Micro", "C"), 24.8),
        ("RQ2: large composite C", band("Large", "C"), 80.5),
        ("RQ2: micro breach rate", band("Micro", "R"), 41.7),
        ("RQ2: small breach rate", band("Small", "R"), 46.3),
        ("RQ2: medium breach rate", band("Medium", "R"), 64.8),
        ("RQ2: large breach rate", band("Large", "R"), 68.5),
        ("RQ3: micro AI adoption", band("Micro", "A"), 20.1),
        ("RQ3: large AI adoption", band("Large", "A"), 44.7),
        ("RQ3: micro G", band("Micro", "G"), -4.7),
        ("RQ3: large G", band("Large", "G"), -35.8),
        ("RQ3: micro V", band("Micro", "V"), 21.5),
        ("RQ3: small V", band("Small", "V"), 27.3),
        ("RQ3: medium V", band("Medium", "V"), 45.6),
        ("RQ3: large V", band("Large", "V"), 65.7),
        ("RQ3: micro D", band("Micro", "D"), -3.3),
        ("RQ3: small D", band("Small", "D"), -13.4),
        ("RQ3: medium D", band("Medium", "D"), -14.5),
        ("RQ3: large D", band("Large", "D"), -14.7),
        ("RQ3: micro unmanaged U", band("Micro", "U"), 78.5),
        ("RQ3: small unmanaged U", band("Small", "U"), 72.7),
        ("RQ3: medium unmanaged U", band("Medium", "U"), 54.4),
        ("RQ3: large unmanaged U", band("Large", "U"), 34.3),
    ]

    rows = []
    for desc, got, quoted in checks:
        delta = got - quoted
        rows.append({"check": desc, "recomputed": round(got, 2),
                     "in_dissertation": quoted, "delta_pp": round(delta, 2),
                     "match": bool(abs(delta) <= 0.06)})
    v = pd.DataFrame(rows)

    n_ok = int(v["match"].sum())
    print(f"\n{n_ok} of {len(v)} quoted values reproduce to within 0.06 pp "
          "(rounding tolerance).\n")
    bad = v[~v["match"]]
    if bad.empty:
        print("No discrepancies.")
    else:
        print("DISCREPANCIES REQUIRING ATTENTION:\n")
        print(bad.to_string(index=False))

    v.to_csv(OUT / "verification.csv", index=False)

    lines = ["# Verification report", "",
             f"Recomputed {len(v)} values quoted in the dissertation directly from the "
             "two source spreadsheets.", "",
             f"**{n_ok} of {len(v)} reproduce to within 0.06 pp.**", "",
             "| Check | Recomputed | In dissertation | Delta (pp) | Match |",
             "|---|---:|---:|---:|:--:|"]
    for r in v.itertuples():
        lines.append(f"| {r.check} | {r.recomputed} | {r.in_dissertation} | "
                     f"{r.delta_pp:+.2f} | {'yes' if r.match else 'NO'} |")
    (OUT / "verification_report.md").write_text("\n".join(lines), encoding="utf-8")

    results["verification"] = {"n_checks": len(v), "n_match": n_ok,
                               "discrepancies": bad.to_dict("records")}
    return v


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("Analysis of UK national survey evidence on AI adoption and cyber capability")
    print("Structured to the three research questions of the final dissertation.")
    print(f"Random seed fixed at {SEED}; all results are deterministic.")

    bics = run_rq1()
    ind = ld.build_indicator_frame(str(CSBS_PATH))

    # persist the indicator frame with intervals attached
    ind_out = ind.copy()
    for key in ["T", "B", "P", "A", "R"]:
        ci = [sc.wilson_ci(p, n) for p, n in zip(ind[key], ind["n_all"])]
        ind_out[f"{key}_ci_low"] = [c[0] for c in ci]
        ind_out[f"{key}_ci_high"] = [c[1] for c in ci]
    ci_v = [sc.wilson_ci(p, n) for p, n in zip(ind["V"], ind["n_ai"])]
    ind_out["V_ci_low"] = [c[0] for c in ci_v]
    ind_out["V_ci_high"] = [c[1] for c in ci_v]
    ind_out.to_csv(OUT / "indicators.csv", index=False)

    run_rq2(ind)
    d = run_rq3(ind)
    verify(ind, bics, d)

    (OUT / "results.json").write_text(json.dumps(results, indent=2, default=str),
                                      encoding="utf-8")
    rule("DONE")
    print(f"\nOutputs written to {OUT}")
    for f in sorted(OUT.glob("*")):
        print("  ", f.name)


if __name__ == "__main__":
    main()
