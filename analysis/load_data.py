"""
load_data.py
============
Loading, cleaning and extraction for the two national datasets.

Design principles carried over from the earlier version of this module and
retained deliberately:

  * every value is extracted by *label*, never by cell reference, so that a
    change in the layout of a published table raises an error rather than
    silently returning the wrong row;
  * suppressed cells ('c' in the CSBS, '[c]' in the BICS release) are returned
    as NaN and never imputed;
  * nothing is transcribed by hand.

What is new: effective base sizes are extracted alongside every rate. The
published tables report both a respondent count and an effective base size, and
all inference in this project uses the latter. Several CSBS questions are asked
of conditional subpopulations (Table 23 is asked only of organisations using or
considering AI, with effective bases of 273/156/143/85 rather than
985/441/259/131), so the base cannot be assumed constant across tables.

Author: Frida Alejandra Ibarra Zavaleta
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SIZE_COLS = [
    "Business size: Micro (1 to 9)",
    "Business size: Small (10 to 49)",
    "Business size: Medium (50 to 249)",
    "Business size: Large (250+)",
]
SIZE_LABELS = ["Micro", "Small", "Medium", "Large"]

SUPPRESSED = {"c", "[c]", "x", "[x]", "*", ""}

# Sectors as published in the BICS ad hoc release, in the order used in the
# dissertation's tables (descending adoption).
BICS_SECTORS = [
    "Life Sciences",
    "Creative Industries",
    "Advanced Manufacturing",
    "Digital and Technologies",
    "Professional and Business Services",
    "Defence",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num(value) -> float:
    """Coerce a published cell to float, mapping suppression markers to NaN."""
    if value is None:
        return np.nan
    s = str(value).strip().lower()
    if s in SUPPRESSED or s == "nan":
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# BICS
# ---------------------------------------------------------------------------

def load_bics_ai_usage(path: str, wave: str = "Wave 147") -> pd.DataFrame:
    """Load the BICS 'AI Current Usage' sheet for one wave.

    Returns one row per sector with the headline adoption rate and the four
    technology-type rates used in the composition analysis. The headline rate
    is derived as 1 - (not currently using), which is how the release intends
    it to be read: the technology columns are not mutually exclusive and do not
    sum to the headline.

    No base sizes are published in this release, so no significance test is
    available for any BICS comparison. Every RQ1 result is descriptive.
    """
    raw = pd.read_excel(path, sheet_name="AI Current Usage", header=None)

    header_row = None
    for i in range(raw.shape[0]):
        if str(raw.iloc[i, 0]).strip() == "Dates":
            header_row = i
            break
    if header_row is None:
        raise ValueError("Could not locate the header row in 'AI Current Usage'.")

    cols = [str(c).strip() for c in raw.iloc[header_row].tolist()]
    df = raw.iloc[header_row + 1:].copy()
    df.columns = cols
    df = df[df["Wave"].astype(str).str.strip() == wave].copy()
    if df.empty:
        raise ValueError(f"Wave {wave!r} not present in the BICS sheet.")

    df["sector"] = df["Industry"].astype(str).str.strip()

    rename = {
        "Text generation using Large Language Models": "ai_llm_rate",
        "Data processing using machine learning": "ai_ml_data_rate",
        "Image processing using machine learning": "ai_image_ml_rate",
        "Robotics": "ai_robotics_rate",
        "Business is not currently using artificial intelligence technologies": "ai_none_rate",
    }
    missing = [k for k in rename if k not in df.columns]
    if missing:
        raise ValueError(f"BICS columns missing from the release: {missing}")

    out = pd.DataFrame({"sector": df["sector"].values})
    for src, dst in rename.items():
        out[dst] = [_num(v) for v in df[src].values]
    out["ai_use_rate"] = 1 - out["ai_none_rate"]

    all_business = out[out["sector"] == "All businesses"]
    out.attrs["all_business_rate"] = (
        float(all_business["ai_use_rate"].iloc[0]) if not all_business.empty else np.nan
    )

    out = out[out["sector"].isin(BICS_SECTORS)].copy()
    out["sector"] = pd.Categorical(out["sector"], categories=BICS_SECTORS, ordered=True)
    out = out.sort_values("sector").reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# CSBS
# ---------------------------------------------------------------------------

@dataclass
class CsbsTable:
    """A parsed CSBS worksheet."""
    sheet: str
    question: str
    base_statement: str
    frame: pd.DataFrame
    respondents: dict
    effective_base: dict

    def rate(self, label: str) -> dict:
        """Return {size_label: rate} for one result row, matched exactly."""
        col = self.frame.columns[0]
        hit = self.frame[self.frame[col].astype(str).str.strip() == label]
        if hit.empty:
            raise KeyError(
                f"Result row {label!r} not found in {self.sheet}. "
                f"Available: {sorted(self.frame[col].astype(str).str.strip().unique())[:12]}"
            )
        row = hit.iloc[0]
        return {lab: _num(row[c]) for lab, c in zip(SIZE_LABELS, SIZE_COLS)}

    def rate_series(self, label: str) -> list[float]:
        d = self.rate(label)
        return [d[lab] for lab in SIZE_LABELS]

    def bases(self) -> list[float]:
        return [self.effective_base[lab] for lab in SIZE_LABELS]


def load_csbs_table(path: str, sheet: str) -> CsbsTable:
    """Parse one CSBS worksheet, capturing question text, base and bases."""
    raw = pd.read_excel(path, sheet_name=sheet, header=None)

    question = str(raw.iloc[0, 0]).strip()
    base_statement = ""
    header_row = None
    for i in range(raw.shape[0]):
        first = str(raw.iloc[i, 0]).strip()
        if first.lower().startswith("base:"):
            base_statement = first
        if first == "Result":
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"Could not locate the 'Result' header row in {sheet}.")

    cols = [str(c).strip() for c in raw.iloc[header_row].tolist()]
    body = raw.iloc[header_row + 1:].copy()
    body.columns = cols
    body = body.dropna(how="all")
    body = body[body[cols[0]].notna()]

    def _grab(row_label: str) -> dict:
        hit = body[body[cols[0]].astype(str).str.strip() == row_label]
        if hit.empty:
            return {lab: np.nan for lab in SIZE_LABELS}
        row = hit.iloc[0]
        return {lab: _num(row[c]) for lab, c in zip(SIZE_LABELS, SIZE_COLS)}

    respondents = _grab("Respondents")
    effective = _grab("Effective Base Size")

    drop = {"Respondents", "Effective Base Size"}
    body = body[~body[cols[0]].astype(str).str.strip().isin(drop)].reset_index(drop=True)

    return CsbsTable(
        sheet=sheet,
        question=question,
        base_statement=base_statement,
        frame=body,
        respondents=respondents,
        effective_base=effective,
    )


# ---------------------------------------------------------------------------
# Indicator assembly
# ---------------------------------------------------------------------------

# Exact published row labels. Kept as constants so that a change in the source
# wording fails loudly at load time rather than quietly at analysis time.
LBL_TRAINING = "Yes"
LBL_BOARD = ("Board members/trustees/a governor or senior manager with "
             "responsibility for cyber security")
LBL_POLICY = "A formal policy or policies in place covering cyber security risks"
LBL_OUTSOURCED = "An outsourced provider that manages your cyber security"
LBL_AI_ADOPTED = "We have adopted some AI tools in our organisation"
LBL_AI_PRACTICES = "Yes"
LBL_BREACH_NET = "NET: Any cyber security breaches/attacks, including phishing"


def build_indicator_frame(path: str) -> pd.DataFrame:
    """Assemble every indicator used in Chapters 4 and 5, with its own base.

    Returns one row per size band. Note the two distinct base columns: `n_all`
    applies to indicators asked of all organisations, `n_ai` to Table 23, which
    is asked only of organisations using or considering AI.
    """
    t9 = load_csbs_table(path, "Table9")     # governance arrangements
    t19 = load_csbs_table(path, "Table19")   # training
    t22 = load_csbs_table(path, "Table22")   # organisational AI use
    t23 = load_csbs_table(path, "Table23")   # AI-specific cyber practices
    t28 = load_csbs_table(path, "Table28")   # breaches

    # Guard: the general-population tables must share one base.
    base_all = t9.bases()
    for t in (t19, t22, t28):
        if not np.allclose(t.bases(), base_all):
            raise ValueError(
                f"{t.sheet} effective bases {t.bases()} differ from Table9 {base_all}; "
                "these indicators cannot be treated as sharing a base."
            )

    df = pd.DataFrame({
        "size_band": SIZE_LABELS,
        "n_all": base_all,
        "n_ai": t23.bases(),
        "respondents_all": [t9.respondents[l] for l in SIZE_LABELS],
        "T": t19.rate_series(LBL_TRAINING),          # training
        "B": t9.rate_series(LBL_BOARD),              # board responsibility
        "P": t9.rate_series(LBL_POLICY),             # formal policy
        "outsourced": t9.rate_series(LBL_OUTSOURCED),
        "A": t22.rate_series(LBL_AI_ADOPTED),        # AI adoption
        "V": t23.rate_series(LBL_AI_PRACTICES),      # AI-specific practices
        "R": t28.rate_series(LBL_BREACH_NET),        # breaches
    })

    df["C"] = df[["T", "B", "P"]].mean(axis=1)       # composite cyber capability
    df["G"] = df["A"] - df["C"]                      # naive gap (not construct-valid)
    df["D"] = df["V"] - df["C"]                      # AI-specific differential
    df["U"] = 1 - df["V"]                            # unmanaged AI exposure

    df.attrs["question_text"] = {
        "T": t19.question, "B": t9.question, "P": t9.question,
        "A": t22.question, "V": t23.question, "R": t28.question,
    }
    df.attrs["base_statements"] = {
        "T": t19.base_statement, "B": t9.base_statement, "P": t9.base_statement,
        "A": t22.base_statement, "V": t23.base_statement, "R": t28.base_statement,
    }
    return df


def load_ai_intentions(path: str) -> pd.DataFrame:
    """Table 23 in full: the intentions breakdown among AI-engaged firms."""
    t23 = load_csbs_table(path, "Table23")
    rows = {
        "yes": "Yes",
        "may_within_12m": "No – but we may implement these in the next 12 months",
        "no_plans": "No – and no plans to do so now or in the next twelve months",
        "dont_know": "Don't know",
    }
    out = {"size_band": SIZE_LABELS, "n_ai": t23.bases()}
    for key, label in rows.items():
        try:
            out[key] = t23.rate_series(label)
        except KeyError:
            out[key] = [np.nan] * 4
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Backwards compatibility with the original module
# ---------------------------------------------------------------------------

def build_csbs_size_summary(path: str) -> pd.DataFrame:
    """Legacy interface retained so the original notebook still executes."""
    df = build_indicator_frame(path)
    return pd.DataFrame({
        "size_band": df["size_band"],
        "cyber_training_rate": df["T"],
        "board_responsibility_rate": df["B"],
        "cyber_policy_rate": df["P"],
        "ai_use_org_rate_csbs": df["A"],
        "breach_rate": df["R"],
    })
