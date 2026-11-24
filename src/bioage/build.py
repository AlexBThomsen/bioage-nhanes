"""Join NHANES tables + mortality into one analysis table (one row per adult).

Usage:  python -m bioage.build
Output: data/processed/nhanes.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .download import CYCLES, RAW, ROOT, table_names

OUT = ROOT / "data" / "processed" / "nhanes.parquet"

SPLITS = {
    1999: "train", 2001: "train", 2003: "train", 2005: "train",
    2007: "val", 2009: "test",
    2011: "unlabelled", 2013: "unlabelled",   # no CRP -> no PhenoAge features
    2015: "external", 2017: "external",
}

# output name -> candidate NHANES columns (first one present wins)
COLS = {
    # demographics / design
    "age": ["RIDAGEYR"], "sex": ["RIAGENDR"], "race": ["RIDRETH1"],
    "pir": ["INDFMPIR"], "educ": ["DMDEDUC2"],
    "wt2": ["WTMEC2YR"], "wt4": ["WTMEC4YR"],
    "strata": ["SDMVSTRA"], "psu": ["SDMVPSU"],
    # PhenoAge markers in the units the formula expects
    "albumin": ["LBDSALSI"],                 # g/L
    "creatinine": ["LBDSCRSI"],              # umol/L
    "glucose": ["LBDSGLSI"],                 # mmol/L
    "alp": ["LBXSAPSI", "LBDSAPSI"],         # U/L
    "lymph_pct": ["LBXLYPCT"],               # %
    "mcv": ["LBXMCVSI"],                     # fL
    "rdw": ["LBXRDW"],                       # %
    "wbc": ["LBXWBCSI"],                     # 1000 cells/uL
    "crp_mgdl": ["LBXCRP"],                  # mg/dL (1999-2010)
    "hscrp_mgl": ["LBXHSCRP"],               # mg/L  (2015-2018)
    # extra labs
    "total_chol": ["LBXTC"], "hdl": ["LBDHDD", "LBXHDD", "LBDHDL"],   # mg/dL
    "hba1c": ["LBXGH"],
    # body
    "bmi": ["BMXBMI"], "waist": ["BMXWAIST"],
    # full lab panel (conventional units; used by scripts/full_panel.py)
    "lab_mono_pct": ["LBXMOPCT"], "lab_neut_pct": ["LBXNEPCT"], "lab_eos_pct": ["LBXEOPCT"],
    "lab_baso_pct": ["LBXBAPCT"], "lab_rbc": ["LBXRBCSI"], "lab_hgb": ["LBXHGB"], "lab_hct": ["LBXHCT"],
    "lab_mch": ["LBXMCHSI"], "lab_mchc": ["LBXMC"], "lab_plt": ["LBXPLTSI"], "lab_mpv": ["LBXMPSI"],
    "lab_alt": ["LBXSATSI"], "lab_ast": ["LBXSASSI"], "lab_bun": ["LBXSBU"], "lab_calcium": ["LBXSCA"],
    "lab_bicarb": ["LBXSC3SI"], "lab_ggt": ["LBXSGTSI"], "lab_iron": ["LBXSIR"],
    "lab_ldh": ["LBXSLDSI", "LBDSLDSI"], "lab_phos": ["LBXSPH", "LBDSPH"], "lab_bili": ["LBXSTB", "LBDSTB"],
    "lab_protein": ["LBXSTP"], "lab_trig": ["LBXSTR"], "lab_uric": ["LBXSUA"], "lab_sodium": ["LBXSNASI"],
    "lab_potassium": ["LBXSKSI"], "lab_chloride": ["LBXSCLSI"], "lab_osmo": ["LBXSOSSI"],
    "lab_globulin": ["LBXSGB"], "lab_chol_biopro": ["LBXSCH"],
    # questionnaires
    "bpq020": ["BPQ020"], "bpq040a": ["BPQ040A"], "bpq050a": ["BPQ050A"],
    "smq020": ["SMQ020"], "smq040": ["SMQ040"],
    "diq010": ["DIQ010"],
    "mcq_stroke": ["MCQ160F"], "mcq_chd": ["MCQ160C"], "mcq_mi": ["MCQ160E"],
    "mcq_chf": ["MCQ160B"], "mcq_cancer": ["MCQ220"],
}

MORT_SPEC = [  # 1-based inclusive columns from the CDC dictionary
    ("SEQN", 1, 14), ("eligstat", 15, 15), ("mortstat", 16, 16),
    ("ucod_leading", 17, 19), ("permth_int", 43, 45), ("permth_exm", 46, 48),
]


def read_mortality(path: Path) -> pd.DataFrame:
    colspecs = [(a - 1, b) for _, a, b in MORT_SPEC]
    names = [n for n, _, _ in MORT_SPEC]
    df = pd.read_fwf(path, colspecs=colspecs, names=names, header=None, dtype=str)
    df = df.apply(lambda s: pd.to_numeric(s.str.strip(), errors="coerce"))
    return df


def bp_means(bpx: pd.DataFrame) -> pd.DataFrame:
    """Mean of available readings. Auscultatory (BPXSY*) preferred, oscillometric (BPXOSY*) fallback."""
    out = pd.DataFrame({"SEQN": bpx["SEQN"]})
    for kind, prefix_a, prefix_o in [("sbp", "BPXSY", "BPXOSY"), ("dbp", "BPXDI", "BPXODI")]:
        a = bpx[[c for c in bpx if c.startswith(prefix_a) and c[-1].isdigit()]]
        o = bpx[[c for c in bpx if c.startswith(prefix_o) and c[-1].isdigit()]]
        if kind == "dbp":   # DBP of 0 = Korotkoff sounds to zero, not a real value
            a, o = a.where(a > 0), o.where(o > 0)
        ma = a.mean(axis=1) if a.shape[1] else pd.Series(np.nan, index=bpx.index)
        mo = o.mean(axis=1) if o.shape[1] else pd.Series(np.nan, index=bpx.index)
        out[kind] = ma.fillna(mo)
        out[f"{kind}_n"] = a.notna().sum(axis=1) if a.shape[1] else 0
    out["bp_device"] = np.where(out["sbp_n"] > 0, "auscultatory", np.where(out["sbp"].notna(), "oscillometric", None))
    return out.drop(columns=["sbp_n", "dbp_n"])


def build_cycle(year: int) -> pd.DataFrame:
    cyc = f"{year}_{year + 1}"
    d = RAW / cyc
    frames = []
    for table, stems in table_names(year).items():
        parts = [pd.read_sas(d / f"{s}.xpt", format="xport") for s in stems if (d / f"{s}.xpt").exists()]
        if not parts:
            continue
        df = parts[0]
        for p in parts[1:]:
            df = df.merge(p, on="SEQN", how="outer", suffixes=("", "_dup"))
        if table == "BPX":
            df = bp_means(df)
        frames.append(df)

    base = frames[0] if "RIDAGEYR" in frames[0] else next(f for f in frames if "RIDAGEYR" in f)
    merged = base
    for f in frames:
        if f is base:
            continue
        f = f.drop(columns=[c for c in f.columns if c in merged.columns and c != "SEQN"])
        merged = merged.merge(f, on="SEQN", how="left")

    out = pd.DataFrame({"SEQN": merged["SEQN"].astype(int), "cycle": year})
    for name, cands in COLS.items():
        hit = next((c for c in cands if c in merged), None)
        out[name] = merged[hit] if hit else np.nan
    for c in ["sbp", "dbp", "bp_device"]:
        out[c] = merged[c] if c in merged else np.nan

    mort = read_mortality(d / f"MORT_{cyc}.dat")
    out = out.merge(mort, on="SEQN", how="left")
    return out


def derive(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # SAS missing codes like 5.4e-79 show up as tiny floats; treat as missing
    num = df.select_dtypes("number").columns
    df[num] = df[num].mask((df[num].abs() < 1e-20) & (df[num] != 0))

    df["crp"] = df["crp_mgdl"].fillna(df["hscrp_mgl"] / 10)          # mg/dL everywhere
    # CDC-recommended serum creatinine recalibration to the standardized (IDMS-traceable) method
    cr = df["creatinine"] / 88.4017                                   # umol/L -> mg/dL
    cr = np.where(df["cycle"] == 1999, 1.013 * cr + 0.147, np.where(df["cycle"] == 2005, -0.016 + 0.978 * cr, cr))
    df["creatinine"] = cr * 88.4017
    df["female"] = (df["sex"] == 2).astype(float)
    df["pulse_pressure"] = df["sbp"] - df["dbp"]
    df["map"] = df["dbp"] + df["pulse_pressure"] / 3

    yes = lambda s: s.map({1: 1.0, 2: 0.0})  # noqa: E731  (7/9 = refused/don't know -> NaN)
    df["told_hbp"] = yes(df["bpq020"])
    df["on_bp_meds"] = np.where(df["bpq050a"] == 1, 1.0, np.where(df["told_hbp"].notna(), 0.0, np.nan))
    df["diabetes_dx"] = yes(df["diq010"])
    for c in ["mcq_stroke", "mcq_chd", "mcq_mi", "mcq_chf", "mcq_cancer"]:
        df[c] = yes(df[c])
    df["prior_cvd"] = df[["mcq_stroke", "mcq_chd", "mcq_mi", "mcq_chf"]].max(axis=1)

    df["smoking"] = np.select(
        [df["smq020"] == 2, df["smq040"].isin([1, 2]), df["smq040"] == 3],
        ["never", "current", "former"], default=None)
    df["current_smoker"] = (df["smoking"] == "current").astype(float).where(df["smoking"].notna())
    df["ever_smoker"] = df["smoking"].isin(["current", "former"]).astype(float).where(df["smoking"].notna())

    # ACC/AHA 2017 stage from measured BP
    s, d = df["sbp"], df["dbp"]
    df["bp_stage"] = np.select(
        [s.isna(), (s >= 140) | (d >= 90), (s >= 130) | (d >= 80), s >= 120],
        [None, "stage2", "stage1", "elevated"], default="normal")
    df["measured_htn"] = ((s >= 140) | (d >= 90)).astype(float).where(s.notna())

    # outcomes
    df["dead"] = df["mortstat"]
    df["time_months"] = df["permth_exm"]
    df["time_years"] = df["permth_exm"] / 12
    df["cause"] = df["ucod_leading"]
    df["stroke_death"] = (df["ucod_leading"] == 5).astype(float).where(df["dead"].notna())
    df["cvd_death"] = df["ucod_leading"].isin([1, 5]).astype(float).where(df["dead"].notna())

    # survey weights for pooled analyses (CDC guidance): 1999-2002 use 4-yr weight
    n_cycles = df["cycle"].nunique()
    df["wt_pooled"] = np.where(df["cycle"] <= 2001, df["wt4"] * 2, df["wt2"]) / n_cycles

    df["split"] = df["cycle"].map(SPLITS)
    df["age_topcoded"] = ((df["cycle"] <= 2005) & (df["age"] >= 85)) | ((df["cycle"] >= 2007) & (df["age"] >= 80))
    return df


PHENO_MARKERS = ["albumin", "creatinine", "glucose", "crp", "lymph_pct", "mcv", "rdw", "alp", "wbc"]


def main() -> None:
    raw = pd.concat([build_cycle(y) for y in CYCLES], ignore_index=True)
    df = derive(raw)
    n0 = len(df)
    df = df[(df["age"] >= 20) & (df["eligstat"] == 1)].reset_index(drop=True)
    drop = [c for c in df.columns if c in {"sex", "bpq020", "bpq040a", "bpq050a", "smq020", "smq040",
                                            "diq010", "crp_mgdl", "hscrp_mgl", "mortstat", "permth_exm", "ucod_leading"}]
    df = df.drop(columns=drop)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)

    print(f"{n0} participants -> {len(df)} adults 20+ eligible for mortality linkage")
    complete = df[PHENO_MARKERS + ["age"]].notna().all(axis=1)
    summ = df.assign(pheno_complete=complete).groupby("cycle").agg(
        n=("SEQN", "size"), pheno_complete=("pheno_complete", "sum"),
        deaths=("dead", "sum"), stroke=("stroke_death", "sum"),
        fu_yrs=("time_years", "median"), sbp=("sbp", "mean"))
    print(summ.round(1).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
