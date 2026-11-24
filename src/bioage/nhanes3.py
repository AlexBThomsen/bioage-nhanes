"""NHANES III (1988-1994) -> same columns as the main table. PhenoAge's original training cohort.

Usage:  python -m bioage.nhanes3      (downloads ~320 MB once, writes data/processed/nhanes3.parquet)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .build import read_mortality
from .download import RAW, ROOT

D3 = RAW / "nhanes3"
OUT = ROOT / "data" / "processed" / "nhanes3.parquet"
BASE = "https://wwwn.cdc.gov/nchs/data/nhanes3/1a/"
MORT = ("https://ftp.cdc.gov/pub/Health_Statistics/NCHS/datalinkage/linked_mortality/"
        "NHANES_III_MORT_2019_PUBLIC.dat")

VARS = {
    "lab": ["SEQN", "CRP", "CEP", "SGP", "AMP", "APPSI", "LMPPCNT", "MVPSI", "RWP", "WCPSI", "TCP", "HDP", "GHP",
            # full-panel extras (same analyte as the 1999-2018 lab_* columns)
            "RCP", "HGP", "HTP", "MCPSI", "MHP", "PLP", "PVPSI", "ATPSI", "ASPSI", "GGPSI", "LDPSI", "BUP",
            "CAPSI", "C3PSI", "FEP", "PSP", "TBP", "TPP", "TGP", "UAP", "NAPSI", "SKPSI", "CLPSI", "OSPSI", "GBP"],
    "exam": ["SEQN", "HSAGEIR", "HSSEX", "DMARETHN", "WTPFEX6", "SDPSTRA6", "SDPPSU6",
             "PEPMNK1R", "PEPMNK5R", "BMPBMI", "BMPWAIST"],
    "adult": ["SEQN", "HAE2", "HAE5A", "HAR1", "HAR3", "HAD1", "HAC1C", "HAC1D", "HAC1O", "HAF10"],
}


def download():
    D3.mkdir(parents=True, exist_ok=True)
    files = [(BASE + f"{f}.{e}", D3 / f"{f}.{e}") for f in VARS for e in ("dat", "sas")] + [(MORT, D3 / "MORT.dat")]
    for url, dest in files:
        if not (dest.exists() and dest.stat().st_size > 0):
            r = requests.get(url, timeout=600)
            r.raise_for_status()
            dest.write_bytes(r.content)


def positions(sas: Path) -> dict[str, tuple[int, int]]:
    """Column positions from the INPUT block of a NHANES III SAS program."""
    txt = sas.read_text(errors="ignore")
    block = txt[re.search(r"^\s*INPUT\s*$", txt, re.M).end():]
    block = block[:block.index(";")]
    pos = {}
    for name, a, b in re.findall(r"([A-Z][A-Z0-9_]*)\s+(\d+)(?:-(\d+))?", block):
        pos[name] = (int(a), int(b or a))
    return pos


def read_fixed(name: str) -> pd.DataFrame:
    pos = positions(D3 / f"{name}.sas")
    cols = {v: pos[v] for v in VARS[name]}
    rows = {v: [] for v in cols}
    with open(D3 / f"{name}.dat", encoding="latin-1") as fh:
        for line in fh:
            for v, (a, b) in cols.items():
                rows[v].append(line[a - 1:b])
    out = {}
    for v, raw in rows.items():
        s = pd.Series(raw)
        width = cols[v][1] - cols[v][0] + 1
        # missing / not-applicable codes fill the whole field with 8s or 9s (e.g. '888', '8888', '9999.9')
        filled = s.str.len().eq(width) & s.str.fullmatch(r"[8.]+|[9.]+") & ~s.str.startswith(" ")
        out[v] = pd.to_numeric(s.str.strip(), errors="coerce").mask(filled)
    return pd.DataFrame(out)


def build() -> pd.DataFrame:
    lab, exam, adult = read_fixed("lab"), read_fixed("exam"), read_fixed("adult")
    m = exam.merge(lab, on="SEQN", how="left").merge(adult, on="SEQN", how="left")
    mort = read_mortality(D3 / "MORT.dat")
    m = m.merge(mort, on="SEQN", how="left")

    yes = lambda s: s.map({1: 1.0, 2: 0.0})  # noqa: E731
    cr_mgdl = m["CEP"] * 0.960 - 0.184           # CDC-recommended NHANES III creatinine recalibration
    d = pd.DataFrame({
        "SEQN": m["SEQN"].astype(int), "cycle": 1988, "split": "nhanes3",
        "age": m["HSAGEIR"], "female": (m["HSSEX"] == 2).astype(float),
        "race": m["DMARETHN"].map({1: 3, 2: 4, 3: 1, 4: 5}),          # -> RIDRETH1 coding
        "wt2": m["WTPFEX6"], "strata": m["SDPSTRA6"], "psu": m["SDPPSU6"],
        "albumin": m["AMP"] * 10, "creatinine": cr_mgdl * 88.4017, "glucose": m["SGP"] * 0.0555,
        "crp": m["CRP"], "lymph_pct": m["LMPPCNT"], "mcv": m["MVPSI"], "rdw": m["RWP"],
        "alp": m["APPSI"], "wbc": m["WCPSI"],
        "total_chol": m["TCP"], "hdl": m["HDP"], "hba1c": m["GHP"],
        "sbp": m["PEPMNK1R"], "dbp": m["PEPMNK5R"].where(m["PEPMNK5R"] > 0),
        "bmi": m["BMPBMI"], "waist": m["BMPWAIST"],
        "told_hbp": yes(m["HAE2"]),
        "diabetes_dx": yes(m["HAD1"]),
        "mcq_chf": yes(m["HAC1C"]), "mcq_stroke": yes(m["HAC1D"]), "mcq_mi": yes(m["HAF10"]),
        "mcq_cancer": yes(m["HAC1O"]),
        "eligstat": m["eligstat"], "dead": m["mortstat"], "time_months": m["permth_exm"],
        "cause": m["ucod_leading"],
    })
    # full-panel extras, converted to the units of the 1999-2018 columns (see build.COLS). White-cell differential
    # percentages are not mapped: NHANES III used a three-part Coulter differential.
    extras = {"lab_rbc": m["RCP"], "lab_hgb": m["HGP"], "lab_hct": m["HTP"], "lab_mch": m["MCPSI"],
              "lab_mchc": m["MHP"], "lab_plt": m["PLP"], "lab_mpv": m["PVPSI"], "lab_alt": m["ATPSI"],
              "lab_ast": m["ASPSI"], "lab_ggt": m["GGPSI"], "lab_ldh": m["LDPSI"], "lab_bun": m["BUP"],
              "lab_calcium": m["CAPSI"] * 4.008, "lab_bicarb": m["C3PSI"], "lab_iron": m["FEP"], "lab_phos": m["PSP"],
              "lab_bili": m["TBP"], "lab_protein": m["TPP"], "lab_trig": m["TGP"], "lab_uric": m["UAP"],
              "lab_sodium": m["NAPSI"], "lab_potassium": m["SKPSI"], "lab_chloride": m["CLPSI"],
              "lab_osmo": m["OSPSI"], "lab_globulin": m["GBP"], "lab_chol_biopro": m["TCP"]}
    for k, v in extras.items():
        d[k] = v
    d["on_bp_meds"] = np.where(m["HAE5A"] == 1, 1.0, np.where(d["told_hbp"].notna(), 0.0, np.nan))
    d["smoking"] = np.select([m["HAR1"] == 2, m["HAR3"] == 1, m["HAR3"] == 2], ["never", "current", "former"], None)
    d["current_smoker"] = (d["smoking"] == "current").astype(float).where(d["smoking"].notna())
    d["ever_smoker"] = d["smoking"].isin(["current", "former"]).astype(float).where(d["smoking"].notna())
    d["prior_cvd"] = d[["mcq_stroke", "mcq_mi", "mcq_chf"]].max(axis=1)
    d["pulse_pressure"] = d["sbp"] - d["dbp"]
    d["time_years"] = d["time_months"] / 12
    d["stroke_death"] = (d["cause"] == 5).astype(float).where(d["dead"].notna())
    d["cvd_death"] = d["cause"].isin([1, 5]).astype(float).where(d["dead"].notna())
    s, dd = d["sbp"], d["dbp"]
    d["bp_stage"] = np.select([s.isna(), (s >= 140) | (dd >= 90), (s >= 130) | (dd >= 80), s >= 120],
                              [None, "stage2", "stage1", "elevated"], default="normal")
    # adults, linkage-eligible, and examined (people interviewed but not examined have no exam follow-up time)
    d = d[(d["age"] >= 20) & (d["eligstat"] == 1) & d["time_years"].notna()].reset_index(drop=True)
    return d


def main():
    download()
    d = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT, index=False)
    from .phenoage import MARKERS
    print(f"{len(d)} adults; PhenoAge-complete {d[MARKERS + ['age']].notna().all(axis=1).sum()}; "
          f"deaths {int(d.dead.sum())}; stroke deaths {int(d.stroke_death.sum())}; "
          f"median follow-up {d.time_years.median():.1f} y")
    print(d[["age", "albumin", "creatinine", "glucose", "crp", "lymph_pct", "mcv", "rdw", "alp", "wbc",
             "sbp", "dbp", "bmi", "hba1c", "total_chol"]].describe().T[["count", "mean", "50%", "min", "max"]].round(2))


if __name__ == "__main__":
    main()
