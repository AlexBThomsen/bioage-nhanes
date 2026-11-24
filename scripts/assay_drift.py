"""Assay drift across NHANES cycles and its effect on biological age.

RDW jumps ~0.9 percentage points between 2011-12 and 2013-14 (an analyzer change, not biology). This script
quantifies the effect and re-scores the 2015-18 external check after quantile-mapping 2013+ RDW onto the
2011-12 distribution (adjacent cycles; assumes the population RDW distribution didn't change in 2 years).

Usage:  PYTHONPATH=src python scripts/assay_drift.py   (after run_models.py)
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import joblib  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sksurv.metrics import concordance_index_censored  # noqa: E402

from bioage import models as M  # noqa: E402
from bioage.phenoage import MARKERS, phenoage  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
REP, FIG = ROOT / "reports", ROOT / "reports" / "figures"


def harmonize_rdw(df: pd.DataFrame) -> pd.DataFrame:
    """Quantile-map RDW of cycles >= 2013 onto the 2011-12 distribution."""
    df = df.copy()
    ref = np.sort(df.loc[df.cycle == 2011, "rdw"].dropna().to_numpy())
    for c in df.loc[df.cycle >= 2013, "cycle"].unique():
        m = (df.cycle == c) & df.rdw.notna()
        pct = df.loc[m, "rdw"].rank(pct=True).to_numpy()
        df.loc[m, "rdw"] = np.quantile(ref, np.clip(pct, 0, 1))
    return df


def main():
    df = pd.concat([pd.read_parquet(ROOT / "data/processed/nhanes3.parquet"),
                    pd.read_parquet(ROOT / "data/processed/nhanes.parquet")], ignore_index=True)
    med = df.groupby("cycle")[["rdw", "alp", "albumin", "creatinine"]].median()

    bundle = joblib.load(ROOT / "models" / "xgboost_C.joblib")
    xgb, ref = bundle["model"], bundle["age_ref"]
    out = {"rdw_median_by_cycle": med.rdw.round(2).to_dict()}
    for label, data in [("as measured", df), ("RDW harmonized", harmonize_rdw(df))]:
        ex = data[data.split == "external"].dropna(subset=MARKERS + ["age", "dead", "time_years"]).copy()
        ex["time_years"] = ex.time_years.clip(lower=1 / 24)
        y = M.y_of(ex)
        pa = phenoage(ex)
        bio = ref.bio_age(1 - xgb.surv(ex, [10.0])[:, 0])
        out[label] = {
            "external_n": int(len(ex)),
            "mean_phenoage_accel": float((pa - ex.age).mean()),
            "mean_xgb_bioage_accel": float((bio - ex.age).mean()),
            "c_phenoage": float(concordance_index_censored(y["event"], y["time"], pa.to_numpy())[0]),
            "c_xgb": float(concordance_index_censored(y["event"], y["time"], xgb.risk(ex))[0]),
        }
    tv = df[df.split.isin(["train", "val", "test"])].dropna(subset=MARKERS + ["age"])
    out["reference_mean_phenoage_accel_1999_2010"] = float((phenoage(tv) - tv.age).mean())
    (REP / "assay_drift.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, c in zip(axes, med.columns):
        ax.plot(med.index, med[c], "o-")
        ax.axvspan(2012, 2018.5, color="#E45756", alpha=0.07)
        ax.set_title(f"median {c}", fontsize=10)
        ax.set_xticks([1988, 1999, 2005, 2011, 2017])
    fig.suptitle("Lab values by NHANES cycle (shaded: after the 2013 haematology analyzer change)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "assay_drift.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
