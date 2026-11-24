"""Blood pressure deep-dive: four analyses from the spec, one chart each.

Usage:  PYTHONPATH=src python scripts/bp_deepdive.py
Writes: reports/bp_results.json, reports/figures/bp_*.png

All results are associations in observational data, not causal effects.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # see bioage/__init__.py

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from lifelines import AalenJohansenFitter, CoxPHFitter  # noqa: E402

from bioage.competing import fine_gray_data, weighted_cox  # noqa: E402
from bioage.phenoage import MARKERS, phenoage  # noqa: E402

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
REP, FIG = ROOT / "reports", ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
OUT: dict = {}

BASE_COVS = ["age", "female", "race_mexam", "race_othhisp", "race_black", "race_other",
             "current_smoker", "ever_smoker", "bmi"]
FULL_COVS = BASE_COVS + ["diabetes", "total_chol", "hdl", "on_bp_meds", "prior_cvd"]
STAGES = ["normal", "elevated", "stage1", "stage2"]


def load():
    return prep(pd.read_parquet(ROOT / "data" / "processed" / "nhanes.parquet"))


def prep(df):
    df = df.copy()
    for code, name in [(1, "mexam"), (2, "othhisp"), (4, "black"), (5, "other")]:
        df[f"race_{name}"] = (df["race"] == code).astype(float)
    df["diabetes"] = ((df["hba1c"] >= 6.5) | (df["diabetes_dx"] == 1)).astype(float)
    df.loc[df["hba1c"].isna() & df["diabetes_dx"].isna(), "diabetes"] = np.nan
    df["sbp10"] = df["sbp"] / 10
    df["pp10"] = df["pulse_pressure"] / 10
    df["time_years"] = df["time_years"].clip(lower=1 / 24)
    return df


def cox_hr(d, duration, event, covs, terms, weights=None, entry=None, robust=False):
    covs = [c for c in covs if d[c].nunique() > 1]          # e.g. prior_cvd after excluding prior CVD
    cols = [duration, event] + covs + ([weights] if weights else []) + ([entry] if entry else [])
    cph = CoxPHFitter(penalizer=1e-5).fit(d[cols], duration, event, weights_col=weights,
                                          entry_col=entry, robust=robust)
    s = cph.summary
    return {t: [float(s.loc[t, "exp(coef)"]), float(s.loc[t, "exp(coef) lower 95%"]),
                float(s.loc[t, "exp(coef) upper 95%"])] for t in terms}, cph


# ---------------------------------------------------------------- 1. BP vs biological age
def bp_vs_bioage(df):
    d = df.dropna(subset=MARKERS + ["age", "sbp"] + BASE_COVS).copy()
    d["phenoage"] = phenoage(d)
    d["accel"] = d["phenoage"] - d["age"]
    f = "accel ~ sbp10 + age + female + race_mexam + race_othhisp + race_black + race_other + current_smoker + ever_smoker + bmi"
    m = smf.ols(f, d).fit(cov_type="HC3")
    m_meds = smf.ols(f + " + on_bp_meds", d).fit(cov_type="HC3")
    untreated = d[d["on_bp_meds"] == 0]
    m_untr = smf.ols(f, untreated).fit(cov_type="HC3")
    m_w = smf.wls(f, d, weights=d["wt2"].fillna(d["wt2"].median())).fit(cov_type="HC3")
    m_hi = smf.ols(f, d[d.sbp >= 120]).fit(cov_type="HC3")      # curve is U-shaped; slope above the nadir
    m_lo = smf.ols(f, d[d.sbp < 120]).fit(cov_type="HC3")

    ci = lambda mm: [float(mm.params["sbp10"]), *map(float, mm.conf_int().loc["sbp10"])]  # noqa: E731
    OUT["1_bioage"] = {
        "n": int(len(d)), "cycles": "1999-2010, 2015-2018",
        "phenoage_accel_years_per_10mmHg": ci(m),
        "adjusted_also_for_bp_meds": ci(m_meds),
        "untreated_only": ci(m_untr) + [int(len(untreated))],
        "survey_weighted": ci(m_w),
        "sbp_120_plus_only": ci(m_hi) + [int((d.sbp >= 120).sum())],
        "sbp_below_120_only": ci(m_lo) + [int((d.sbp < 120).sum())],
    }

    # optional: same with ML bio-age (feature set A; contains no BP), out-of-sample predictions
    p = ROOT / "data/processed/predictions.parquet"
    if p.exists():
        res = pd.read_csv(REP / "model_results.csv")
        a = res[(res.features == "A") & (res.model != "PhenoAge")].sort_values("val_c").iloc[-1]
        col = f"bioage_{a.model}_A"
        pr = pd.read_parquet(p)[["SEQN", "cycle", col]]
        dm = d.merge(pr, on=["SEQN", "cycle"])
        dm["accel_ml"] = dm[col] - dm["age"]
        mm = smf.ols(f.replace("accel ~", "accel_ml ~"), dm).fit(cov_type="HC3")
        OUT["1_bioage"][f"ml_{a.model}_A_accel_per_10mmHg (val/test/external only)"] = ci(mm) + [int(len(dm))]

    # spline curve
    fs = f.replace("sbp10", "cr(sbp, df=4)")
    ms = smf.ols(fs, d[(d.sbp >= 90) & (d.sbp <= 200)]).fit(cov_type="HC3")
    grid = pd.DataFrame({"sbp": np.arange(95, 196, 1.0)})
    for c in BASE_COVS:
        grid[c] = d[c].mean()
    pr = ms.get_prediction(grid).summary_frame()
    ref = pr.loc[grid.sbp == 120, "mean"].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.fill_between(grid.sbp, pr.mean_ci_lower - ref, pr.mean_ci_upper - ref, alpha=0.25, color="#E45756")
    ax.plot(grid.sbp, pr["mean"] - ref, color="#E45756", lw=2)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(120, color="grey", ls=":", lw=1)
    b = ci(m)
    ax.set_title(f"PhenoAge acceleration vs systolic BP (n={len(d):,})\n"
                 f"linear: +{b[0]:.2f} yrs per 10 mmHg (95% CI {b[1]:.2f} to {b[2]:.2f})", fontsize=10)
    ax.set_xlabel("systolic blood pressure (mmHg)")
    ax.set_ylabel("biological age difference vs 120 mmHg (years)")
    ax.text(0.01, 0.02, "adjusted for age, sex, race/ethnicity, smoking, BMI; natural spline",
            transform=ax.transAxes, fontsize=8, color="grey")
    fig.tight_layout()
    fig.savefig(FIG / "bp_1_bioage.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 2. stroke, competing risks
def fg_hr(d, covs, terms):
    """Fine-Gray sub-distribution HRs with cluster-robust 95% CIs (bioage.competing)."""
    covs = [c for c in covs if d[c].nunique() > 1]
    fg = fine_gray_data(d[["time_years"] + covs], "time_years", d["cause"].to_numpy())
    r = weighted_cox(fg, covs)
    return {t: [float(r.loc[t, "hr"]), float(r.loc[t, "lo_robust"]), float(r.loc[t, "hi_robust"])] for t in terms}


def stroke(df):
    d = df[(df.cycle <= 2013)].dropna(subset=["sbp", "dead"] + FULL_COVS).copy()
    d["cause"] = np.select([d.stroke_death == 1, d.dead == 1], [1, 2], 0)
    for s in STAGES[1:]:
        d[f"st_{s}"] = (d.bp_stage == s).astype(float)
    st = [f"st_{s}" for s in STAGES[1:]]
    res = {"n": int(len(d)), "stroke_deaths": int((d.cause == 1).sum()),
           "other_deaths": int((d.cause == 2).sum()), "cycles": "1999-2014 (2015+ public files have no stroke code)"}

    d["is_stroke"] = (d.cause == 1).astype(int)
    hr, _ = cox_hr(d, "time_years", "is_stroke", ["sbp10"] + FULL_COVS, ["sbp10", "age"])
    res["cause_specific_hr_per_10mmHg"] = hr["sbp10"]
    res["cause_specific_hr_per_year_age"] = hr["age"]
    hr_b, _ = cox_hr(d, "time_years", "is_stroke", ["sbp10"] + BASE_COVS, ["sbp10"])
    res["cause_specific_hr_per_10mmHg_basic_adjustment"] = hr_b["sbp10"]
    hr_st, _ = cox_hr(d, "time_years", "is_stroke", st + FULL_COVS, st)
    res["cause_specific_hr_by_stage_vs_normal"] = hr_st
    for nm, ev in [("all_cause", "dead"), ("cvd", "cvd_death")]:
        h, _ = cox_hr(d.assign(e=d[ev].astype(int)), "time_years", "e", ["sbp10"] + FULL_COVS, ["sbp10"])
        res[f"{nm}_hr_per_10mmHg"] = h["sbp10"]
    for lo, hi in [(20, 59), (60, 120)]:
        sub = d[(d.age >= lo) & (d.age <= hi)]
        h, _ = cox_hr(sub, "time_years", "is_stroke", ["sbp10"] + FULL_COVS, ["sbp10"])
        res[f"cause_specific_hr_per_10mmHg_age_{lo}_{hi}"] = h["sbp10"]

    hr_fg = fg_hr(d, ["sbp10"] + FULL_COVS, ["sbp10"])
    res["fine_gray_shr_per_10mmHg"] = hr_fg["sbp10"]
    hr_fgs = fg_hr(d, st + FULL_COVS, st)
    res["fine_gray_shr_by_stage_vs_normal"] = hr_fgs
    OUT["2_stroke"] = res

    # chart: cumulative incidence of stroke death by stage + HR table
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    colors = dict(zip(STAGES, ["#54A24B", "#EECA3B", "#F58518", "#E45756"]))
    for s in STAGES:
        sub = d[d.bp_stage == s]
        aj = AalenJohansenFitter(seed=0).fit(sub.time_years, sub.cause, event_of_interest=1)
        cif = aj.cumulative_density_
        ax.step(cif.index, cif.iloc[:, 0] * 100, where="post", color=colors[s],
                label=f"{s} (n={len(sub):,}, {int((sub.cause == 1).sum())} stroke deaths)")
    ax.set_xlim(0, 18)
    ax.set_xlabel("years since exam")
    ax.set_ylabel("cumulative stroke death (%)")
    ax.set_title("Stroke death by BP stage (Aalen–Johansen, unadjusted;\nother deaths as competing risk)", fontsize=10)
    ax.legend(fontsize=8)
    labels, cs, fgv = STAGES[1:], [hr_st[f"st_{s}"] for s in STAGES[1:]], [hr_fgs[f"st_{s}"] for s in STAGES[1:]]
    y = np.arange(len(labels))
    for off, vals, c, lab in [(0.12, cs, "#4C78A8", "cause-specific HR"), (-0.12, fgv, "#E45756", "Fine–Gray sub-HR")]:
        v = np.array(vals)
        ax2.errorbar(v[:, 0], y + off, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", color=c, capsize=3, label=lab)
    ax2.axvline(1, color="k", lw=0.8)
    ax2.set_xscale("log")
    ax2.set_yticks(y, labels)
    ax2.set_xlabel("hazard ratio vs normal BP (log scale)")
    ax2.set_title("Adjusted for age, sex, race, smoking, BMI, diabetes,\nlipids, BP meds, prior CVD", fontsize=10)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "bp_2_stroke.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 3. pulse pressure
def pulse_pressure(df):
    d = df[(df.cycle <= 2013)].dropna(subset=["sbp", "dbp", "dead"] + FULL_COVS).copy()
    d["is_stroke"] = d.stroke_death.astype(int)
    res = {"n": int(len(d))}
    rows = []
    for grp, sub in [("all ages", d), ("age < 60", d[d.age < 60]), ("age >= 60", d[d.age >= 60])]:
        for outcome in ["is_stroke", "dead"]:
            s = sub.assign(e=sub[outcome].astype(int))
            hr_pp, _ = cox_hr(s, "time_years", "e", ["pp10"] + FULL_COVS, ["pp10"])
            hr_both, cph_both = cox_hr(s, "time_years", "e", ["sbp10", "pp10"] + FULL_COVS, ["sbp10", "pp10"])
            _, cph_sbp = cox_hr(s, "time_years", "e", ["sbp10"] + FULL_COVS, ["sbp10"])
            lr = 2 * (cph_both.log_likelihood_ - cph_sbp.log_likelihood_)
            from scipy.stats import chi2
            rows.append({"group": grp, "outcome": "stroke death" if outcome == "is_stroke" else "all-cause death",
                         "pp_alone": hr_pp["pp10"], "pp_given_sbp": hr_both["pp10"], "sbp_given_pp": hr_both["sbp10"],
                         "lr_test_p_adding_pp": float(chi2.sf(lr, 1)),
                         "c_sbp": float(cph_sbp.concordance_index_), "c_sbp_pp": float(cph_both.concordance_index_)})
    res["models"] = rows

    # PP vs biological age, beyond SBP
    dd = df.dropna(subset=MARKERS + ["age", "sbp", "dbp"] + BASE_COVS).copy()
    dd["accel"] = phenoage(dd) - dd["age"]
    f = "accel ~ sbp10 + pp10 + age + female + race_mexam + race_othhisp + race_black + race_other + current_smoker + ever_smoker + bmi"
    m = smf.ols(f, dd).fit(cov_type="HC3")
    res["phenoage_accel_per_10mmHg_pp_given_sbp"] = [float(m.params["pp10"]), *map(float, m.conf_int().loc["pp10"])]
    OUT["3_pulse_pressure"] = res

    fig, ax = plt.subplots(figsize=(8, 4.5))
    labels, y = [], []
    for i, r in enumerate([r for r in rows if r["outcome"] == "stroke death"]):
        for j, (k, c, lab) in enumerate([("pp_alone", "#F58518", "PP alone"), ("pp_given_sbp", "#4C78A8", "PP adjusted for SBP")]):
            v = r[k]
            yy = i + (0.15 if j == 0 else -0.15)
            ax.errorbar(v[0], yy, xerr=[[v[0] - v[1]], [v[2] - v[0]]], fmt="o", color=c, capsize=3,
                        label=lab if i == 0 else None)
        labels.append(f"{r['group']}\n(LR p={r['lr_test_p_adding_pp']:.2g})")
        y.append(i)
    ax.axvline(1, color="k", lw=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("stroke death hazard ratio per 10 mmHg pulse pressure")
    ax.set_title("Does pulse pressure (arterial stiffness proxy) predict stroke beyond systolic BP?", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "bp_3_pulse_pressure.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 4. unaware & untreated
def awareness(df):
    d = df.dropna(subset=["sbp", "dbp", "told_hbp"]).copy()
    high = (d.sbp >= 140) | (d.dbp >= 90)
    d["htn"] = high | (d.on_bp_meds == 1)
    h = d[d.htn].copy()
    h["group"] = np.select(
        [h.told_hbp == 0, (h.told_hbp == 1) & (h.on_bp_meds != 1), (h.on_bp_meds == 1) & high[h.index], h.on_bp_meds == 1],
        ["unaware", "aware, untreated", "treated, uncontrolled", "treated, controlled"], "other")
    h["period"] = pd.cut(h.cycle, [1998, 2006, 2014, 2018], labels=["1999-2006", "2007-2014", "2015-2018"])
    groups = ["unaware", "aware, untreated", "treated, uncontrolled", "treated, controlled"]

    measured_high = d[high]
    res = {
        "definition": "hypertension = measured >=140/90 or on BP medication",
        "n_measured_high": int(len(measured_high)),
        "share_measured_high_never_told_weighted": float(np.average(measured_high.told_hbp == 0, weights=measured_high.wt2.fillna(0) + 1e-9)),
        "share_measured_high_not_on_meds_weighted": float(np.average(measured_high.on_bp_meds != 1, weights=measured_high.wt2.fillna(0) + 1e-9)),
    }
    shares = h.groupby("period").apply(
        lambda g: pd.Series({k: np.average(g.group == k, weights=g.wt2.fillna(0) + 1e-9) for k in groups}))
    res["weighted_shares_by_period"] = shares.round(3).to_dict(orient="index")

    # outcomes vs treated-controlled
    o = h[h.group != "other"].dropna(subset=["dead"] + BASE_COVS + ["diabetes", "prior_cvd"]).copy()
    dums = []
    for g in groups[:3]:
        c = "g_" + g.replace(", ", "_").replace(" ", "_")
        o[c] = (o.group == g).astype(float)
        dums.append(c)
    covs = dums + BASE_COVS + ["diabetes", "prior_cvd"]
    hr_all, _ = cox_hr(o.assign(e=o.dead.astype(int)), "time_years", "e", covs, dums)
    o14 = o[o.cycle <= 2013]
    hr_cvd, _ = cox_hr(o14.assign(e=o14.cvd_death.astype(int)), "time_years", "e", covs, dums)
    hr_str, _ = cox_hr(o14.assign(e=o14.stroke_death.astype(int)), "time_years", "e", covs, dums)
    res["n_hypertensive"] = int(len(o))
    res["hr_vs_treated_controlled"] = {"all_cause (1999-2018)": hr_all, "cvd (1999-2014)": hr_cvd, "stroke (1999-2014)": hr_str}
    OUT["4_awareness"] = res

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#E45756", "#F58518", "#EECA3B", "#54A24B"]
    left = np.zeros(len(shares))
    for g, c in zip(groups, colors):
        ax.barh(shares.index.astype(str), shares[g] * 100, left=left, color=c, label=g)
        for i, (l, v) in enumerate(zip(left, shares[g] * 100)):
            if v > 6:
                ax.text(l + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8)
        left += shares[g].to_numpy() * 100
    ax.set_xlabel("% of adults with hypertension (survey-weighted)")
    ax.set_title("Awareness and treatment of hypertension (>=140/90 or on meds)", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)
    y = np.arange(3)
    for off, (lab, hrs), c in zip([0.2, 0, -0.2], res["hr_vs_treated_controlled"].items(), ["#4C78A8", "#E45756", "#B279A2"]):
        v = np.array([hrs[k] for k in dums])
        ax2.errorbar(v[:, 0], y + off, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", color=c, capsize=3, label=lab)
    ax2.axvline(1, color="k", lw=0.8)
    ax2.set_yticks(y, groups[:3])
    ax2.set_xlabel("hazard ratio vs treated & controlled")
    ax2.set_title("Mortality by group (adjusted for age, sex, race, smoking,\nBMI, diabetes, prior CVD)", fontsize=10)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "bp_4_awareness.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 3b. pulse pressure x age, non-linear
COVF = "female + race_mexam + race_othhisp + race_black + race_other + current_smoker + ever_smoker + bmi"


def accel_frame(df):
    d = df.dropna(subset=MARKERS + ["age", "sbp", "dbp"] + BASE_COVS).copy()
    d["accel"] = phenoage(d) - d["age"]
    d["age10c"] = (d["age"] - 50) / 10
    return d


def ols_ci(m, term):
    return [float(m.params[term]), *map(float, m.conf_int().loc[term])]


def pp_by_age(df):
    d = accel_frame(df)
    res = {}
    f = f"accel ~ sbp10 + pp10 + pp10:age10c + cr(age, df=6) + {COVF}"
    m = smf.ols(f, d).fit(cov_type="HC3")
    res["pp_years_per_10mmHg_at_age_50"] = ols_ci(m, "pp10")
    res["interaction_change_per_decade_of_age"] = ols_ci(m, "pp10:age10c")
    groups = [("20-44", 20, 44), ("45-64", 45, 64), ("65+", 65, 120)]
    by = {}
    for lab, lo, hi in groups:
        sub = d[(d.age >= lo) & (d.age <= hi)]
        mm = smf.ols(f"accel ~ sbp10 + pp10 + cr(age, df=3) + {COVF}", sub).fit(cov_type="HC3")
        by[lab] = ols_ci(mm, "pp10") + [int(len(sub))]
    res["pp_given_sbp_by_age_group"] = by

    # non-linear PP curve at fixed SBP
    dd = d[(d.pulse_pressure >= 20) & (d.pulse_pressure <= 110)]
    ms = smf.ols(f"accel ~ cr(pulse_pressure, df=4) + sbp10 + cr(age, df=6) + {COVF}", dd).fit(cov_type="HC3")
    grid = pd.DataFrame({"pulse_pressure": np.arange(25, 101, 1.0)})
    for c in BASE_COVS + ["sbp10"]:
        grid[c] = dd[c].mean()
    pr = ms.get_prediction(grid).summary_frame()
    ref = pr.loc[grid.pulse_pressure == 40, "mean"].iloc[0]
    OUT["3b_pp_age"] = res

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax.fill_between(grid.pulse_pressure, pr.mean_ci_lower - ref, pr.mean_ci_upper - ref, alpha=0.25, color="#4C78A8")
    ax.plot(grid.pulse_pressure, pr["mean"] - ref, color="#4C78A8", lw=2)
    ax.axhline(0, color="k", lw=0.8)
    ax.axvline(40, color="grey", ls=":", lw=1)
    ax.set_xlabel("pulse pressure (mmHg)")
    ax.set_ylabel("PhenoAge acceleration vs PP 40 (years)")
    ax.set_title(f"Pulse pressure vs biological age at fixed systolic BP (n={len(dd):,})\n"
                 "adjusted for spline(age), sex, race, smoking, BMI", fontsize=10)
    v = np.array([by[g][:3] for g, _, _ in groups])
    ax2.errorbar(v[:, 0], range(3), xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", color="#4C78A8", capsize=3)
    ax2.axvline(0, color="k", lw=0.8)
    ax2.set_yticks(range(3), [f"age {g} (n={by[g][3]:,})" for g, _, _ in groups])
    ax2.set_xlabel("PhenoAge yrs per +10 mmHg PP (SBP fixed)")
    ax2.set_title("Is the stiffness signal stronger at some ages?", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "bp_3b_pp_age.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 5. reverse-causation sensitivity
def sensitivity(df):
    """Re-run headline estimates (a) excluding prior CVD/cancer, (b) dropping the first 2 years of follow-up."""
    healthy = lambda x: x[(x.prior_cvd != 1) & (x.mcq_cancer != 1)]  # noqa: E731

    def landmark(x, yrs=2.0):
        x = x[x.time_years > yrs].copy()
        x["time_years"] = x["time_years"] - yrs
        return x

    early_death = lambda x: x[~((x.dead == 1) & (x.time_years <= 2))]  # noqa: E731  (cross-sectional analogue)
    scen = {"main": (lambda x: x, lambda x: x),
            "no prior CVD/cancer": (healthy, healthy),
            "skip first 2 yrs": (early_death, landmark),
            "both": (lambda x: early_death(healthy(x)), lambda x: landmark(healthy(x)))}
    rows = []
    a = accel_frame(df.dropna(subset=["dead"]))
    s = df[df.cycle <= 2013].dropna(subset=["sbp", "dbp", "dead"] + FULL_COVS).copy()
    s["cause"] = np.select([s.stroke_death == 1, s.dead == 1], [1, 2], 0)
    s["is_stroke"] = (s.cause == 1).astype(int)
    for name, (fa, fs) in scen.items():
        aa, ss = fa(a), fs(s)
        m1 = smf.ols(f"accel ~ sbp10 + cr(age, df=6) + {COVF}", aa[aa.sbp >= 120]).fit(cov_type="HC3")
        m2 = smf.ols(f"accel ~ sbp10 + pp10 + cr(age, df=6) + {COVF}", aa).fit(cov_type="HC3")
        cs, _ = cox_hr(ss, "time_years", "is_stroke", ["sbp10"] + FULL_COVS, ["sbp10"])
        fgv = fg_hr(ss, ["sbp10"] + FULL_COVS, ["sbp10"])
        pps, _ = cox_hr(ss, "time_years", "is_stroke", ["sbp10", "pp10"] + FULL_COVS, ["pp10"])
        ppa, _ = cox_hr(ss.assign(e=ss.dead.astype(int)), "time_years", "e", ["sbp10", "pp10"] + FULL_COVS, ["pp10"])
        rows.append({"scenario": name, "n_bioage": int(len(aa)), "n_stroke_cohort": int(len(ss)),
                     "stroke_deaths": int(ss.is_stroke.sum()),
                     "bioage_per_10mmHg_sbp_120plus": ols_ci(m1, "sbp10"),
                     "bioage_per_10mmHg_pp_given_sbp": ols_ci(m2, "pp10"),
                     "stroke_cs_hr_per_10mmHg": cs["sbp10"], "stroke_fg_shr_per_10mmHg": fgv["sbp10"],
                     "stroke_hr_per_10mmHg_pp_given_sbp": pps["pp10"],
                     "allcause_hr_per_10mmHg_pp_given_sbp": ppa["pp10"]})
    OUT["5_sensitivity"] = rows

    panels = [("bioage_per_10mmHg_sbp_120plus", "bio-age yrs / +10 mmHg SBP (SBP>=120)", 0),
              ("bioage_per_10mmHg_pp_given_sbp", "bio-age yrs / +10 mmHg PP (SBP fixed)", 0),
              ("stroke_cs_hr_per_10mmHg", "stroke HR / +10 mmHg SBP", 1),
              ("allcause_hr_per_10mmHg_pp_given_sbp", "all-cause HR / +10 mmHg PP (SBP fixed)", 1)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6), sharey=True)
    for ax, (k, title, null) in zip(axes, panels):
        v = np.array([r[k] for r in rows])
        y = np.arange(len(rows))[::-1]
        ax.errorbar(v[:, 0], y, xerr=[v[:, 0] - v[:, 1], v[:, 2] - v[:, 0]], fmt="o", color="#4C78A8", capsize=3)
        ax.axvline(null, color="k", lw=0.8)
        ax.set_title(title, fontsize=9)
    axes[0].set_yticks(np.arange(len(rows))[::-1], [r["scenario"] for r in rows])
    fig.suptitle("Reverse-causation checks: do the headline estimates survive?", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "bp_5_sensitivity.png", dpi=150)
    plt.close(fig)


def main():
    df = load()
    for step in (bp_vs_bioage, stroke, pulse_pressure, pp_by_age, awareness, sensitivity):
        print(f"-> {step.__name__}", flush=True)
        step(df)
    (REP / "bp_results.json").write_text(json.dumps(OUT, indent=2, default=float))
    print(json.dumps(OUT, indent=2, default=float))


if __name__ == "__main__":
    main()
