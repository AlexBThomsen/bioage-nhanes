"""Analysis 2 (pre-registered, PREREGISTRATION.md): two-sample MR of pulse pressure on PhenoAge acceleration.

Exposure: Evangelou 2018 (GCST006629 pulse pressure, GCST006624 systolic BP), ~750k, mmHg per allele.
Outcome:  Kuo 2021 PhenoAgeAccel GWAS (figshare 12620291), UK Biobank N=107,460, years per allele.
Usage:    PYTHONPATH=src python scripts/mr_pulse_pressure.py
Writes:   reports/mr_results.json, reports/figures/mr_*.png
"""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bioage import mr  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/mr"
REP, FIG = ROOT / "reports", ROOT / "reports" / "figures"
P_THRESH = 5e-8
N_KUO = 107_460
MHC = (6, 25_000_000, 34_000_000)   # long-range LD; distance clumping can't handle it, so excluded


def load_evangelou(name: str) -> pd.DataFrame:
    cache = RAW / f"{name}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    # header line carries stray bytes before "MarkerName", so name the 9 columns explicitly
    names = ["MarkerName", "Allele1", "Allele2", "Freq1", "Effect", "StdErr", "P", "TotalSampleSize", "N_effective"]
    d = pd.read_csv(RAW / f"{name}.txt.gz", sep=r"\s+", header=None, skiprows=1, names=names, dtype={"MarkerName": str})
    parts = d["MarkerName"].str.split(":", expand=True)
    d = d[parts[2] == "SNP"].copy()
    parts = parts[parts[2] == "SNP"]
    d["chr"] = pd.to_numeric(parts[0], errors="coerce")
    d["pos"] = pd.to_numeric(parts[1], errors="coerce")
    d = d.dropna(subset=["chr", "pos"])
    out = pd.DataFrame({"SNP": d.chr.astype(int).astype(str) + ":" + d.pos.astype(int).astype(str),
                        "chr": d.chr.astype(int), "pos": d.pos.astype(int),
                        "ea": d.Allele1.str.upper(), "oa": d.Allele2.str.upper(), "eaf": d.Freq1,
                        "beta": d.Effect, "se": d.StdErr, "p": d.P, "n": d.TotalSampleSize})
    out.to_parquet(cache, index=False)
    return out


def load_kuo() -> pd.DataFrame:
    cache = RAW / "phenoage.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    cols = ["SNP", "CHR", "BP", "ALLELE1", "ALLELE0", "A1FREQ", "INFO", "BETA", "SE", "P_BOLT_LMM"]
    parts = []
    with tarfile.open(RAW / "phenoage.tar.gz") as tf:
        member = max(tf.getmembers(), key=lambda m: m.size)
        for d in pd.read_csv(tf.extractfile(member), sep="\t", usecols=cols, chunksize=1_000_000,
                             dtype={"SNP": str, "CHR": str, "ALLELE1": str, "ALLELE0": str}):
            d = d[pd.to_numeric(d.CHR, errors="coerce").notna() & (d.INFO >= 0.8)]
            parts.append(pd.DataFrame({
                "SNP": d.CHR.astype(int).astype(str) + ":" + d.BP.astype(str), "rsid": d.SNP,
                "chr": d.CHR.astype(int), "pos": d.BP, "ea": d.ALLELE1.str.upper(), "oa": d.ALLELE0.str.upper(),
                "eaf": d.A1FREQ.astype("float32"), "beta": d.BETA, "se": d.SE, "p": d.P_BOLT_LMM}))
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(cache, index=False)
    return out


REF = RAW / "1kg" / "EUR"
_BIM = None


def instruments(gwas: pd.DataFrame) -> pd.DataFrame:
    """Pre-registered primary: LD clumping r2 < 0.001 within 10 Mb on 1000G Europeans.
    Fallback (no reference panel): strongest SNP per 1 Mb window. MHC excluded in both."""
    global _BIM
    s = gwas[gwas.p < P_THRESH]
    s = s[~((s.chr == MHC[0]) & s.pos.between(MHC[1], MHC[2]))]
    if not REF.with_suffix(".bed").exists():
        return mr.distance_clump(s, window=1_000_000)
    if _BIM is None:
        _BIM = pd.read_csv(REF.with_suffix(".bim"), sep="\t", header=None, usecols=[0, 3], names=["chr", "pos"],
                           dtype={"chr": str})
        _BIM["chr"] = pd.to_numeric(_BIM["chr"], errors="coerce")
    n = sum(1 for _ in open(REF.with_suffix(".fam")))
    # pre-thin to the strongest SNP per 10 kb to keep the genotype matrix small; LD clumping does the rest
    s = s.assign(bin=s.pos // 10_000).sort_values("p").drop_duplicates(["chr", "bin"]).drop(columns="bin")
    return mr.ld_clump(s, _BIM, REF.with_suffix(".bed"), n, r2_max=0.001, window=10_000_000)


def run(exp: pd.DataFrame, out: pd.DataFrame, n_x: int, n_y: int, scale: float, label: str) -> tuple[dict, pd.DataFrame]:
    ins = instruments(exp)
    h = mr.harmonize(ins[["SNP", "ea", "oa", "beta", "se", "eaf"]], out[["SNP", "ea", "oa", "beta", "se", "eaf"]])
    f = mr.f_stat(h)
    hs = mr.steiger_filter(h, n_x, n_y)
    res = {"label": label, "n_instruments_clumped": int(len(ins)), "n_harmonized": int(len(h)),
           "mean_F": float(f.mean()), "min_F": float(f.min()), "scale": scale}
    for name, fn, data in [("ivw", mr.ivw, h), ("egger", mr.egger, h), ("weighted_median", mr.weighted_median, h),
                           ("ivw_steiger", mr.ivw, hs)]:
        r = fn(data)
        r = {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v for k, v in r.items()}
        r["beta_scaled"] = r["beta"] * scale
        r["ci_scaled"] = [(r["beta"] - 1.96 * r["se"]) * scale, (r["beta"] + 1.96 * r["se"]) * scale]
        res[name] = r
    res["n_steiger_kept"] = int(len(hs))
    return res, h


def main():
    pp, sbp, kuo = load_evangelou("PP"), load_evangelou("SBP"), load_kuo()
    print(f"PP {len(pp):,} SNPs, SBP {len(sbp):,}, PhenoAgeAccel {len(kuo):,}", flush=True)
    n_ev = int(pp.n.median())
    out = {"data": {"exposure": "Evangelou 2018 (GCST006629 PP, GCST006624 SBP)", "outcome": "Kuo 2021 PhenoAgeAccel",
                    "n_exposure_median": n_ev, "n_outcome": N_KUO,
                    "clumping": ("LD clumping r2<0.001 within 10 Mb, 1000G EUR (n=503), MHC excluded" if REF.with_suffix(".bed").exists() else "strongest SNP per 1 Mb window (fallback), MHC excluded")}}

    res_pp, h_pp = run(pp, kuo, n_ev, N_KUO, 10, "PP -> PhenoAgeAccel (years per 10 mmHg)")
    res_sbp, h_sbp = run(sbp, kuo, n_ev, N_KUO, 10, "SBP -> PhenoAgeAccel (years per 10 mmHg)")
    res_rev, h_rev = run(kuo, pp, N_KUO, n_ev, 1, "PhenoAgeAccel -> PP (mmHg per year)")
    out["primary_pp"], out["secondary_sbp"], out["reverse"] = res_pp, res_sbp, res_rev

    # multivariable MR: PP and SBP together (mirrors "pulse pressure at fixed systolic BP")
    lead = pd.concat([pp[pp.p < P_THRESH], sbp[sbp.p < P_THRESH]]).sort_values("p").drop_duplicates("SNP")
    lead = lead[~((lead.chr == MHC[0]) & lead.pos.between(MHC[1], MHC[2]))]
    ins = instruments(lead.assign(p=lead.p))[["SNP"]]
    a = ins.merge(pp, on="SNP").merge(sbp[["SNP", "ea", "oa", "beta", "se"]], on="SNP", suffixes=("", "_sbp"))
    a["beta_sbp"] = np.where((a.ea == a.ea_sbp) & (a.oa == a.oa_sbp), a.beta_sbp,
                             np.where((a.ea == a.oa_sbp) & (a.oa == a.ea_sbp), -a.beta_sbp, np.nan))
    a = a.dropna(subset=["beta_sbp"])
    h = mr.harmonize(a[["SNP", "ea", "oa", "beta", "se", "eaf"]], kuo[["SNP", "ea", "oa", "beta", "se", "eaf"]])
    h = h.merge(a[["SNP", "beta_sbp"]], on="SNP")
    mv = mr.mvmr_ivw(np.column_stack([h.bx, h.beta_sbp]), h.by.to_numpy(), h.sy.to_numpy(), ["PP", "SBP"])
    out["mvmr"] = {"n_snps": int(len(h)),
                   "PP_years_per_10mmHg": [float(mv.loc["PP", "beta"] * 10), float((mv.loc["PP", "beta"] - 1.96 * mv.loc["PP", "se"]) * 10),
                                           float((mv.loc["PP", "beta"] + 1.96 * mv.loc["PP", "se"]) * 10), float(mv.loc["PP", "p"])],
                   "SBP_years_per_10mmHg": [float(mv.loc["SBP", "beta"] * 10), float((mv.loc["SBP", "beta"] - 1.96 * mv.loc["SBP", "se"]) * 10),
                                            float((mv.loc["SBP", "beta"] + 1.96 * mv.loc["SBP", "se"]) * 10), float(mv.loc["SBP", "p"])]}

    ivw_pp = res_pp["ivw"]
    supported = (ivw_pp["p"] < 0.05 and np.sign(res_pp["egger"]["beta"]) == np.sign(ivw_pp["beta"])
                 and np.sign(res_pp["weighted_median"]["beta"]) == np.sign(ivw_pp["beta"]) and ivw_pp["beta"] > 0)
    out["H10_supported"] = bool(supported)
    out["observational_comparison_years_per_10mmHg_pp"] = 0.57
    (REP / "mr_results.json").write_text(json.dumps(out, indent=2, default=float))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, (h, res, xl, yl) in zip(axes, [(h_pp, res_pp, "SNP effect on pulse pressure (mmHg)", "SNP effect on PhenoAgeAccel (years)"),
                                           (h_sbp, res_sbp, "SNP effect on systolic BP (mmHg)", "SNP effect on PhenoAgeAccel (years)"),
                                           (h_rev, res_rev, "SNP effect on PhenoAgeAccel (years)", "SNP effect on pulse pressure (mmHg)")]):
        s = np.sign(h.bx)
        bx, by = h.bx * s, h.by * s
        ax.errorbar(bx, by, xerr=1.96 * h.sx, yerr=1.96 * h.sy, fmt="o", ms=3, alpha=0.35, color="#4C78A8", elinewidth=0.6)
        xs = np.linspace(0, bx.max() * 1.05, 10)
        ax.plot(xs, res["ivw"]["beta"] * xs, color="#E45756", lw=2, label=f"IVW {res['ivw']['beta_scaled']:+.3f}")
        ax.plot(xs, res["egger"]["intercept"] + res["egger"]["beta"] * xs, color="#54A24B", lw=1.5, ls="--",
                label=f"Egger {res['egger']['beta_scaled']:+.3f}")
        ax.plot(xs, res["weighted_median"]["beta"] * xs, color="#B279A2", lw=1.5, ls=":",
                label=f"W. median {res['weighted_median']['beta_scaled']:+.3f}")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"{res['label']}\n{res['n_harmonized']} SNPs, mean F {res['mean_F']:.0f}", fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "mr_scatter.png", dpi=150)
    plt.close(fig)
    print(json.dumps(out, indent=1, default=lambda x: round(float(x), 4)))


if __name__ == "__main__":
    main()
