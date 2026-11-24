"""Two-sample Mendelian randomization from GWAS summary statistics.

Estimators follow the standard definitions (Burgess et al., "Mendelian Randomization", 2nd ed.;
Bowden et al. 2015 for MR-Egger, 2016 for the weighted median). Effects are per unit of exposure.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


def harmonize(exp: pd.DataFrame, out: pd.DataFrame, palindromic_maf: float = 0.42) -> pd.DataFrame:
    """Align outcome effects to the exposure effect allele.

    Both frames need: SNP, ea (effect allele), oa (other allele), beta, se, eaf. Returns merged rows with
    columns bx, sx, by, sy, eaf. Drops allele mismatches and ambiguous palindromic SNPs (eaf in [0.42, 0.58])."""
    m = exp.merge(out, on="SNP", suffixes=("_x", "_y"))
    for c in ["ea_x", "oa_x", "ea_y", "oa_y"]:
        m[c] = m[c].str.upper()
    same = (m.ea_x == m.ea_y) & (m.oa_x == m.oa_y)
    flip = (m.ea_x == m.oa_y) & (m.oa_x == m.ea_y)
    comp = lambda s: s.map(COMPLEMENT)  # noqa: E731
    same_c = (m.ea_x == comp(m.ea_y)) & (m.oa_x == comp(m.oa_y))
    flip_c = (m.ea_x == comp(m.oa_y)) & (m.oa_x == comp(m.ea_y))
    pal = m.ea_x == comp(m.oa_x)
    m["by"] = np.where(same | same_c, m.beta_y, np.where(flip | flip_c, -m.beta_y, np.nan))
    m["eaf_y_aligned"] = np.where(same | same_c, m.eaf_y, 1 - m.eaf_y)
    ambiguous = pal & m.eaf_x.between(palindromic_maf, 1 - palindromic_maf)
    # palindromic SNPs whose strand is resolvable: check frequencies agree
    pal_mismatch = pal & ((m.eaf_x - 0.5) * (m.eaf_y_aligned - 0.5) < 0)
    keep = m.by.notna() & ~ambiguous & ~pal_mismatch
    r = m[keep].rename(columns={"beta_x": "bx", "se_x": "sx", "se_y": "sy", "eaf_x": "eaf"})
    return r[["SNP", "bx", "sx", "by", "sy", "eaf"]].reset_index(drop=True)


def ivw(h: pd.DataFrame) -> dict:
    """Inverse-variance weighted, multiplicative random effects (SE scaled by residual SD if > 1)."""
    w = 1 / h.sy ** 2
    b = np.sum(w * h.bx * h.by) / np.sum(w * h.bx ** 2)
    se_fixed = 1 / np.sqrt(np.sum(w * h.bx ** 2))
    resid = (h.by - b * h.bx) / h.sy
    q = float(np.sum(resid ** 2))
    k = len(h)
    phi = max(1.0, q / (k - 1))
    se = se_fixed * np.sqrt(phi)
    return {"beta": b, "se": se, "p": 2 * stats.norm.sf(abs(b / se)), "Q": q, "Q_p": stats.chi2.sf(q, k - 1), "n_snps": k}


def egger(h: pd.DataFrame) -> dict:
    """MR-Egger: regress by on bx with intercept, weights 1/sy^2, after orienting bx positive."""
    s = np.sign(h.bx)
    bx, by = h.bx * s, h.by * s
    w = 1 / h.sy ** 2
    X = np.column_stack([np.ones(len(h)), bx])
    W = np.diag(w)
    XtWX = X.T @ W @ X
    coef = np.linalg.solve(XtWX, X.T @ W @ by)
    resid = by - X @ coef
    k = len(h)
    sigma2 = max(1.0, float(np.sum(w * resid ** 2)) / (k - 2))
    cov = np.linalg.inv(XtWX) * sigma2
    se = np.sqrt(np.diag(cov))
    t = stats.t
    return {"beta": coef[1], "se": se[1], "p": 2 * t.sf(abs(coef[1] / se[1]), k - 2),
            "intercept": coef[0], "intercept_se": se[0], "intercept_p": 2 * t.sf(abs(coef[0] / se[0]), k - 2)}


def weighted_median(h: pd.DataFrame, n_boot: int = 1000, seed: int = 0) -> dict:
    ratio = h.by / h.bx
    se_ratio = h.sy / np.abs(h.bx)

    def wm(r, s):
        w = 1 / s ** 2
        o = np.argsort(r)
        r, w = np.asarray(r)[o], np.asarray(w)[o]
        cw = np.cumsum(w) - 0.5 * w
        cw /= w.sum()
        j = np.searchsorted(cw, 0.5)
        if j == 0:
            return r[0]
        if j >= len(r):
            return r[-1]
        return r[j - 1] + (r[j] - r[j - 1]) * (0.5 - cw[j - 1]) / (cw[j] - cw[j - 1])

    b = wm(ratio, se_ratio)
    rng = np.random.default_rng(seed)
    boots = [wm(rng.normal(h.by, h.sy) / rng.normal(h.bx, h.sx), se_ratio) for _ in range(n_boot)]
    se = float(np.std(boots))
    return {"beta": b, "se": se, "p": 2 * stats.norm.sf(abs(b / se))}


def r2_from_beta(beta, se, n, eaf):
    """Variance explained by a SNP for a continuous trait, from summary stats (approximation via t-statistic)."""
    t2 = (beta / se) ** 2
    return t2 / (t2 + n - 2)


def steiger_filter(h: pd.DataFrame, n_x: int, n_y: int) -> pd.DataFrame:
    """Keep SNPs explaining more variance in the exposure than in the outcome."""
    rx = r2_from_beta(h.bx, h.sx, n_x, h.eaf)
    ry = r2_from_beta(h.by, h.sy, n_y, h.eaf)
    return h[rx > ry].reset_index(drop=True)


def mvmr_ivw(bx: np.ndarray, by: np.ndarray, sy: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Multivariable IVW: weighted regression of outcome effects on several exposure effects, no intercept."""
    w = 1 / sy ** 2
    XtWX = bx.T @ (bx * w[:, None])
    coef = np.linalg.solve(XtWX, bx.T @ (w * by))
    resid = by - bx @ coef
    k, p = bx.shape
    phi = max(1.0, float(np.sum(w * resid ** 2)) / (k - p))
    se = np.sqrt(np.diag(np.linalg.inv(XtWX)) * phi)
    return pd.DataFrame({"beta": coef, "se": se, "p": 2 * stats.norm.sf(np.abs(coef / se))}, index=names)


def f_stat(h: pd.DataFrame) -> pd.Series:
    return (h.bx / h.sx) ** 2


def distance_clump(df: pd.DataFrame, p_col="p", window=1_000_000) -> pd.DataFrame:
    """Fallback clumping: repeatedly keep the strongest SNP and drop others within `window` bp on the same chromosome."""
    df = df.sort_values(p_col).reset_index(drop=True)
    keep = []
    taken = {}
    for i, r in df.iterrows():
        pos = taken.setdefault(r.chr, [])
        if all(abs(r.pos - q) > window for q in pos):
            keep.append(i)
            pos.append(r.pos)
    return df.loc[keep].reset_index(drop=True)


def read_bed_dosages(bed_path, n_samples: int, idx: np.ndarray) -> np.ndarray:
    """Read allele-1 dosages (0/1/2, NaN missing) for selected SNP indices from a SNP-major PLINK .bed file."""
    nbytes = (n_samples + 3) // 4
    lut = np.array([2.0, np.nan, 1.0, 0.0])          # PLINK codes 00, 01, 10, 11
    out = np.empty((len(idx), n_samples))
    with open(bed_path, "rb") as fh:
        if fh.read(3) != b"\x6c\x1b\x01":
            raise ValueError("not a SNP-major PLINK .bed file")
        for k, i in enumerate(idx):
            fh.seek(3 + int(i) * nbytes)
            raw = np.frombuffer(fh.read(nbytes), dtype=np.uint8)
            codes = np.stack([(raw >> s) & 3 for s in (0, 2, 4, 6)], axis=1).ravel()[:n_samples]
            out[k] = lut[codes]
    return out


def ld_clump(ins: pd.DataFrame, bim: pd.DataFrame, bed_path, n_samples: int,
             r2_max: float = 0.001, window: int = 10_000_000) -> pd.DataFrame:
    """Greedy LD clumping on a reference panel: keep SNPs in p-value order, dropping any with r2 > r2_max
    to an already-kept SNP within `window` bp. SNPs absent from the reference are dropped."""
    ref = bim.reset_index().rename(columns={"index": "bed_idx"})
    m = ins.merge(ref[["chr", "pos", "bed_idx"]], on=["chr", "pos"]).drop_duplicates(["chr", "pos"])
    m = m.sort_values("p").reset_index(drop=True)
    G = read_bed_dosages(bed_path, n_samples, m.bed_idx.to_numpy())
    G = np.where(np.isnan(G), np.nanmean(G, axis=1, keepdims=True), G)
    G = (G - G.mean(1, keepdims=True)) / (G.std(1, keepdims=True) + 1e-12)
    kept = []
    for i in range(len(m)):
        ok = True
        for j in kept:
            if m.chr[j] == m.chr[i] and abs(m.pos[j] - m.pos[i]) <= window:
                r = float(G[i] @ G[j]) / n_samples
                if r * r > r2_max:
                    ok = False
                    break
        if ok:
            kept.append(i)
    return m.loc[kept].drop(columns="bed_idx").reset_index(drop=True)
