"""MR estimators recover a known causal effect on simulated summary statistics."""
import numpy as np
import pandas as pd

from bioage.mr import distance_clump, egger, harmonize, ivw, mvmr_ivw, weighted_median


def simulate(k=200, theta=0.05, pleio=0.0, seed=0):
    rng = np.random.default_rng(seed)
    bx = rng.normal(0, 0.5, k)
    sx = np.full(k, 0.02)
    sy = np.full(k, 0.01)
    # directional pleiotropy: a shared direct effect on the outcome, signed relative to the exposure-raising allele
    by = theta * bx + pleio * np.sign(bx) + rng.normal(0, sy)
    return pd.DataFrame({"SNP": [f"rs{i}" for i in range(k)], "bx": bx, "sx": sx, "by": by, "sy": sy,
                         "eaf": rng.uniform(0.05, 0.4, k)})


def test_estimators_recover_effect():
    h = simulate()
    for est in (ivw, egger, weighted_median):
        r = est(h)
        assert abs(r["beta"] - 0.05) < 3 * r["se"] + 1e-3, est.__name__


def test_egger_detects_directional_pleiotropy():
    h = simulate(pleio=0.01)
    e = egger(h)
    assert e["intercept_p"] < 0.01 and abs(e["beta"] - 0.05) < 0.01
    assert abs(ivw(h)["beta"] - 0.05) > abs(e["beta"] - 0.05)     # IVW is biased, Egger isn't


def test_null_effect():
    h = simulate(theta=0.0, seed=3)
    assert ivw(h)["p"] > 0.01


def test_harmonize_flips_and_drops():
    exp = pd.DataFrame({"SNP": ["a", "b", "c", "d"], "ea": ["A", "C", "A", "G"], "oa": ["G", "T", "T", "A"],
                        "beta": [1.0, 1.0, 1.0, 1.0], "se": [.1] * 4, "eaf": [.2, .3, .5, .1]})
    out = pd.DataFrame({"SNP": ["a", "b", "c", "d"], "ea": ["G", "C", "A", "C"], "oa": ["A", "T", "T", "T"],
                        "beta": [0.5, 0.5, 0.5, 0.5], "se": [.1] * 4, "eaf": [.8, .3, .5, .9]})
    h = harmonize(exp, out).set_index("SNP")
    assert h.loc["a", "by"] == -0.5          # alleles swapped -> flip sign
    assert h.loc["b", "by"] == 0.5           # same orientation
    assert "c" not in h.index                # A/T palindrome at 50% frequency is ambiguous -> dropped
    assert h.loc["d", "by"] == 0.5           # opposite strand: C/T is G/A complemented, same orientation


def test_mvmr_separates_two_exposures():
    rng = np.random.default_rng(1)
    k = 300
    b1, b2 = rng.normal(0, .5, k), rng.normal(0, .5, k)
    by = 0.1 * b1 + 0.0 * b2 + rng.normal(0, .01, k)
    r = mvmr_ivw(np.column_stack([b1, b2]), by, np.full(k, .01), ["x1", "x2"])
    assert abs(r.loc["x1", "beta"] - 0.1) < 0.01 and r.loc["x2", "p"] > 0.01


def test_distance_clump():
    df = pd.DataFrame({"chr": [1, 1, 1, 2], "pos": [100, 500_000, 2_000_000, 100], "p": [1e-10, 1e-12, 1e-9, 1e-8]})
    c = distance_clump(df)
    assert sorted(zip(c.chr, c.pos)) == [(1, 500_000), (1, 2_000_000), (2, 100)]


def test_read_bed_dosages(tmp_path):
    from bioage.mr import read_bed_dosages
    # 5 samples, 2 SNPs. Codes per sample (2 bits, low bits first): 00=2, 10=1, 11=0, 01=missing
    snp1 = [0b00, 0b10, 0b11, 0b01, 0b00]
    snp2 = [0b11, 0b11, 0b10, 0b00, 0b00]
    def pack(codes):
        codes = codes + [0] * (-len(codes) % 4)
        return bytes(sum(c << (2 * k) for k, c in enumerate(codes[i:i + 4])) for i in range(0, len(codes), 4))
    p = tmp_path / "x.bed"
    p.write_bytes(b"\x6c\x1b\x01" + pack(snp1) + pack(snp2))
    G = read_bed_dosages(p, 5, np.array([1, 0]))
    assert np.allclose(G[0], [0, 0, 1, 2, 2])
    assert np.allclose(G[1][[0, 1, 2, 4]], [2, 1, 0, 2]) and np.isnan(G[1][3])
