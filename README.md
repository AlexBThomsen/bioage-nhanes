# bioage

Biological age from routine blood tests + blood pressure, trained on NHANES 1999–2018 linked to death records (follow-up to 31 Dec 2019), validated on NHANES III (1988–94), plus a Mendelian randomization test of pulse pressure.

- **Paper draft:** [`paper/preprint.md`](paper/preprint.md) (not peer reviewed)
- **Pre-registration and deviations:** [`PREREGISTRATION.md`](PREREGISTRATION.md)
- **Result files:** `reports/` (JSON, logs, figures)

Headline findings: gradient-boosted Cox on routine blood markers beats PhenoAge (C 0.873 vs 0.858); BP adds little to mortality prediction; a 39-marker panel adds a small gain that transfers across laboratories in ranking but not in calibration (ALT, bicarbonate assay shifts); pulse pressure is not causally linked to PhenoAge acceleration in MR.

All data are public (CDC NHANES, GWAS Catalog, figshare, 1000 Genomes) and are downloaded by the scripts; none are stored in this repository.

## Setup

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Python 3.12 is used because scikit-survival / torch don't ship wheels for 3.14 yet.

## Run

```bash
export PYTHONPATH=src
.venv/bin/python -m bioage.download          # ~140 files, ~400 MB, cached in data/raw/
.venv/bin/python -m bioage.build             # -> data/processed/nhanes.parquet (one row per adult)
.venv/bin/python -m pytest -q tests          # formula + data sanity checks
.venv/bin/python scripts/run_models.py       # all models x feature sets, ~20 min (bootstrap CIs)
.venv/bin/python scripts/bp_deepdive.py      # blood pressure analyses (run after run_models.py)
.venv/bin/python -m bioage.nhanes3           # NHANES III 1988-94 (~320 MB) -> data/processed/nhanes3.parquet
.venv/bin/python scripts/nhanes3_analysis.py # PhenoAge on its training cohort, backward validation, pooled stroke
.venv/bin/python scripts/assay_drift.py      # lab drift across cycles, RDW-harmonized external check
.venv/bin/python scripts/full_panel.py       # pre-registered: 39-marker panel, leave-one-era-out (~40 min)
.venv/bin/python scripts/mr_pulse_pressure.py # pre-registered: Mendelian randomization (needs data/raw/mr, see below)
.venv/bin/python scripts/nhanes3_external.py # pre-registered: external validation of the full panel in NHANES III
```

Mendelian randomization inputs go in `data/raw/mr/`: Evangelou 2018 PP and SBP summary statistics (GWAS Catalog
GCST006629, GCST006624), Kuo 2021 PhenoAgeAccel (figshare 12620291), and the 1000 Genomes EUR reference
(`1kg.v3.tgz` from the MRC IEU, extracted to `1kg/EUR.{bed,bim,fam}`). See `PREREGISTRATION.md` for the plan and deviations.

## Serve

```bash
.venv/bin/uvicorn bioage.api:app --reload    # standalone; POST /bioage/predict, GET /bioage/model-info
```

To mount in an existing FastAPI app: `from bioage.api import router; app.include_router(router)`.
Inputs are SI units (g/L, umol/L, mmol/L, CRP mg/L, HbA1c mmol/mol); optional fields are median-imputed and
listed in `imputed_fields`. The model file `models/xgboost_C.joblib` is written by `run_models.py`.

## Layout

```text
src/bioage/
  download.py   fetch NHANES cycles + mortality files
  build.py      join, clean, derive features/outcomes, assign splits
  phenoage.py   Levine 2018 Phenotypic Age formula
  models.py     PhenoAge, Cox, Coxnet, RSF, XGBoost-Cox, DeepSurv behind one interface; bio-age mapping
  evaluate.py   C-index (Harrell, Uno), time-dependent AUC, integrated Brier, calibration, bootstrap
  competing.py  Fine-Gray via IPCW-weighted Cox with cluster-robust SEs (fast own fitter)
  mr.py         two-sample MR: harmonization, IVW, MR-Egger, weighted median, Steiger, MVMR, LD clumping
  nhanes3.py    parse NHANES III fixed-width files into the same columns
  predict.py    one-person prediction from a lab report (SI units)
  api.py        FastAPI router around predict.py
scripts/
  run_models.py   train on 1999-2006, select on 2007-08, report on 2009-10, external check 2015-18
  bp_deepdive.py  BP vs bio-age, stroke competing risks, pulse pressure (+ x age), awareness, sensitivity
  nhanes3_analysis.py  PhenoAge on NHANES III, backward validation, pooled stroke (815 stroke deaths)
  assay_drift.py  lab drift by cycle; RDW-harmonized external check
notebooks/01_explore.ipynb  marker trends by cycle, PhenoAge vs age, SBP by age/sex
reports/          results CSV/JSON, figures, RESULTS.md
```

## Gotchas found while building

- **macOS + OpenMP:** xgboost and torch each bundle `libomp`; with more than one thread they segfault or deadlock. `bioage/__init__.py` sets `OMP_NUM_THREADS=1`, and `models.py` imports xgboost before torch.
- **SAS missing codes:** `read_sas` returns tiny floats (~5e-79) for some missing values. They are cleaned in `build.derive`, which must not also turn real zeros into NaN (there is a regression test for this).
- **CRP units:** `LBXCRP` (1999–2010) is mg/dL; `LBXHSCRP` (2015–2018) is mg/L, so divide by 10.
- **2015–2018 causes of death:** the public-use files only distinguish heart disease (001), cancer (002) and "other" (010). Stroke analyses therefore use 1999–2014.
- **Age top-coding:** 85+ in 1999–2006, 80+ from 2007. See the `age_topcoded` flag.
- **PhenoAge predicts aging-related death, not all-cause death.** Its predicted 10-yr risk matches observed deaths excluding accidents, flu/pneumonia and "other" causes, so it under-predicts all-cause risk by design. Constants are taken verbatim from the BioAge R package; an early version mistyped the denominator (0.09165 vs 0.090165), and a test now pins it.
- **Assay drift:** creatinine is recalibrated for 1999-2000 and 2005-06 (CDC equations) and NHANES III. RDW rises ~0.9 points from 2013 (analyzer change), which adds ~3 PhenoAge years to 2015-18 participants. See `scripts/assay_drift.py`.
- **Test horizon:** 2009–10 participants have 9–11 years of follow-up, so test metrics use 5- and 9-year horizons (not 10).
