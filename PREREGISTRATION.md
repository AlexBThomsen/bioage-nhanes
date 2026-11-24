# Pre-registration: full lab panel and Mendelian randomization

Written 24 Nov 2026, before either analysis was run. Self-registered (not on OSF), so it's only as good as the honesty of the timestamp. Any deviation will be reported as a deviation.

## Analysis 1: Does the full NHANES lab panel improve mortality prediction?

**Question.** The models so far use 9 PhenoAge markers (plus up to 11 clinical extras). NHANES measures about 40 chemistry and blood-count markers. Does using all of them improve prediction, and does the improvement hold up across survey eras?

**Data.** NHANES 1999–2018, adults 20+, linkage-eligible, with all 9 PhenoAge markers except CRP (CRP is missing by design in 2011–14 and is left as missing; XGBoost handles missing values natively). NHANES III is excluded because its extra markers aren't mapped.

**Extra markers (30).**
- Blood count: monocyte, neutrophil, eosinophil and basophil %, red cell count, haemoglobin, haematocrit, MCH, MCHC, platelets, mean platelet volume.
- Chemistry: ALT, AST, urea nitrogen, calcium, bicarbonate, GGT, iron, LDH, phosphorus, bilirubin, total protein, triglycerides, uric acid, sodium, potassium, chloride, osmolality, globulin, cholesterol.

**Harmonization (fixed in advance).** Creatinine recalibrated with the CDC equations (already in place). RDW from 2013 onward quantile-mapped onto the 2011–12 distribution. No other marker is harmonized.

**Design: leave-one-era-out.** Five eras: 1999–2002, 2003–06, 2007–10, 2011–14, 2015–18. For each era, train on the other four (a random 15% of training rows is held out for early stopping) and test on the held-out era. Follow-up is administratively censored at 10 years so eras are comparable. 2015–18 has only 2–4 years.

**Models.**
- M0: age-only Cox
- M1: published PhenoAge
- M2: XGBoost, 9 markers + age
- M3: XGBoost, all 39 markers + age
- M4: M3 + BP, lipids, HbA1c, BMI, waist, smoking, sex

XGBoost settings are the same fixed values as before (eta 0.03, depth 3, subsample 0.8, colsample 0.8, min_child_weight 20) with early stopping. There is no tuning on held-out eras.

**Primary endpoint.** Harrell's C-index difference M3 − M2 in each held-out era, with 500 paired bootstrap resamples. The pooled estimate is the sample-size-weighted mean across eras.

**Hypotheses.**
- **H8:** M3 beats M2 by at least 0.005 in C-index, pooled across held-out eras, with the pooled 95% CI excluding 0.
- **H9:** M2 beats PhenoAge (M1) in every held-out era.

**Secondary.** M4 vs M3 (does lifestyle add beyond the full panel?); SHAP ranking of which new markers matter.

**Known caveat.** No NHANES 1999–2018 data remain untouched by earlier analyses. Leave-one-era-out guards against overfitting to a period, not against analyst knowledge.

## Analysis 2: Is pulse pressure a cause of faster biological aging? (Two-sample Mendelian randomization)

**Question.** The observational association between pulse pressure and PhenoAge acceleration (+0.57 years per 10 mmHg at fixed systolic BP) could be causal, confounded, or reversed. Genetic variants that raise pulse pressure are fixed at conception, so they are largely free of confounding by lifestyle and of reverse causation.

**Data (public summary statistics).**
- **Exposure:** pulse pressure, Evangelou et al. 2018, ~757k Europeans (UK Biobank + ICBP), GWAS Catalog GCST006629. Systolic BP from GCST006624.
- **Outcome:** PhenoAge acceleration, Kuo et al. 2021, UK Biobank, N = 107,460, figshare 12620291. This is the same blood-chemistry PhenoAge as in the main analysis.

**Instruments.** Variants with exposure p < 5×10⁻⁸, clumped to independence (r² < 0.001 within 10 Mb using 1000 Genomes Europeans if available; otherwise keep the strongest variant per 1 Mb window). Palindromic variants with allele frequency 0.42–0.58 are dropped. Each instrument must have F > 10.

**Estimators.** The primary estimator is inverse-variance weighted (IVW), multiplicative random effects. Sensitivity estimators: MR-Egger (intercept tests directional pleiotropy), weighted median, and Steiger filtering (drops variants that explain more variance in the outcome than the exposure).

**Hypothesis.**
- **H10:** Genetically higher pulse pressure causes higher PhenoAge acceleration. Supported if IVW p < 0.05 and weighted median and Egger estimates point the same way.

**Secondary.**
- Systolic BP → PhenoAge acceleration.
- Multivariable MR with pulse pressure and systolic BP together, which mirrors the observational "at fixed systolic BP" result.
- **Reverse MR:** PhenoAge acceleration → pulse pressure (the literature suggests aging may raise BP).

**Known caveat.** Both GWAS include UK Biobank participants. Sample overlap biases MR toward the observational estimate when instruments are weak; strong instruments (high F) limit this. Pleiotropy, where variants affect aging through paths other than pulse pressure, is the main threat; the Egger and median estimators address it partly.

---

## Outcome and deviations (added after running, 24 Nov 2026)

**Results.**
- **H8 supported:** full panel vs 9 markers, pooled ΔC +0.0060 (95% CI +0.0044 to +0.0076). Positive in all 5 held-out eras.
- **H9 supported** in the 4 evaluable eras: XGBoost on 9 markers beats PhenoAge, pooled ΔC +0.0074 (+0.0053 to +0.0094).
- **H10 not supported:** pulse pressure → PhenoAge acceleration, IVW −0.034 years per 10 mmHg (−0.22 to +0.16), with 340 LD-clumped instruments. The observational estimate (+0.57) lies outside the interval.

**Deviations from the plan.**
1. **Clumping order.** The 1000 Genomes reference download was incomplete at first, so the MR was first run with the 1 Mb distance fallback. Once the reference was complete, the pre-registered LD clumping was run and is reported as primary. The distance results (1, 5 and 10 Mb windows) are kept as sensitivity analyses. All point in the same direction.
2. **plink unavailable.** LD clumping was implemented in Python (reading the PLINK .bed directly), with variants pre-thinned to the strongest per 10 kb before clumping.
3. **MHC excluded** (chr6:25–34 Mb). This was not in the plan; it's standard practice because of long-range LD there.
4. **Outcome QC.** Kuo variants filtered to imputation INFO ≥ 0.8. Not in the plan.
5. **PhenoAge vs XGBoost (H9)** was compared on participants with CRP within each era (pairwise-complete), as CRP is partly missing even outside 2011–14.
6. **Two coding bugs** were fixed before any result was read: a missing validation set for the age-only model, and PhenoAge skipped when any CRP value was missing.

---

# Pre-registration: external validation in NHANES III (1988–94)

Written 24 Nov 2026, before the analysis was run. Same self-registration caveat as above.

**Question.** Do the models trained on NHANES 1999–2018 keep their advantage in a cohort they have never seen, measured a decade earlier with different analyzers and protocols? In particular, does the full lab panel's gain (H8) survive a change of laboratory?

**What is already known (so it is not a blind test).** An earlier backward validation scored XGBoost-C (trained on 1999–2006) on NHANES III. It beat PhenoAge there (C 0.864 vs 0.852). That result is known to me. The full-panel models have never been scored on NHANES III, and its extra lab markers have never been loaded.

**Data.** NHANES III adults 20–85 (the training data are top-coded at 85), linkage-eligible, with all 9 PhenoAge markers. Follow-up is censored at 10 years, matching Analysis 1.

**Marker mapping (fixed in advance).** 26 of the 30 extra markers have a same-analyte NHANES III variable, converted to the 1999–2018 units: RBC, haemoglobin, haematocrit, MCH, MCHC, platelets, MPV, ALT, AST, GGT, LDH, urea nitrogen, calcium (SI × 4.008), bicarbonate, iron, phosphorus, bilirubin, total protein, triglycerides, uric acid, sodium, potassium, chloride, osmolality, globulin and cholesterol. The 4 white-cell differential percentages (neutrophil, monocyte, eosinophil, basophil) were measured differently in NHANES III (a three-part Coulter differential, plus a manual count in a subsample), so they are excluded. **No marker is recalibrated**, apart from creatinine, which already had the CDC equation applied. The point of the test is to see what happens under real lab differences.

**Models.** All are trained once on all of NHANES 1999–2018 (RDW harmonized as before, 15% held out for early stopping), with the same fixed XGBoost settings as Analysis 1:
- M0: age-only Cox
- M1: published PhenoAge (**not out of sample:** NHANES III is its training cohort)
- M2: XGBoost, 9 markers + age
- M3x: XGBoost, 9 markers + 26 mapped extras + age
- M4x: M3x + BP, lipids, HbA1c, BMI, waist, smoking, sex

**Primary endpoint.** Harrell's C at 10 years in NHANES III, with 1,000 paired bootstrap resamples for the differences.

**Hypotheses.**
- **H11:** M3x beats M2 in NHANES III, with ΔC > 0 and the 95% CI excluding 0. This is the new test: whether the full-panel gain transfers across laboratories.
- **H12:** M2 beats PhenoAge in NHANES III, even though PhenoAge was fitted on it. Confirmatory only, since the earlier XGBoost-C result is known.

**Secondary.**
- Calibration: mean predicted vs Kaplan–Meier observed 10-year risk, overall and by decile.
- Within-era reference: M3x − M2 in the held-out 2007–10 era from the same fit procedure, so the size of any shrinkage can be judged.
- Marker shift: the median of each extra marker in NHANES III vs 1999–2018.

## Outcome and deviations (added after running, 24 Nov 2026)

**Results** (n = 15,007, 2,202 deaths within 10 years; `reports/nhanes3_external.json`).
- **H11 supported:** M3x − M2 = +0.0035 (95% CI +0.0013 to +0.0057). This is less than half the within-era reference gain of +0.0078 (2007–10 held out).
- **H12 supported:** M2 − PhenoAge = +0.0059 (+0.0036 to +0.0083).
- **Calibration:** M2 predicted 14.1% 10-year mortality vs 14.7% observed. M3x predicted 17.9% (over-prediction).

**Deviations.** None from the pre-registered analysis.

**Post-hoc analysis** (exploratory; `reports/nhanes3_external_posthoc.txt`). Dropping the most-shifted markers shows that ALT (median 14 vs 20 U/L) and bicarbonate account for most of the over-prediction. Dropping all eight shifted markers restores calibration (14.1%) while keeping C = 0.8675.

---

# Pre-registration: recalibrating a new laboratory without outcomes (NHANES III)

Written 24 Nov 2026, before the analysis was run.

**Question.** The 35-marker model over-predicted NHANES III mortality (17.9% vs 14.7%) because several assays read differently there. Can a lab fix this using **only its own marker values, without any outcome data**, by translating each marker onto the training data's scale?

**Method (primary).** For each of the 26 extra markers, quantile-map NHANES III values onto the 1999–2018 training distribution, within strata of sex × age band (20–39, 40–59, 60–85). A person at the 30th percentile of ALT among NHANES III women aged 40–59 receives the 30th percentile of ALT among training-data women aged 40–59. Stratifying by age and sex stops differences in age structure from being "corrected away". The nine PhenoAge markers are left unchanged. No death data from NHANES III are used in the mapping. The model is the same 35-marker model (M3x) as in H11, not retrained.

**Secondary methods.**
- (a) Unstratified quantile mapping.
- (b) Median-ratio scaling: multiply each marker by (training median ÷ NHANES III median). This is the simplest correction a lab could apply from reference-range information alone.
- (c) Primary method applied to all 35 markers, including the nine PhenoAge markers.

**Hypotheses.**
- **H13:** After primary mapping, M3x's mean predicted 10-year risk is within ±10% (relative) of the Kaplan–Meier observed risk (13.2%–16.2%), and its C-index is no more than 0.002 below unmapped M3x.
- **H14:** Mapped M3x still beats M2 in C-index, with the paired bootstrap 95% CI excluding 0 (1,000 resamples).

**Also reported.** Calibration by decile, and the integrated calibration index (mean absolute predicted − observed across deciles).

**Known caveat.** Quantile mapping removes all differences in marker distribution between populations, including real biological ones. For example, ALT may truly have been lower in 1988–94, when obesity and fatty liver were less common. If calibration improves, that shows the correction works for risk prediction. It does not show that the whole shift was a laboratory artefact.

## Outcome (added after running, 24 Nov 2026)

Results are in `reports/recalibration.json`. Observed 10-year mortality was 14.7%.

- **H13 supported.** After primary mapping, mean predicted mortality was 14.2% (unmapped: 17.9%). The C-index was 0.8680, against 0.8672 unmapped (ΔC +0.0008, 95% CI −0.0005 to +0.0020). Decile ICI fell from 0.034 to 0.012.
- **H14 supported.** Mapped M3x − M2 = +0.0043 (+0.0023 to +0.0062).
- **Secondary methods:**
  - Unstratified mapping: 14.4%.
  - Median-ratio scaling: 15.0%.
  - Mapping all 35 markers: 13.4% (slight over-correction).

**Deviations.** None.
