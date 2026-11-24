# Multi-marker biological-age models lose calibration across laboratories, and a simple outcome-free correction restores it: a pre-registered NHANES study of blood markers, blood pressure and a Mendelian randomization test of pulse pressure

**Alexander Thomsen**
Independent researcher

*Draft preprint, 24 November 2026. Not peer reviewed.*

---

## Abstract

**Background.** Biological-age measures built from routine blood tests, such as PhenoAge, predict mortality beyond chronological age. It is unclear how much blood pressure (BP) adds to these measures, whether broader laboratory panels and flexible models improve on them, whether such models remain valid when laboratory methods change, and whether the observed link between pulse pressure and accelerated biological aging is causal.

**Methods.** I used US National Health and Nutrition Examination Survey (NHANES) participants aged ≥20 years: 54,945 from 1999–2018 and 16,520 from NHANES III (1988–94), linked to death records through 2019. Gradient-boosted Cox models (XGBoost) and five comparison models were trained on 1999–2006, tuned on 2007–08 and evaluated once on 2009–10. Four follow-up analyses were pre-registered before they were run:
1. A 39-marker laboratory panel, evaluated by leave-one-era-out validation across five survey eras.
2. External validation of the panel in NHANES III, a cohort measured with different laboratory methods and never used for training.
3. An outcome-free recalibration, in which each marker is translated onto the training distribution within age and sex strata.
4. Two-sample Mendelian randomization (MR) of pulse pressure on PhenoAge acceleration, using public genome-wide summary statistics (exposure: about 750,000 Europeans; outcome: 107,460 UK Biobank participants).

**Results.**
- **Prediction.** On the test set, XGBoost on blood markers, BP and lifestyle factors reached a C-index of 0.873, compared with 0.858 for published PhenoAge (difference 0.015, 95% CI 0.010–0.020) and 0.838 for age alone. Adding BP to blood markers changed the C-index by only 0.0017.
- **Full panel.** The 39-marker panel improved discrimination over the nine PhenoAge markers in all five held-out eras (pooled ΔC 0.0060, 0.0044–0.0076).
- **External validation.** In NHANES III the full-panel gain halved to 0.0035 (0.0013–0.0057), and the model over-predicted 10-year mortality (17.9% predicted vs 14.7% observed). The cause was between-laboratory shifts in individual assays, chiefly alanine aminotransferase (ALT; median 14 vs 20 U/L) and bicarbonate. The nine-marker model stayed well calibrated (14.1%).
- **Recalibration.** Translating the extra markers onto the training scale, without retraining and without any outcome data, restored calibration (14.2% predicted; decile calibration error 0.034 → 0.012). Discrimination was unchanged, and the full-panel gain was kept (0.0043, 0.0023–0.0062). Scaling each marker by a single ratio of medians worked almost as well (15.0%).
- **Pulse pressure.** Observationally, each 10 mmHg of pulse pressure was associated with 0.57 years (0.48–0.66) of PhenoAge acceleration at fixed systolic BP. Genetically, the estimate was −0.03 years (−0.22 to 0.16), using 340 independent instruments with mean F = 77 and no evidence of directional pleiotropy.

**Conclusions.**
- Routine blood markers carry most of the mortality information that BP would add.
- A broader panel adds a small, consistent gain in ranking, but its absolute risks depend on how each laboratory measures each marker. A cheap, outcome-free translation of each marker onto the training scale fixes this, and it should be a standard step before such models report risks or biological ages to individuals.
- The pulse pressure–biological age association is unlikely to be causal.

**Keywords:** biological age; PhenoAge; survival analysis; gradient boosting; external validation; laboratory harmonization; blood pressure; pulse pressure; Mendelian randomization; NHANES

---

## 1. Introduction

Chronological age is the strongest single predictor of death, but people of the same age differ widely in health. Second-generation biological-age measures are trained on mortality rather than on age. They express risk as the age at which an average person carries the same risk. PhenoAge [1] combines nine routine blood markers with age in a Gompertz model: albumin, creatinine, glucose, C-reactive protein (CRP), lymphocyte percentage, mean cell volume, red cell distribution width (RDW), alkaline phosphatase and white cell count. It is widely used because it needs only a standard blood test.

This study asks four questions:

1. **Do flexible survival models and broader laboratory panels improve on PhenoAge?** Gradient boosting and neural survival models can capture non-linear relations and interactions. A routine chemistry panel and blood count measure roughly 40 analytes, not nine.
2. **Does BP add information?** BP is the leading modifiable risk factor for stroke [2, 3] and is measured at almost every clinical encounter, yet it is not part of PhenoAge.
3. **Do such models transfer across laboratories?** Biological-age models are increasingly applied to results from laboratories other than the one that generated the training data. Assay methods differ between laboratories and over time, and the effect of this on multi-marker models is rarely tested.
4. **Is the link between pulse pressure and biological age causal?** Pulse pressure (systolic minus diastolic) reflects large-artery stiffness. In my data it was associated with accelerated PhenoAge independently of systolic BP. That association could be causal, confounded or reversed. Mendelian randomization can help separate these.

## 2. Methods

### 2.1 Data and participants

NHANES is a continuous, nationally representative cross-sectional survey of the US population that includes a physical examination and laboratory tests. I used all 2-year cycles from 1999–2000 to 2017–18 and NHANES III (1988–94), linked to the 2019 public-use Linked Mortality Files (follow-up to 31 December 2019). I included adults aged ≥20 years who were eligible for linkage.

**Table 1. Cohorts.**

| Cohort | Adults | Deaths | Median follow-up | Role |
|---|---|---|---|---|
| NHANES III, 1988–94 | 16,520 | 7,082 | 26.2 y | External validation |
| 1999–2006 | 20,290 | 5,699 | 14–20 y | Training |
| 2007–08 | 5,926 | 1,123 | 11.8 y | Tuning |
| 2009–10 | 6,201 | 859 | 9.8 y | Test (evaluated once) |
| 2011–14 | 11,303 | 1,093 | 6–8 y | Leave-one-era-out only (no CRP) |
| 2015–18 | 11,225 | 419 | 2–4 y | Temporal validation |

### 2.2 Measurements and harmonization

- **BP:** the mean of up to four readings at the examination.
- **Creatinine:** recalibrated with the CDC equations for 1988–94, 1999–2000 and 2005–06.
- **RDW:** a haematology analyzer change in 2013 shifted the RDW distribution, so RDW from 2013 onward was quantile-mapped onto the 2011–12 distribution.
- **Other markers:** not harmonized.
- **PhenoAge:** computed with the published coefficients, as implemented in the BioAge reference package [4]. The implementation was verified against that reference in automated tests.

### 2.3 Prediction models

All models predict time to death. I compared:
- the published PhenoAge formula, without refitting
- Cox proportional hazards
- elastic-net Cox
- random survival forest
- gradient-boosted trees with a Cox objective (XGBoost)
- a neural Cox model (DeepSurv) [5]

Each model was fitted on three nested feature sets:
- **A:** the nine PhenoAge markers plus age
- **B:** A plus systolic BP, pulse pressure and BP medication
- **C:** B plus total and HDL cholesterol, HbA1c, BMI, waist circumference, smoking and sex

Preprocessing was fitted on training data only. Hyperparameters were chosen on 2007–08, and the 2009–10 test set was used only for final evaluation.

**Biological age.** Each model's predicted 10-year risk was converted to the age at which an age-only Cox model gives the same risk.

**Metrics.**
- Discrimination: Harrell's C-index.
- Model differences: paired bootstrap confidence intervals.
- Secondary metrics: time-dependent AUC, integrated Brier score, and calibration by decile against Kaplan–Meier estimates.

### 2.4 Pre-registered follow-up analyses

Three analyses were specified in a dated, self-registered protocol (`PREREGISTRATION.md` in the code repository) before they were run. Deviations are reported in that document.

**Full laboratory panel (H8, H9).**
- *Markers:* 30 markers added to the nine PhenoAge markers. The blood count contributed the leukocyte differential, red-cell indices, platelets and mean platelet volume. The chemistry panel contributed liver enzymes, urea nitrogen, electrolytes, calcium, phosphorus, iron, bilirubin, total protein, globulin, triglycerides, uric acid, osmolality and cholesterol.
- *Design:* leave-one-era-out across five eras (1999–2002, 2003–06, 2007–10, 2011–14, 2015–18). Follow-up was censored at 10 years.
- *XGBoost settings (fixed in advance):* learning rate 0.03, depth 3, subsample and column sample 0.8, minimum child weight 20. Early stopping used a random 15% of training rows.
- *H8:* the full panel improves the pooled C-index by ≥0.005, with the 95% CI excluding 0.
- *H9:* XGBoost on the nine markers beats PhenoAge in every held-out era.

**External validation in NHANES III (H11, H12).**
- *Training:* models trained once on all of 1999–2018.
- *Test:* scored unchanged on NHANES III adults aged 20–85, with 10-year follow-up.
- *Markers:* 26 of the 30 extra markers had a same-analyte NHANES III variable and were converted to common units. The four white-cell differential percentages were excluded because NHANES III used a three-part differential.
- *No recalibration:* no marker was recalibrated, so the test reflects real between-laboratory differences.
- *H11:* the full-panel gain over nine markers persists, with ΔC > 0 and the CI excluding 0.
- *H12:* XGBoost on nine markers beats PhenoAge.
- *Caveats:* NHANES III is PhenoAge's training cohort, which favours PhenoAge in H12. An earlier backward validation of a feature-set-C model on NHANES III was known to me, so H12 is confirmatory. H11 involved models and markers never before applied to NHANES III.

**Recalibration without outcomes (H13, H14).**
- *Method:* each of the 26 extra markers in NHANES III was quantile-mapped onto the 1999–2018 training distribution within strata of sex × age band (20–39, 40–59, 60–85). The nine PhenoAge markers were left unchanged. The model was not retrained, and no NHANES III outcomes were used.
- *Secondary variants:* unstratified mapping; scaling each marker by the ratio of training to local medians; mapping all 35 markers.
- *H13:* after mapping, mean predicted 10-year risk lies within ±10% of the Kaplan–Meier observed risk, and the C-index falls by no more than 0.002.
- *H14:* the mapped 35-marker model still beats the 9-marker model (paired bootstrap CI excluding 0).
- *Calibration summary:* the integrated calibration index (ICI), the mean absolute gap between predicted and observed risk across deciles.

**Mendelian randomization (H10).**
- *Exposure:* pulse pressure and systolic BP from Evangelou et al. [6] (about 750,000 Europeans; GWAS Catalog GCST006629, GCST006624).
- *Outcome:* PhenoAge acceleration from Kuo et al. [7] (UK Biobank, N = 107,460; figshare 12620291).
- *Instruments:* variants with p < 5×10⁻⁸, clumped to r² < 0.001 within 10 Mb using 1000 Genomes Europeans, with the HLA region excluded. Ambiguous palindromic variants were removed, and outcome variants were filtered to imputation INFO ≥ 0.8.
- *Primary estimator:* inverse-variance weighted (IVW) with multiplicative random effects.
- *Sensitivity estimators:* MR-Egger [8], weighted median and Steiger filtering.
- *Secondary analyses:* systolic BP as the exposure; multivariable MR of pulse pressure conditional on systolic BP, which mirrors the observational "at fixed systolic BP" estimate; and reverse MR.
- *H10:* supported if IVW p < 0.05 with concordant sensitivity estimates.

### 2.5 Association analyses

PhenoAge acceleration was regressed on systolic BP and pulse pressure. Models used natural splines for age, adjustment for sex, race/ethnicity, smoking, BMI, diabetes, cholesterol, medication and prior cardiovascular disease, and robust standard errors. Stroke mortality was analysed with cause-specific Cox and Fine–Gray models in 1999–2014, alone and pooled with NHANES III.

### 2.6 Software

All code is in Python, with 34 automated tests. The tests:
- pin the PhenoAge constants to the BioAge reference
- verify the survival estimators against lifelines
- check that the MR estimators recover known effects in simulated data
- verify the PLINK genotype reader against hand-packed files

LD clumping was implemented directly on the PLINK binary files.

## 3. Results

### 3.1 Prediction and the contribution of BP

On the 2009–10 test set, XGBoost on feature set C reached a C-index of 0.873 (0.863–0.885). This was higher than published PhenoAge (0.858; Δ 0.015, 0.010–0.020), Cox regression on the same inputs (0.865; Δ 0.008) and age alone (0.838). DeepSurv performed the same as XGBoost (0.872). XGBoost also beat PhenoAge in the 2015–18 cohort (0.879 vs 0.860). Adding BP to the blood markers changed the C-index by 0.0017. Blood markers alone beat BP alone by 0.021.

### 3.2 Full laboratory panel (H8, H9 supported)

Among 48,681 participants (5,370 deaths within 10 years), the 39-marker panel beat the nine markers in every held-out era. Gains ranged from 0.004 (1999–2002) to 0.008 (2015–18), with a pooled gain of 0.0060 (0.0044–0.0076). XGBoost on the nine markers beat PhenoAge in all four eras where CRP was measured (pooled 0.0074, 0.0053–0.0094). Adding BP and lifestyle factors gave a further 0.0070. The most informative new markers were ALT, GGT, chloride, urea nitrogen, platelets and potassium.

### 3.3 External validation in NHANES III (H11, H12 supported; calibration failed)

**Table 2. Models trained on NHANES 1999–2018, tested on NHANES III (n = 15,007; 2,202 deaths within 10 years).**

| Model | C-index | Mean predicted 10-y risk | Observed |
|---|---|---|---|
| Age only | 0.838 | — | 14.7% |
| PhenoAge (fitted on this cohort) | 0.858 | — | 14.7% |
| XGBoost, 9 markers | 0.864 | 14.1% | 14.7% |
| XGBoost, 35 markers | 0.867 | 17.9% | 14.7% |
| XGBoost, 35 markers + clinical | 0.873 | 17.8% | 14.7% |

All models kept their development rank order.
- The 35-marker model beat the 9-marker model by 0.0035 (0.0013–0.0057). The same procedure gave 0.0078 when 2007–10 was held out.
- XGBoost on nine markers beat PhenoAge on PhenoAge's own training cohort (0.0059, 0.0036–0.0083).

Discrimination transferred, but calibration did not. The 35-marker model over-predicted mortality by about a fifth. Several extra markers had shifted between surveys:
- ALT 30% lower (median 14 vs 20 U/L)
- lactate dehydrogenase 20% higher
- bicarbonate 12% higher
- globulin and sodium higher

In a post-hoc analysis (not pre-registered):
- Dropping ALT alone reduced the mean prediction to 16.1%.
- Dropping bicarbonate alone reduced it to 16.6%.
- Dropping all eight shifted markers restored calibration (14.1%) with no loss of discrimination (C 0.8675).

### 3.4 Recalibration without outcomes (H13, H14 supported)

**Table 3. 35-marker model in NHANES III after translating markers onto the training scale (observed 14.7%).**

| Correction | C-index | Mean predicted | ICI |
|---|---|---|---|
| None | 0.867 | 17.9% | 0.034 |
| **Quantile map within age × sex (primary)** | 0.868 | 14.2% | 0.012 |
| Quantile map, unstratified | 0.868 | 14.4% | 0.010 |
| Median-ratio scaling | 0.868 | 15.0% | 0.012 |
| Quantile map, all 35 markers | 0.868 | 13.4% | 0.013 |
| *9-marker model, no correction* | 0.864 | 14.1% | 0.010 |

The primary correction brought predicted mortality to within 4% of observed and cut the decile calibration error by two-thirds, to the level of the 9-marker model. Discrimination was unchanged (ΔC +0.0008, −0.0005 to 0.0020), and the 35-marker model kept its advantage over 9 markers (+0.0043, 0.0023–0.0062). Mapping the nine PhenoAge markers as well slightly over-corrected, consistent with those markers already being comparable across surveys.

### 3.5 BP, biological age and stroke

**Systolic BP and PhenoAge acceleration.** The association was U-shaped, with its minimum at 120–130 mmHg. Above 120 mmHg, each 10 mmHg was associated with 0.31 years (0.21–0.41) of acceleration, and 0.38 years in NHANES III.

**Systolic BP and stroke mortality.** Each 10 mmHg was associated with stroke mortality: cause-specific HR 1.11 (1.08–1.15) and Fine–Gray sub-distribution HR 1.09 (1.05–1.13), from 815 stroke deaths in the pooled data. The association was strongest at age 20–59 (HR 1.38, 1.27–1.50) and attenuated at ≥75 years (HR 1.04).

**Pulse pressure.** At fixed systolic BP, pulse pressure was not associated with stroke (HR 0.96, 0.90–1.02). It was associated with PhenoAge acceleration (0.57 years per 10 mmHg, 0.48–0.66), and this replicated in NHANES III (0.56).

### 3.6 Mendelian randomization (H10 not supported)

**Table 4. Two-sample MR (years of PhenoAge acceleration per 10 mmHg; 95% CI).**

| Analysis | Variants | IVW | Weighted median | MR-Egger | Steiger-filtered |
|---|---|---|---|---|---|
| Pulse pressure | 340 | −0.03 (−0.22, 0.16) | 0.08 (−0.11, 0.28) | 0.07 (−0.41, 0.54) | 0.06 (−0.08, 0.21) |
| Systolic BP | 393 | 0.10 (−0.02, 0.22) | 0.11 (−0.02, 0.24) | 0.02 (−0.28, 0.33) | 0.10 (0.00, 0.20) |
| PP given SBP (multivariable) | 447 | −0.28 (−0.72, 0.15) | — | — | — |
| Reverse: PhenoAgeAccel → PP (mmHg/yr) | 50 | 0.04 (−0.06, 0.15) | 0.02 (−0.05, 0.09) | −0.19 (−0.49, 0.11) | 0.04 (−0.06, 0.15) |
| *Observational, for comparison* | — | *+0.57 (0.48, 0.66)* | | | |

The instruments were strong (mean F = 77). The MR-Egger intercept gave no evidence of directional pleiotropy (p = 0.66). Distance-based clumping (1, 5 and 10 Mb windows) gave the same conclusion. The IVW interval excludes the observational estimate.

## 4. Discussion

### 4.1 Principal findings

Flexible survival models improved on PhenoAge in temporal, era-wise and external validation, including on PhenoAge's own training cohort. They beat Cox regression on identical inputs, and gradient boosting matched a neural network. Together this suggests the gain comes from non-linear relations and interactions, and that the ceiling is set by the inputs rather than the algorithm.

BP added almost nothing to mortality prediction once blood markers were known. Markers of kidney function, glucose metabolism and inflammation probably already capture much of the downstream damage that BP causes. For stroke, systolic BP remained the informative measure. The age-specific effect sizes match the Prospective Studies Collaboration [2].

### 4.2 Laboratory transportability

The NHANES III test is the most practically relevant result. The broader panel's advantage in ranking people by risk survived a change of laboratory, at about half its size. Its absolute risks did not. Markers whose assays differ between laboratories, chiefly ALT and bicarbonate, shifted everyone's predicted risk.

The mechanism is simple. Low ALT is a frailty signal in the training data, so a laboratory that reads ALT systematically lower makes the whole population look frailer. The nine PhenoAge markers transferred cleanly, probably because they are well standardised and were already harmonised where needed.

Biological-age calculators are increasingly offered direct to consumers and applied to results from arbitrary laboratories. This finding therefore has a direct implication: **a multi-marker model is only as portable as its least standardised assay.** A model like this should not report absolute risks or biological ages from a new laboratory without a recalibration step.

The pre-registered recalibration test shows that this step is cheap. Translating each marker onto the training distribution, using only the laboratory's own marker values, restored calibration without retraining and without outcome data. A single median per marker was almost as good, so even a laboratory that publishes only reference intervals could apply it. I suggest that multi-marker biological-age models be distributed with their training-set marker distributions, so that users can perform this translation, and that tools applying them report which markers were translated.

### 4.3 Pulse pressure is a marker, not a driver, of biological aging

The observational association between pulse pressure and PhenoAge acceleration was robust and replicated across cohorts 10–30 years apart. The genetic estimate, however, was close to zero and excluded the observational value. The most plausible explanation is that wide pulse pressure and accelerated PhenoAge share causes, such as kidney disease, diabetes and inflammation, or are parallel expressions of the same underlying aging. Earlier MR work suggesting that biological aging may raise BP [9], and that BP and arterial stiffness affect each other in both directions [10], is consistent with this. My reverse MR was inconclusive.

### 4.4 Strengths and limitations

**Strengths:**
- pre-registration of the follow-up analyses, with deviations reported
- a single-use test set
- an external cohort measured with different laboratory methods
- triangulation of observational and genetic evidence
- fully public data and code, with automated tests

**Limitations:**
- **Self-registration.** The pre-registration was self-hosted with a timestamp, not lodged with a registry.
- **No untouched data for the full panel.** No NHANES 1999–2018 data remained unseen for the full-panel analysis.
- **NHANES III is still American.** It is a US cohort, and PhenoAge's home advantage there makes H12 conservative.
- **Exploratory post-hoc analysis.** The marker-dropping analysis was post hoc.
- **What mapping removes.** Quantile mapping removes all distributional differences between populations, including real ones. ALT may truly have been lower in 1988–94, when fatty liver was less common. The success of mapping shows that it works for prediction, not that the whole shift was a laboratory artefact. Mapping should use a laboratory's general reference population, not a selected patient group, whose marker distribution would be shifted by disease.
- **Sample overlap in the MR.** Both GWAS include UK Biobank participants. Sample overlap biases MR toward the observational estimate, so the null is conservative.
- **MR heterogeneity and precision.** Heterogeneity was high (Cochran's Q p < 10⁻⁴⁰), although the robust estimators agreed. The outcome GWAS limits precision: effects below about 0.2 years per 10 mmHg cannot be excluded.
- **Single-visit BP.** BP was measured at one visit, so regression dilution attenuates its associations.
- **Cross-sectional association analyses.** Biological age and BP were measured at the same time.
- **Mortality is the only outcome.**
- **No European validation.** No European cohort with open individual-level laboratory and mortality data exists. UK Biobank, the natural candidate, covers about 30 of the 39 markers and requires an approved application.

### 4.5 Conclusions

- Routine blood markers, modelled flexibly, predict mortality better than PhenoAge.
- BP adds little to mortality prediction but remains central for stroke.
- Broader laboratory panels help modestly. They lose calibration across laboratories, but a simple outcome-free translation of each marker onto the training scale restores it.
- Pulse pressure tracks biological aging but probably does not cause it.

## Data and code availability

All data are public:
- NHANES and the Linked Mortality Files (CDC/NCHS)
- GWAS Catalog GCST006629 and GCST006624
- figshare 12620291
- the 1000 Genomes Phase 3 reference panel

The full pipeline, pre-registration and tests are available at https://github.com/AlexBThomsen/bioage-nhanes.

## Use of AI tools

The analysis code, figures and a first draft of this manuscript were produced with the help of an AI coding assistant (Claude, Anthropic) working under my direction. I defined the research questions and the follow-up analyses and reviewed the results. Automated tests and the pre-registration were used to guard against errors in the generated code. Responsibility for the content rests with me.

## Competing interests

None declared.

## Funding

None.

## References

1. Levine ME, et al. An epigenetic biomarker of aging for lifespan and healthspan. *Aging* 2018;10:573–591.
2. Lewington S, et al. Age-specific relevance of usual blood pressure to vascular mortality. *Lancet* 2002;360:1903–1913.
3. Ettehad D, et al. Blood pressure lowering for prevention of cardiovascular disease and death. *Lancet* 2016;387:957–967.
4. Kwon D, Belsky DW. A toolkit for quantifying aging and testing healthspan extension in NHANES. *GeroScience* 2021.
5. Katzman JL, et al. DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. *BMC Med Res Methodol* 2018;18:24.
6. Evangelou E, et al. Genetic analysis of over 1 million people identifies 535 new loci associated with blood pressure traits. *Nat Genet* 2018;50:1412–1425.
7. Kuo CL, et al. Genetic associations for two biological age measures point to distinct aging phenotypes. *Aging Cell* 2021.
8. Bowden J, Davey Smith G, Burgess S. Mendelian randomization with invalid instruments: effect estimation and bias detection through Egger regression. *Int J Epidemiol* 2015;44:512–525.
9. Phenotypic age acceleration and hypertension: a Mendelian randomization study. *J Nutr Health Aging* 2024. *[full citation to be completed]*
10. Bidirectional causal relations between blood pressure and arterial stiffness. PMC7544538. *[full citation to be completed]*
