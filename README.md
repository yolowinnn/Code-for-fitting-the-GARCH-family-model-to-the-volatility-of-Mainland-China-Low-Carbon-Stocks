# Mixed-Frequency Investor Sentiment and Stock-Market Volatility: a GARCH-MIDAS Family Study

> 《基于 GARCH-MIDAS 族的混频投资者情绪对股市波动影响》
> Undergraduate research project, School of Mathematics, Southwest Jiaotong University — **first author**, 2022–2023.
> Li Jiawei, Wang Xiang, Jiang Tian-ai.
> Code, data, empirical output and the final report are all in this repository.

---

## What this asks

Realized volatility is measured at high frequency; investor sentiment arrives at a different, usually
lower frequency. Most studies resolve that mismatch by aggregating one series to match the other, which
discards information. **MIDAS (mixed-data sampling) lets the two frequencies stay where they are.**

The question here: *once you already have realized variance in the model, does investor sentiment still
carry incremental information about future volatility — and does it survive an honest out-of-sample test?*

Asset universe: a mainland-China low-carbon equity index (DT50). Sentiment is built from social-media
text at both instantaneous and low frequency (`NDDT_instant`, `nddt_lowf.csv`).

## The model ladder

The fitting scripts implement six specifications, each a strict extension of the previous one, so that any gain is attributable (the paper evaluates eight, adding two parameterised variants):

| Model | Adds | File |
|---|---|---|
| `GARCH` | baseline | `Code/Forecasting Model Code/.../GARCH.py` |
| `RGARCH` | realized measure in the variance equation | `RGARCH.py` |
| `GARCH-MIDAS` | long-run component at a second frequency | `GARCH-MIDAS.py` |
| `RGARCH-MIDAS` | realized measure **+** MIDAS long-run component | `RGARCH_MIDAS.py` |
| `GARCH-MIDAS + RV + X` | exogenous sentiment as a MIDAS regressor | `GARCH-MIDAS+RV+X.py` |
| **`RGARCH-MIDAS + RV + X`** | **full model — two MIDAS frequency bands + RV + sentiment** | `RGARCH-MIDAS+RV+X.py` |

The full specification carries two separate MIDAS weighting windows (`period1`, `period2`), which is the
"多混频 / multi-mixed-frequency" part: realized variance and sentiment are allowed to load on the long-run
component over different horizons rather than being forced to share one.

**The likelihoods are hand-written.** Each model implements its own log-likelihood and is fitted with
`scipy.optimize.minimize`, with standard errors from a numerical Hessian (`statsmodels.tools.numdiff.approx_hess`)
rather than calling a packaged GARCH routine. That was deliberate — the Realized-GARCH-MIDAS-X likelihood with
two Beta weighting schemes is not something an off-the-shelf package exposes.

## How it is evaluated

In-sample fit is not the claim. The out-of-sample suite (MATLAB, `Code/Out-of-Sample Testing Code/`):

- **MCS — Model Confidence Set** (`MCS_test.m`, `MCSPOSS0.m`): which models survive as statistically
  indistinguishable from the best, rather than just ranking them
- **Out-of-sample R²** (`Roos2222.m`, `Roos2cw.m`) against the historical-mean benchmark
- **Clark–West test** (`Perform_CW_test.m`) for nested-model forecast comparison
- **Pesaran–Timmermann directional test** (`directional_test_fordiff_PT.m`)

Using MCS rather than a single loss ranking matters here: with six nested models and a short sample, a
point estimate of "best" is not credible on its own.

## Findings

Eight specifications were compared out of sample (the repository ships six fitting scripts; two further
variants are parameterisations of them).

**Model Confidence Set**, 90% confidence, so a model survives at p > 0.1:

| | HMSE (T_R / T_SQ) | HMAE (T_R / T_SQ) |
|---|---|---|
| GARCH | 0.215 / 0.124 | 0.194 / 0.096 |
| RGARCH | 0.215 / 0.096 | 0.194 / 0.081 |
| GARCH-MIDAS | 0.215 / 0.112 | 0.194 / 0.093 |
| RGARCH-MIDAS | 0.205 / 0.088 | 0.194 / 0.076 |
| GARCH-MIDAS-X | 0.215 / 0.236 | 0.194 / 0.125 |
| **RGARCH-MIDAS-RV-X** | **1.000** | **1.000** |

Most models survive the MCS under HMSE/HMAE, which is the honest reading: on this sample the family as a
whole forecasts this index reasonably. What separates the full specification is that it sits at p = 1 under
both loss functions — it is the model the others are being measured against, not merely one that survived.
Under QLIKE, MSE and MAE the simpler models are rejected outright.

**Directional accuracy** (Pesaran–Timmermann), full out-of-sample window:

| Model | Success rate | PT statistic | p |
|---|---|---|---|
| RGARCH-MIDAS+X | 0.718 | 11.838 | 0.000 |
| GARCH-MIDAS+RV+X | 0.720 | 11.709 | 0.000 |
| **RGARCH-MIDAS+RV+X** | **0.750** | **13.165** | 0.000 |

The two ablations are the interesting comparison: dropping either the realized measure or the intraday
high-frequency channel costs roughly three percentage points of directional accuracy, and only the model
carrying both reaches 0.750.

**Robustness**: results hold across different forecast windows and different constructions of the realized
measure. Full parameter estimates, in-sample diagnostics and the robustness tables are in `Final Report/`.

## Reproduction status

All six fitting scripts execute end to end on the pinned versions in `requirements.txt`
(verified September 2026, Python 3.13). Data paths resolve relative to the repository, so a plain
`git clone` is enough.

Two caveats worth stating up front, both about the environment rather than the paper:

- **The baseline specifications reproduce cleanly.** `GARCH` and `GARCH-MIDAS` converge to interior
  solutions with all parameters significant — see the expected output below.
- **Two of the richer specifications land on parameter bounds under these versions.** In
  `GARCH-MIDAS+RV+X`, α and β settle exactly at 0.100 and 0.800, which are the edges of their feasible
  intervals (0, 0.1) and (0.8, 1), and θ₂ at −5.0; in `RGARCH_MIDAS`, α collapses to ~6.7e-05 with a
  standard error two orders of magnitude larger than the estimate. Those are boundary solutions, not
  converged interior optima.

The original estimation was done on Python 3.7 with the SciPy of that era, and the numbers reported in
the paper come from that environment. SLSQP's behaviour on tightly bounded, near-integrated likelihoods
is sensitive to the optimizer version, so this is most likely an environment difference rather than a
disagreement with the published results — but it has not been isolated, and it is stated here rather
than left for a reader to discover. Reproducing the paper's estimates for those two specifications
would mean pinning the original environment or re-tuning the starting values and bounds.

## Units and conventions

Returns are base-10 log returns of the index close, scaled by 100:

```
rt = 100 · log₁₀(Pₜ / Pₜ₋₁)
```

This affects how levels are read, not the dynamics. The GARCH persistence parameters (α, β, α+β), the MCS
comparison, the out-of-sample R² against the historical-mean benchmark and the Pesaran–Timmermann
directional test are all invariant under a linear rescaling of the return series; ω and the implied
variance level carry the factor.

For reference when reading the baseline output: an implied unconditional volatility of 0.734 in model
units is **1.69% daily, ≈26.8% annualised**, consistent with the realised volatility of the index over
this sample.

## Repository layout

```
Code/
  Forecasting Model Code/      six GARCH-family specifications (Python)
  Out-of-Sample Testing Code/  MCS, R²_OOS, Clark–West, PT directional (MATLAB)
Data/
  NDDT_instant/                high-frequency sentiment series
  nddt_lowf.csv                low-frequency sentiment
  s_index.csv, DT50低频.csv     index price / low-frequency series
  Prediction Results Data/     model forecasts
Empirical Evidence/            fitted plots — sentiment index, volatility, returns, price
Final Report/                  final paper (PDF + DOCX) and project close-out report
```

## Reproducing

```bash
pip install -r requirements.txt
python example.py
```

`example.py` fits the baseline GARCH(1,1) on the DT50 series and prints the parameter table with standard
errors, AIC/HQ, and in/out-of-sample forecast MSE — about thirty seconds, no configuration. Expected output:

```
DT50 low-carbon index — 2691 daily observations, 1883 in sample / 808 out of sample

         result    stderr        stat   p_value
mu     0.019482  0.011264    1.729545  0.041938
omega  0.001681  0.000719    2.337060  0.009770
alpha  0.052311  0.007405    7.063942  0.000000
beta   0.944567  0.007154  132.028078  0.000000

Persistence check: alpha + beta = 0.9969
```

The remaining specifications are in `Code/Forecasting Model Code/`; the MIDAS variants additionally take an
exogenous indicator series and two MIDAS window lengths. Out-of-sample tests (MCS, R²_OOS, Clark–West,
Pesaran–Timmermann) are MATLAB and live in `Code/Out-of-Sample Testing Code/`.

## Notes and limitations

- The sentiment series is specific to one Chinese social-media source over one sample window; the result
  should not be read as a general claim about sentiment and volatility.
- Sample length limits how much can be asked of a six-model comparison — which is why the MCS is reported
  rather than a single winner.
- The final paper is in Chinese.

---

*Author: Jiawei Li (李嘉伟) · [github.com/yolowinnn](https://github.com/yolowinnn)*
