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
pip install numpy pandas matplotlib statsmodels scipy
python "Code/Forecasting Model Code/Code for Six GARCH Family Models/RGARCH-MIDAS+RV+X.py"
```

Out-of-sample tests require MATLAB.

## Notes and limitations

- The sentiment series is specific to one Chinese social-media source over one sample window; the result
  should not be read as a general claim about sentiment and volatility.
- Sample length limits how much can be asked of a six-model comparison — which is why the MCS is reported
  rather than a single winner.
- The final paper is in Chinese.

---

*Author: Jiawei Li (李嘉伟) · [github.com/yolowinnn](https://github.com/yolowinnn)*
