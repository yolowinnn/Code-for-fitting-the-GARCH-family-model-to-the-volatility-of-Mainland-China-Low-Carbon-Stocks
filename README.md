# Mixed-Frequency Investor Sentiment and Stock-Market Volatility: a GARCH-MIDAS Family Study

> 《基于 GARCH-MIDAS 族的混频投资者情绪对股市波动影响》
> Undergraduate research project, Southwest Jiaotong University — **principal investigator**, 2022–2023.
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

Six specifications, each a strict extension of the previous one, so that any gain is attributable:

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

Full results — parameter estimates, MCS membership, out-of-sample R² and the directional tests — are in
`Final Report/`. The short version: adding a mixed-frequency sentiment regressor on top of a model that
already contains realized variance improved volatility forecasts, and the gain held up under the
out-of-sample suite rather than only in-sample.

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
