# Technical Overview

This document explains the implementation details: data contracts, feature formulas, alignment rules, model search configuration, evaluation, and reproducibility. It complements the high‑level README.

---

## 1) Data Sources & Contracts

### 1.1 Yahoo Finance (via `yfinance`)
- Files written by `scripts/download_data.py` → `src/data/download.py`:
  - `data/raw/nvda_prices.csv`
  - `data/raw/benchmarks/spy_prices.csv`
- Expected schema (per row = trading day):
  - `Date` (ISO date), `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`
- Ingestion rules:
  - Parse `Date` with UTC semantics; coerce numeric columns via `pd.to_numeric(errors="coerce")`.
  - Sort by `Date` and drop index.

### 1.2 Alpha Vantage NEWS_SENTIMENT
- `scripts/download_sentiment.py` fetches and aggregates per‑article scores into a daily table:
  - Output: `data/external/nvda_sentiment.csv`
  - Schema: `date` (ISO), `sentiment_score` (float)
- Aggregation: mean of `overall_sentiment_score` per calendar date; look‑back configurable (default 180 days).
- Alpha Vantage free tier generally returns ~100 most recent items → schedule regular pulls to maintain a longer history.

### 1.3 Optional: FRED (macro)
- Any macro series can be added in `configs/data_sources.yaml` and will hydrate `data/external/` as CSVs with `date,value`.

---

## 2) Feature Engineering (in `src/features/engineering.py`)

Let `P_t` be the closing price on day `t` and `V_t` the volume. We derive:

### 2.1 Returns & Momentum
- Simple return: `r_t = P_t / P_{t-1} - 1`
- Momentum (difference): `m_t(k) = P_t - P_{t-k}`
- Lagged values: `P_{t-1}, P_{t-2}, …` for selected columns (default: `Close`, optionally `Volume`).

### 2.2 Rolling Statistics
- For window `w ∈ {5,10,21}`: `mean_t(w)`, `std_t(w)`, `min_t(w)`, `max_t(w)` over `Close`.

### 2.3 Technical Indicators
- RSI (period `n`):
  - `ΔP_t = P_t - P_{t-1}`; `gain_t = max(ΔP_t, 0)`, `loss_t = max(-ΔP_t, 0)`
  - `RSI_t = 100 - 100/(1 + (avg_gain/avg_loss))`, with zero‑division guard.
- MACD: `EMA_fast(12) − EMA_slow(26)` and signal line `EMA_9(MACD)`.

### 2.4 Volatility & Bands
- Rolling std as volatility proxy: `vol_t(w) = std_t(w)`
- Trading range: `range_t(w) = max_t(w) − min_t(w)`
- Bollinger: `upper = MA_w + k·STD_w`, `lower = MA_w − k·STD_w`, bandwidth `(upper − lower) / MA_w`.

### 2.5 Volume Signals
- Percentage change: `ΔV_t(k) = V_t / V_{t-k} − 1`
- Z‑score (window `w`): `(V_t − mean(V,w)) / std(V,w)`

### 2.6 Calendar Encodings
- `weekday ∈ {0..6}`, `month ∈ {1..12}`
- Cyclic mapping: `sin/cos(2π·weekday/7)`, `sin/cos(2π·month/12)`

### 2.7 Benchmark Relations (SPY)
- Return spread: `r_t(NVDA) − r_t(SPY)`
- Price ratio: `P_t(NVDA) / P_t(SPY)`
- (Optional) rolling correlation of returns.

### 2.8 Sentiment Aggregates
- Daily `sentiment_score` is forward‑filled and back‑filled to cover non‑news days.
- Rolling windows over sentiment (mean/std/change) with configured windows.

### 2.9 Alignment Rules
- All sources keyed on `Date`; left‑join with price table as spine.
- Missing values: forward‑fill then backward‑fill for macro/sentiment/benchmarks.
- Final dataset sorted by `Date` and de‑na’ed just before modeling.

---

## 3) Modeling & Evaluation

### 3.1 Target
- Default: predict `P_{t+1}` (next‑day close). Alternatives supported but not used here: simple return or log‑return.

### 3.2 Split & Leakage Control
- Time‑aware split: 70% train, 15% validation, 15% test (chronological order preserved).
- Validation never spills into training. `Date` is excluded from the feature matrix.

### 3.3 Model Zoo & Search (in `configs/pipeline.yaml`)
- Candidates:
  - RandomForestRegressor (grid over `n_estimators`, `max_depth`)
  - GradientBoostingRegressor (grid over `n_estimators`, `learning_rate`, `max_depth`)
- Selection criterion: lowest validation RMSE; refit winner on `train+val`.
- Guard: if `n_samples < cv`, grid search is skipped and the base estimator is trained.

### 3.4 Metrics
- RMSE, MAE, MAPE, R², Directional Accuracy (sign agreement of `y_true` vs `y_pred`).
- Reported for validation and test sets in `reports/metrics/latest.json`.

---

## 4) Pipeline Diagram (Mermaid)

```mermaid
flowchart LR
  subgraph Sources
    YF[Yahoo Finance\n(NVDA OHLCV, SPY)]
    AV[Alpha Vantage\nNEWS_SENTIMENT]
    FRED[(FRED Macros)]
  end

  YF --> FE[Feature Engineering]
  AV --> FE
  FRED --> FE

  FE --> MS[Model Search\n(Random Forest, Gradient Boosting)]
  MS --> BM[Best Model]
  BM -->|Refit (train+val)| EVAL[Metrics\n(validation/test)]
  BM -->|Predict| PRED[Predictions\n(daily, full, weekly)]
  PRED --> CHARTS[Charts\n(HTML, SVG)]
  EVAL --> METRICS[reports/metrics/latest.json]
```

## 5) Weekly Roll‑Up & Reporting

- After training, predictions are produced for the full window (120 days) and resampled to weekly using `W‑MON` (label/closed = left), then truncated to the last `weekly_weeks` (default 14).
- Artefacts written to:
  - `reports/datasets/latest_predictions_full.csv` (daily, full window)
  - `reports/datasets/latest_predictions_weekly.csv` (weekly, 14 rows)
  - `reports/figures/latest_actual_vs_predicted.html/.svg` (daily recent)
  - `reports/figures/latest_actual_vs_predicted_weekly.html/.svg`

---

## 6) Reproducibility & Ops Notes

- Determinism: fixed `random_state` in models; chronological splits.
- Environment:
  - Python ≥ 3.10; key libs: pandas, plotly, scikit‑learn, statsmodels, yfinance, requests, pytest.
  - Secrets via `.env`: `ALPHA_VANTAGE_API_KEY` (required), `FRED_API_KEY` (optional).
- Network constraints:
  - Yahoo endpoints can be blocked on corporate networks (DNS/firewall). If you see `Could not resolve host`, switch network/VPN or whitelist domains.
  - Alpha Vantage free tier is rate‑limited; stagger requests or cache responses.

---

## 7) Known Behaviour & Limitations

- The RF baseline tracks gradual moves well but under‑reacts to abrupt rallies (e.g., late‑October spike). Consider richer features (options/volatility/skew), more adaptive models, or shorter/longer windows.
- Sentiment coverage depends on the provider’s retention; for long horizons, persist daily pulls.

---

## 8) Extending the System

- Add new models to `configs/pipeline.yaml` → `models` (e.g., XGBoost/LightGBM/LSTM) and widen the search space.
- Introduce additional data feeds under `data/external/` and wire them into the feature spec.
- Promote reports to a Dash app under `reports/dashboard/` and schedule the pipeline in CI/CD.

> Alpha Vantage free plan typically returns ~100 recent news items; schedule frequent downloads if you need a longer sentiment history.
