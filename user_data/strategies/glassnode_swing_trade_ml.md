I will download 1h data from glassnode as csv file and merge them into one

Here is a comprehensive PRD (Product Requirements Document) in Markdown format, along with a starter example config.json, specifically for your FreqAI + Glassnode on-chain/price/volume 1h ML swing trading pipeline.

# Crypto ML Swing Trading Strategy PRD

## Overview

This project aims to develop a machine learning-based swing trading strategy for cryptoassets using 1h candles and merged on-chain, price, and volume features from Glassnode and exchange data. The strategy will utilize FreqAI for feature engineering, training, and backtesting.

## Goals

- Predict forward 5-day returns (or price movement) for selected cryptoassets.
- Combine classic technical indicators, raw price/volume, and engineered on-chain features.
- Build a scalable, repeatable pipeline for feature engineering and model tuning.


## 1. Data

### Input

- 1h OHLCV historical price data for asset(s) (Downloaded via freqtrade)
- 1h on-chain metrics from Glassnode (raw CSVs, pre-merged into main dataframe on timestamp) (provided as csv)


### Output

- Unified dataframe for model training (no missing timestamps, imputed or dropped where necessary)


## 2. Features

| Type | Raw Features | Engineered/Composite Features |
| :-- | :-- | :-- |
| Price/Volume | close, open, high, low, volume | SMA_10, SMA_50, SMA_200, EMA_10, EMA_50,<br> RSI_14, MACD, Volume_SMA_20,<br> price_change_1d/5d/10d, volatility_7d |
| On-chain | mvrv, sopr, dormancy, accumulation_trend_score, exchange_netflow, <br> long_term_holder_supply, short_term_holder_supply, coin_days_destroyed, nvt, realized_profit_loss | mvrv_slope_3d, sopr_rolling_min_7d, dormancy_zscore, accumulation_trend_roc_7d, exchange_netflow_ema_7d, percent_lth, percent_sth, cdd_rolling_sum_7d |

- **Future target:** (for regression/classification)
e.g. `future_return_5d` or `will_rise_5d`


## 3. Pipeline Stages

### Data Import \& Merge

- Import CSVs, standardize/align on timestamp ("date" col assumed)
- Merge all features into single DataFrame


### Feature Engineering

- Use `feature_engineering_expand_all`, `feature_engineering_expand_basic`, and `feature_engineering_standard` functions.
- Engineer features as above, using pandas, TA-Lib, or direct Glassnode csv math.
- Prep features with `%` prefix for FreqAI base/expanded features.


### Label Creation

- Use `set_freqai_targets`.
Example:

```
dataframe["&-target_5d"] = dataframe["close"].shift(-5*24) / dataframe["close"] - 1
```


### FreqAI Preprocessing

- Normalization/standardization as handled by FreqAI pipeline.
- Outlier handling (activate SVR/DBSCAN/DI for removal as needed).
- PCA if dimensionality is excessive.


## 4. ML Model Setup

### Model Selection

- Begin with `XGBoostRegressor`
- Alternatives: `LightGBMRegressor`, `PyTorchMLPRegressor` for sanity check


### Config Example (`freqai`):

```json
"freqai": {
  "enabled": true,
  "identifier": "glassnode_1h_swing",
  "train_period_days": 365,
  "backtest_period_days": 30,
  "activate_tensorboard": true,
  "write_metrics_to_disk": true,
  "feature_parameters": {
    "include_timeframes": ["1h", "4h", "1d"],
    "include_corr_pairlist": [],
    "label_period_candles": 120,    // 5 days for 1h data
    "include_shifted_candles": 2,
    "indicator_periods_candles": [10, 20, 50],
    "principal_component_analysis": false,
    "plot_feature_importances": 20,
    "DI_threshold": 1,
    "use_SVM_to_remove_outliers": true,
    "use_DBSCAN_to_remove_outliers": false,
    "noise_standard_deviation": 0.05,
    "weight_factor": 0.98
  },
  "data_split_parameters": {
    "test_size": 0.2,
    "shuffle": false
  },
  "model_training_parameters": {
    "model_class": "XGBoostRegressor",
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 20,
    "eval_metric": "rmse"
  }
}
```


## 5. Coding Hints

- Use TA-Lib or pandas-ta for price/volume indicators.
- For Glassnode, compute moving averages, deltas, and z-scores as new columns.
- In each `feature_engineering_*` function, be sure to use the correct `%` prefix.
- Avoid future leakage in target calculation.


## 6. Example `feature_engineering_expand_basic` snippet

```python
def feature_engineering_expand_basic(self, dataframe, metadata, **kwargs):
    dataframe["%-close"] = dataframe["close"]
    dataframe["%-volume"] = dataframe["volume"]
    dataframe["%-sma_10"] = ta.SMA(dataframe, timeperiod=10)
    dataframe["%-rsi_14"] = ta.RSI(dataframe, timeperiod=14)
    dataframe["%-mvrv"] = dataframe["mvrv"]
    dataframe["%-sopr"] = dataframe["sopr"]
    dataframe["%-dormancy"] = dataframe["dormancy"]
    # ... add more engineered features as columns ...
    return dataframe
```


## 7. Strategy Testing

- Backtest using FreqAI built-in sliding window
- Evaluate model loss, feature importance, and equity curve


## 8. Success Criteria

- Reasonable directional accuracy on test set (target: will_rise_5d)
- Acceptable drawdown/profit from backtest
- Feature importance highlighting mix of technical/on-chain features


## 9. Deliverables

- Fully commented FreqAI strategy Python file with feature engineering as above
- Config JSON for freqai block as above
- Jupyter notebook/code for raw CSV ingestion/merge if needed

**Appendices:**

- Refer to [FreqAI feature engineering docs](https://www.freqtrade.io/en/stable/freqai-feature-engineering/) and [FreqAI parameter table](https://www.freqtrade.io/en/stable/freqai-parameter-table/) for advanced use.

