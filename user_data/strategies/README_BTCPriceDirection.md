# BTC Price Direction Strategy

A simple, focused machine learning strategy for predicting BTC price direction using only price-based technical indicators.

## Overview

This strategy addresses the key insights from your previous Glassnode-based ML work:

1. **Direction prediction** is more realistic than regression for volatile crypto markets
2. **Price-based features first** provide a solid foundation before adding non-price data
3. **Small feature set** reduces noise and overfitting
4. **Simple implementation** focuses on proven technical indicators

## Key Features

- **Classification Model**: Predicts up/down direction (not magnitude)
- **3-Day Horizon**: Predicts direction for the next 3 days
- **Daily Timeframe**: Reduces noise for swing trading
- **Minimal Features**: Only price-based technical indicators
- **0.5% Threshold**: Avoids noise from small price movements

## Technical Indicators Used

### Core Indicators
1. **RSI (14)**: Momentum indicator
2. **MACD (12,26,9)**: Trend indicator with signal line and histogram
3. **Bollinger Bands (20,2)**: Volatility indicator
4. **Moving Averages**: SMA(20,50) and EMA(12,26)

### Derived Features
- Price ratios to moving averages
- ATR (Average True Range) for volatility
- Price momentum (1d, 3d, 7d changes)
- High-Low range indicators
- Volume indicators (if available)

## Target Variable

- **Binary Classification**: 1 = Price goes up >0.5% in next 3 days, 0 = Down/flat
- **Threshold**: 0.5% to avoid noise from small movements
- **Horizon**: 3 days forward prediction

## Files

- `BTCPriceDirectionStrategy.py`: Main strategy implementation
- `config_btc_price_direction.json`: FreqAI configuration
- `btc_price_direction_analysis.py`: Data analysis script
- `btc_price_direction_backtest.py`: Backtest runner

## Usage

### 1. Data Analysis
```bash
python user_data/report/btc_price_direction_analysis.py
```

### 2. Backtest
```bash
python user_data/report/btc_price_direction_backtest.py
```

### 3. Manual Backtest
```bash
freqtrade backtesting \
  --config user_data/strategies/config_btc_price_direction.json \
  --strategy BTCPriceDirectionStrategy \
  --timerange 20200101-20241231 \
  --timeframe 1d \
  --freqai-backtest-live-models
```

### 4. Hyperopt
```bash
freqtrade hyperopt \
  --config user_data/strategies/config_btc_price_direction.json \
  --strategy BTCPriceDirectionStrategy \
  --timerange 20200101-20231231 \
  --timeframe 1d \
  --epochs 100 \
  --spaces freqai \
  --freqai-backtest-live-models
```

## Configuration

### FreqAI Settings
- **Model**: LightGBM Classifier
- **Train Period**: 365 days
- **Backtest Period**: 90 days
- **Features**: 23 price-based indicators
- **Target**: Binary direction prediction

### Model Parameters
- **n_estimators**: 100
- **learning_rate**: 0.1
- **max_depth**: 6
- **subsample**: 0.8
- **colsample_bytree**: 0.8
- **eval_metric**: logloss

## Analysis Results

From the initial analysis (2020-2025 data):
- **Data Points**: 2,044 daily candles
- **Target Distribution**: 48.5% Up, 51.5% Down/Flat
- **Top Features**: High-Low range, ATR, price changes
- **Feature Count**: 23 total features

## Strategy Advantages

1. **Simplicity**: Easy to understand and debug
2. **Robustness**: Uses proven technical indicators
3. **Scalability**: Can easily add more features later
4. **Interpretability**: Clear feature importance
5. **Direction Focus**: More realistic for crypto markets

## Future Enhancements

1. **Add Volume Features**: Volume-based indicators
2. **Market Regime Detection**: Different models for bull/bear markets
3. **Feature Selection**: Automated feature importance filtering
4. **Ensemble Methods**: Combine multiple models
5. **Non-Price Features**: Gradually add Glassnode data

## Recommendations

1. **Start Simple**: Use this as a baseline before adding complexity
2. **Monitor Performance**: Track accuracy, precision, recall
3. **Feature Engineering**: Add features incrementally
4. **Cross-Validation**: Use walk-forward analysis
5. **Risk Management**: Implement proper position sizing

## Troubleshooting

### Common Issues
1. **Missing Data**: Ensure BTC data is available from 2020
2. **Memory Issues**: Reduce train/backtest periods
3. **Overfitting**: Monitor validation performance
4. **Feature Correlation**: Check for multicollinearity

### Performance Tuning
1. **Adjust Threshold**: Modify 0.5% threshold based on results
2. **Feature Selection**: Remove low-correlation features
3. **Model Parameters**: Tune via hyperopt
4. **Timeframe**: Experiment with different prediction horizons

## Comparison with Glassnode Strategy

| Aspect | Glassnode Strategy | Price Direction Strategy |
|--------|-------------------|-------------------------|
| Features | 50+ on-chain metrics | 23 price indicators |
| Target | 5-day regression | 3-day classification |
| Complexity | High | Low |
| Interpretability | Medium | High |
| Data Requirements | Glassnode API | Standard OHLCV |
| Overfitting Risk | High | Low |

This strategy provides a clean, simple foundation that you can build upon incrementally.
