# Glassnode Swing Trading Strategy

## Overview

The `GlassnodeSwingStrategy` is a machine learning-based swing trading strategy that combines on-chain metrics from Glassnode with technical indicators to predict 5-day forward returns for cryptocurrency trading.

## Features

### Data Sources
- **Price Data**: Downloaded via freqtrade (1d timeframe)
- **On-chain Metrics**: Prefer Glassnode daily data (`btc-glassnode_1d.csv`), fallback to hourly (`btc-glassnode_1h.csv`) resampled to daily
- **Holder Supply Data**: Glassnode daily LTH/STH data (`lth_sth_supply_1d.csv`)

### Available On-chain Features
- `sopr` - Spent Output Profit Ratio
- `exchange_netflow` - Exchange Net Flow
- `realized_pnl_ratio` - Realized Profit/Loss Ratio
- `nvt_price` - Network Value to Transactions Price
- `nvt_price_90D` - 90-day NVT Price
- `90D_NVT_premium` - 90-day NVT Premium
- `whale_exchange_netflow` - Whale Exchange Net Flow
- `long_term_holder_supply` - Long-term Holder Supply
- `short_term_holder_supply` - Short-term Holder Supply

### Engineered Features
- **Technical Indicators**: SMA/EMA (10, 50, 200), RSI, MACD
- **Price Features**: Momentum, volatility, price changes (1d, 5d, 10d)
- **On-chain Features**: 
  - SOPR rolling min, z-score
  - Exchange netflow EMA, momentum
  - Realized PnL momentum
  - NVT ratio and momentum
  - LTH/STH percentages and momentum

### Target
- **5-day Forward Returns**: Predicts price movement over the next 5 days (5 candles for 1d data)

## Configuration

### Strategy Settings
- **Timeframe**: 1d
- **Model**: LightGBMRegressor
- **Training Period**: 365 days
- **Backtest Period**: 60 days
- **Label Period**: 5 candles (5 days)

### Trading Logic
- **Long Entry**: AI predicts >2% return + volume confirmation + on-chain signals
- **Short Entry**: AI predicts <-2% return + volume confirmation + on-chain signals
- **Exit**: Based on AI signal reversal or on-chain exit conditions

## Usage

### 1. Download Price Data
```bash
freqtrade download-data --exchange binance --pairs BTC/USDT:USDT --timeframe 1d
```

### 2. Run Backtest
```bash
freqtrade backtesting --config user_data/strategies/config_glassnode_swing.json --strategy GlassnodeSwingStrategy
```

### 3. Run Hyperopt (Optional)
```bash
freqtrade hyperopt --config user_data/strategies/config_glassnode_swing.json --strategy GlassnodeSwingStrategy --hyperopt-loss SharpeHyperOptLoss
```

## Data Requirements

### Required Files
- `user_data/data/glassnode/btc-glassnode_1d.csv` - Daily on-chain metrics (preferred)
- `user_data/data/glassnode/btc-glassnode_1h.csv` - Hourly on-chain metrics (fallback)
- `user_data/data/glassnode/lth_sth_supply_1d.csv` - Daily holder supply data

### Data Format
- **Daily Data**: CSV with timestamp and on-chain metrics
- **Daily Data**: CSV with timestamp, LTH_supply, STH_supply
- **Date Format**: ISO or DD/MM/YYYY; time optional

## Testing

Run the test script to validate data loading and feature engineering:
```bash
python user_data/strategies/test_glassnode_data.py
```

## Key Features

### Robust Data Handling
- Automatic timestamp alignment between price and on-chain data
- Forward-fill for daily data resampling to hourly
- Comprehensive NaN handling and feature engineering

### Advanced Feature Engineering
- Multiple timeframe expansion via FreqAI
- On-chain metric engineering (z-scores, momentum, ratios)
- Technical indicator integration

### Risk Management
- Conservative entry thresholds (2% expected return)
- On-chain confirmation signals
- Proper exit conditions based on signal reversal

## Performance Considerations

- **Training Time**: Fast for 1d with minimal features
- **Memory Usage**: Moderate (handles large feature sets efficiently)
- **Feature Importance**: Automatically plots top 20 features
- **Outlier Detection**: SVM-based outlier removal enabled

## Troubleshooting

### Common Issues
1. **Missing Data Files**: Ensure Glassnode CSV files are in the correct location (`btc-glassnode_1d.csv` preferred)
2. **Date Format Errors**: Check that timestamps match expected format
3. **Feature Engineering Errors**: Verify all required columns are present in data

### Validation
- Use the test script to validate data loading
- Check logs for data coverage information
- Monitor feature importance plots for model insights

## Future Enhancements

- Add more on-chain metrics (MVRV, dormancy, etc.)
- Implement cross-validation for better model validation
- Add more sophisticated entry/exit conditions
- Optimize hyperparameters for different market conditions 