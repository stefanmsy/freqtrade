import logging
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from freqtrade.strategy import IStrategy, DecimalParameter
from freqtrade.freqai.prediction_models.LightGBMClassifier import LightGBMClassifier
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class BTCPriceDirectionStrategyV2(IStrategy):
    """
    Focused BTC Price Direction Prediction Strategy V2 (Daily-Only)

    This strategy uses a carefully selected set of DAILY features based on:
    - Daily trend & regime indicators
    - Daily momentum signals (RSI, MACD)
    - Daily breakout strength measures
    - Daily volatility context & regime
    - Daily volume pressure indicators
    - Time-lagged target relevance

    Key improvements from V1:
    - Focused feature set (~10 features vs 40+)
    - Daily-only approach for simplicity and clarity
    - Feature correlation analysis
    - Lag features for target relevance
    - Cleaner signal logic without timeframe conflicts
    
    Future V3 will add multi-timeframe (daily trend + hourly entry)
    """

    # Use LightGBM Classifier for direction prediction
    freqai_model = LightGBMClassifier

    # Daily timeframe for main strategy
    timeframe = "1d"

    def informative_pairs(self):
        """Daily-only strategy - no additional timeframes needed."""
        return []

    # No ROI - rely entirely on AI signal for exits
    minimal_roi = {
        "0": 0.99,  # Effectively disabled
    }

    # Conservative stoploss for focused approach
    stoploss = -0.12  # 12% base stoploss

    use_exit_signal = True
    startup_candle_count: int = 100  # More history for multi-timeframe features
    can_short = True  # Enable shorting for futures trading

    # Disable trailing stop for cleaner signals
    trailing_stop = False
    process_only_new_candles = True

    # Direction classification threshold (simplified)
    direction_threshold = DecimalParameter(0.01, 0.03, default=0.015, space="buy", optimize=True)

    plot_config = {
        "main_plot": {
            "close": {"color": "blue"},
        },
        "subplots": {
            "&-target_3d": {"&-target_3d": {"color": "green"}},
            "do_predict": {"do_predict": {"color": "orange"}},
            "daily_features": {
                "%-daily_sma_regime": {"color": "purple"},
                "%-daily_rsi": {"color": "red"},
                "%-close_high_ratio": {"color": "orange"},
                "%-vol_regime": {"color": "gray"},
            },
        },
    }

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Build focused DAILY-ONLY feature set for direction prediction.
        Approximately 10 carefully selected daily features based on your requirements.
        """

        # ================================================================
        # DAILY TREND & REGIME FEATURES (2 features)
        # ================================================================

        # 1. Daily 20-SMA vs 50-SMA crossover (binary bull/bear)
        daily_sma_20 = dataframe["close"].rolling(20).mean()
        daily_sma_50 = dataframe["close"].rolling(50).mean()
        dataframe["%-daily_sma_regime"] = np.where(daily_sma_20 > daily_sma_50, 1, 0)

        # 2. Daily SMA slope: (SMA20 – SMA50) / SMA50
        dataframe["%-daily_sma_slope"] = (daily_sma_20 - daily_sma_50) / daily_sma_50

        # ================================================================
        # DAILY MOMENTUM FEATURES (2 features)
        # ================================================================

        # 3. Daily 14-period RSI (adapted from your hourly requirement)
        dataframe["%-daily_rsi"] = self._calculate_rsi(dataframe, 14)

        # 4. Daily MACD histogram (adapted from your hourly requirement)
        daily_macd = self._calculate_macd(dataframe)
        dataframe["%-daily_macd_hist"] = daily_macd["histogram"]

        # ================================================================
        # DAILY BREAKOUT STRENGTH FEATURES (2 features)
        # ================================================================

        # 5. Days since last 20-bar high (adapted from your "hours since high")
        daily_high_20 = dataframe["high"].rolling(20).max()
        days_since_high = []
        for i in range(len(dataframe)):
            if i < 20:
                days_since_high.append(np.nan)
            else:
                current_high = dataframe["high"].iloc[i]
                if current_high >= daily_high_20.iloc[i]:
                    days_since_high.append(0)
                else:
                    # Count days since last high
                    count = 1
                    for j in range(i - 1, max(0, i - 20), -1):
                        if dataframe["high"].iloc[j] >= daily_high_20.iloc[j]:
                            break
                        count += 1
                    days_since_high.append(min(count, 20))  # Cap at 20 days

        dataframe["%-days_since_high"] = days_since_high

        # 6. Daily close / 20-bar high ratio
        dataframe["%-close_high_ratio"] = dataframe["close"] / daily_high_20

        # ================================================================
        # DAILY VOLATILITY CONTEXT FEATURES (2 features)
        # ================================================================

        # 7. Daily ATR14 / daily close (percent)
        daily_atr = self._calculate_atr(dataframe, 14)
        dataframe["%-daily_atr_percent"] = daily_atr / dataframe["close"]

        # 8. Daily ATR14 raw (adapted from your hourly ATR requirement)
        dataframe["%-daily_atr_raw"] = daily_atr

        # ================================================================
        # DAILY VOLUME PRESSURE FEATURES (1 feature)
        # ================================================================

        # 9. Daily OBV momentum (ΔOBV over 5 days, adapted from 5 hours)
        if "volume" in dataframe.columns:
            daily_obv = self._calculate_obv(dataframe)
            dataframe["%-obv_momentum"] = daily_obv.diff(5)  # 5-day change
        else:
            dataframe["%-obv_momentum"] = 0

        # ================================================================
        # DAILY VOLATILITY REGIME FEATURES (1 feature)
        # ================================================================

        # 10. Daily vol14 divided by its 50-bar rolling mean (volatility regime flag)
        daily_returns = dataframe["close"].pct_change()
        daily_vol = daily_returns.rolling(14).std()
        daily_vol_mean = daily_vol.rolling(50).mean()
        dataframe["%-vol_regime"] = daily_vol / daily_vol_mean

        # ================================================================
        # TIME-LAGGED TARGET-RELEVANCE FEATURES (1 feature - added later)
        # ================================================================

        # 11. Lag-1 of the model's top-2 probability gap (prob_diff_lag1)
        # This will be populated in populate_indicators after FreqAI predictions
        dataframe["%-prob_diff_lag1"] = 0  # Placeholder

        # ================================================================
        # FEATURE CLEANING & VALIDATION
        # ================================================================

        # Clean and validate all features
        feature_columns = [col for col in dataframe.columns if col.startswith("%-")]
        for col in feature_columns:
            # Replace infinite values
            dataframe[col] = dataframe[col].replace([np.inf, -np.inf], np.nan)

            # Forward fill and backward fill
            dataframe[col] = dataframe[col].ffill().bfill()

            # Clip extreme values (winsorize at 1% and 99%)
            if dataframe[col].dtype in ["float64", "float32"]:
                q_low = dataframe[col].quantile(0.01)
                q_high = dataframe[col].quantile(0.99)
                dataframe[col] = dataframe[col].clip(lower=q_low, upper=q_high)

        # Log feature summary for monitoring
        logger.info(f"V2 Daily-Only Strategy Features: {len(feature_columns)} total features")
        logger.info(f"Feature columns: {feature_columns}")

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        Create target variable for 3-day direction prediction.
        Using more conservative threshold for cleaner signals.
        """

        # Set class names for 2-class classification
        self.freqai.class_names = ["down", "up"]

        # Calculate 3-day forward return
        forward_return = dataframe["close"].shift(-3) / dataframe["close"] - 1

        # Create 2-class classification target with conservative threshold
        # Use 2% threshold for clearer, more reliable signals
        threshold = 0.02  # 2% threshold for cleaner signals
        dataframe["&s-direction"] = np.where(
            forward_return > threshold,
            "up",
            "down",
        )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Required method for FreqTrade strategy.
        Includes lag feature creation after FreqAI predictions.
        """
        # Get FreqAI predictions
        dataframe = self.freqai.start(dataframe, metadata, self)

        # ================================================================
        # POST-PREDICTION LAG FEATURES
        # ================================================================

        # Create lag-1 of probability difference for target relevance
        if "up" in dataframe.columns and "down" in dataframe.columns:
            # Calculate probability difference
            dataframe["prob_diff"] = abs(dataframe["up"] - dataframe["down"])

            # Create lag-1 feature
            dataframe["%-prob_diff_lag1"] = dataframe["prob_diff"].shift(1)

            # Clean lag feature
            dataframe["%-prob_diff_lag1"] = dataframe["%-prob_diff_lag1"].fillna(0)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Simplified entry logic based on focused feature predictions.
        """
        # Initialize entry signals
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        # Check for FreqAI prediction columns
        if (
            "do_predict" in dataframe.columns
            and "up" in dataframe.columns
            and "down" in dataframe.columns
        ):
            # Calculate prediction confidence
            dataframe["max_prob"] = dataframe[["up", "down"]].max(axis=1)
            dataframe["prob_diff"] = abs(dataframe["up"] - dataframe["down"])

            # Conservative thresholds for focused approach
            min_confidence = 0.65  # Higher confidence required
            min_prob_diff = 0.30  # Clear probability separation

            # Base prediction mask
            confident_mask = (
                (dataframe["do_predict"] == 1)
                & (dataframe["max_prob"] > min_confidence)
                & (dataframe["prob_diff"] > min_prob_diff)
            )

            # Regime-based filtering using our focused daily features
            bull_regime = dataframe["%-daily_sma_regime"] == 1
            strong_momentum = dataframe["%-daily_rsi"] > 55  # Daily RSI now
            low_volatility = dataframe["%-vol_regime"] < 1.5  # Not in high vol regime

            # Long entry: Strong bullish prediction + supportive regime
            long_mask = (
                confident_mask
                & (dataframe["up"] > dataframe["down"])
                & (bull_regime | strong_momentum)  # Either bull regime OR strong momentum
                & low_volatility  # Avoid high volatility periods
            )

            # Short entry: Strong bearish prediction + bearish regime
            bear_regime = dataframe["%-daily_sma_regime"] == 0
            weak_momentum = dataframe["%-daily_rsi"] < 45  # Daily RSI now

            short_mask = (
                confident_mask
                & (dataframe["down"] > dataframe["up"])
                & bear_regime  # Only short in clear bear regime
                & weak_momentum  # And weak momentum
                & low_volatility  # Avoid high volatility periods
            )

            dataframe.loc[long_mask, "enter_long"] = 1
            dataframe.loc[short_mask, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Simplified exit logic based on prediction reversal.
        """
        # Initialize exit signals
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        # Use same confidence thresholds as entry
        if (
            "do_predict" in dataframe.columns
            and "up" in dataframe.columns
            and "down" in dataframe.columns
        ):
            min_confidence = 0.65
            min_prob_diff = 0.30

            confident_mask = (
                (dataframe["do_predict"] == 1)
                & (dataframe[["up", "down"]].max(axis=1) > min_confidence)
                & (abs(dataframe["up"] - dataframe["down"]) > min_prob_diff)
            )

            # Exit long when strong bearish prediction
            exit_long_mask = confident_mask & (dataframe["down"] > dataframe["up"])

            # Exit short when strong bullish prediction
            exit_short_mask = confident_mask & (dataframe["up"] > dataframe["down"])

            dataframe.loc[exit_long_mask, "exit_long"] = 1
            dataframe.loc[exit_short_mask, "exit_short"] = 1

        return dataframe

    # ================================================================
    # TECHNICAL INDICATOR HELPER METHODS
    # ================================================================

    def _calculate_rsi(self, dataframe: DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = dataframe["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
        self, dataframe: DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> dict:
        """Calculate MACD."""
        exp1 = dataframe["close"].ewm(span=fast).mean()
        exp2 = dataframe["close"].ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

    def _calculate_atr(self, dataframe: DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = dataframe["high"] - dataframe["low"]
        high_close = np.abs(dataframe["high"] - dataframe["close"].shift())
        low_close = np.abs(dataframe["low"] - dataframe["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def _calculate_obv(self, dataframe: DataFrame) -> pd.Series:
        """Calculate On-Balance Volume."""
        obv = []
        obv.append(0)

        for i in range(1, len(dataframe)):
            if dataframe["close"].iloc[i] > dataframe["close"].iloc[i - 1]:
                obv.append(obv[-1] + dataframe["volume"].iloc[i])
            elif dataframe["close"].iloc[i] < dataframe["close"].iloc[i - 1]:
                obv.append(obv[-1] - dataframe["volume"].iloc[i])
            else:
                obv.append(obv[-1])

        return pd.Series(obv, index=dataframe.index)

    def custom_stoploss(
        self,
        pair: str,
        trade: "Trade",
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        Simplified custom stoploss using ATR for volatility adjustment.
        """
        # Get the dataframe for this pair to access ATR
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe is not None and len(dataframe) > 0:
            # Use daily ATR percent for stoploss calculation
            current_atr_percent = dataframe["%-daily_atr_percent"].iloc[-1]

            # Volatility-scaled stoploss (2x ATR distance)
            vol_scaled_sl = -2.0 * current_atr_percent

            # Ensure reasonable bounds
            vol_scaled_sl = max(vol_scaled_sl, -0.20)  # Max 20% stoploss
            vol_scaled_sl = min(vol_scaled_sl, -0.06)  # Min 6% stoploss
        else:
            # Fallback to base stoploss
            vol_scaled_sl = -0.12

        # Simple profit-based tightening
        if current_profit > 0.12:  # If profit > 12%
            return max(vol_scaled_sl, -0.06)  # Tight stoploss at 6%
        elif current_profit > 0.08:  # If profit > 8%
            return max(vol_scaled_sl, -0.10)  # Medium stoploss at 10%
        else:
            return vol_scaled_sl  # Use volatility-scaled stoploss

    # ================================================================
    # FEATURE ANALYSIS METHODS
    # ================================================================

    def analyze_feature_correlation(self, dataframe: DataFrame) -> dict:
        """
        Analyze correlation between features to identify redundancy.
        Call this method during backtesting for feature analysis.
        """
        feature_columns = [col for col in dataframe.columns if col.startswith("%-")]

        if len(feature_columns) == 0:
            return {}

        # Calculate correlation matrix
        corr_matrix = dataframe[feature_columns].corr()

        # Find highly correlated pairs (>0.8)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = abs(corr_matrix.iloc[i, j])
                if corr_val > 0.8:
                    high_corr_pairs.append(
                        {
                            "feature1": corr_matrix.columns[i],
                            "feature2": corr_matrix.columns[j],
                            "correlation": corr_val,
                        }
                    )

        return {
            "correlation_matrix": corr_matrix,
            "high_correlation_pairs": high_corr_pairs,
            "feature_count": len(feature_columns),
        }
