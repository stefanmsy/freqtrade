import logging
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from freqtrade.strategy import IStrategy, DecimalParameter
from freqtrade.freqai.prediction_models.LightGBMClassifier import LightGBMClassifier
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class BTCPriceDirectionStrategyV2_1(IStrategy):
    """
    Enhanced BTC Price Direction Prediction Strategy V2.1

    Based on comprehensive V2.0 analysis, this version focuses on:
    - High-impact volatility features (multi-scale ATR with PCA)
    - Enhanced momentum divergence (MACD slope, breakout persistence)
    - Proper lagged features for persistence/mean-reversion
    - Volatility-based regime detection (not SMA crossover)
    - Removed redundant features (binary regime, correlated features)

    Key improvements from V2.0:
    - Removed: daily_sma_regime, close_high_ratio (redundant/low-value)
    - Enhanced: Multi-scale volatility features with dimensionality reduction
    - Added: Proper lag features, breakout persistence, MACD momentum
    - Replaced: SMA regime with volatility clustering detection
    - Target: ~8-10 high-quality features vs V2.0's 10 mixed-quality features
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
    startup_candle_count: int = 100  # More history for lag features
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
            "v2_1_features": {
                "%-vol_pca_1": {"color": "purple"},
                "%-daily_rsi": {"color": "red"},
                "%-macd_momentum": {"color": "orange"},
                "%-vol_regime_cluster": {"color": "gray"},
            },
        },
    }

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Build V2.1 enhanced feature set focusing on high-impact indicators.
        Approximately 8-10 carefully engineered features.
        """

        # ================================================================
        # 1. MULTI-SCALE VOLATILITY FEATURES (TOP PRIORITY)
        # ================================================================

        # Calculate multi-scale ATRs
        atr_7 = self._calculate_atr(dataframe, 7)
        atr_14 = self._calculate_atr(dataframe, 14)
        atr_21 = self._calculate_atr(dataframe, 21)

        # Normalized ATRs (percent of price)
        dataframe["atr_7_pct"] = atr_7 / dataframe["close"]
        dataframe["atr_14_pct"] = atr_14 / dataframe["close"]
        dataframe["atr_21_pct"] = atr_21 / dataframe["close"]

        # Volatility ratios (multi-timeframe relationships)
        dataframe["%-vol_ratio_7_14"] = atr_7 / (atr_14 + 1e-8)
        dataframe["%-vol_ratio_14_21"] = atr_14 / (atr_21 + 1e-8)

        # ATR momentum (volatility acceleration/deceleration)
        dataframe["%-atr_momentum_3d"] = atr_14.diff(3) / (atr_14.shift(3) + 1e-8)

        # Apply PCA to compress multi-scale ATRs into 2 factors
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            # Prepare ATR data for PCA
            atr_features = (
                dataframe[["atr_7_pct", "atr_14_pct", "atr_21_pct"]]
                .fillna(method="ffill")
                .fillna(0)
            )

            if len(atr_features.dropna()) > 50:  # Enough data for PCA
                scaler = StandardScaler()
                atr_scaled = scaler.fit_transform(atr_features)

                pca = PCA(n_components=2)
                atr_pca = pca.fit_transform(atr_scaled)

                dataframe["%-vol_pca_1"] = atr_pca[:, 0]  # First volatility factor
                dataframe["%-vol_pca_2"] = atr_pca[:, 1]  # Second volatility factor

                logger.info(f"Volatility PCA explained variance: {pca.explained_variance_ratio_}")
            else:
                # Fallback if insufficient data
                dataframe["%-vol_pca_1"] = dataframe["atr_14_pct"]
                dataframe["%-vol_pca_2"] = dataframe["%-vol_ratio_14_21"]

        except ImportError:
            # Fallback if sklearn not available
            dataframe["%-vol_pca_1"] = dataframe["atr_14_pct"]
            dataframe["%-vol_pca_2"] = dataframe["%-vol_ratio_14_21"]

        # ================================================================
        # 2. ENHANCED MOMENTUM DIVERGENCE FEATURES
        # ================================================================

        # Daily RSI (keep for interpretability, remove correlated close_high_ratio)
        dataframe["%-daily_rsi"] = self._calculate_rsi(dataframe, 14)

        # MACD histogram and its momentum (slope)
        macd_data = self._calculate_macd(dataframe)
        dataframe["%-daily_macd_hist"] = macd_data["histogram"]
        dataframe["%-macd_momentum"] = macd_data["histogram"].diff(3)  # 3-day slope

        # Breakout persistence count (sustained momentum)
        daily_high_20 = dataframe["high"].rolling(20).max()
        breakout_count = []

        for i in range(len(dataframe)):
            if i < 20:
                breakout_count.append(0)
            else:
                # Count consecutive days where close > 20-day high
                consecutive_days = 0
                for j in range(i, max(0, i - 10), -1):  # Look back max 10 days
                    if (
                        dataframe["close"].iloc[j] > daily_high_20.iloc[j - 1]
                    ):  # Compare to previous day's 20-high
                        consecutive_days += 1
                    else:
                        break
                breakout_count.append(consecutive_days)

        dataframe["%-breakout_persistence"] = breakout_count

        # ================================================================
        # 3. VOLATILITY-BASED REGIME DETECTION (REPLACE SMA REGIME)
        # ================================================================

        # Volatility clustering regime (much better than SMA crossover)
        vol_regime = atr_14 / atr_14.rolling(50).mean()
        dataframe["%-vol_regime_cluster"] = np.where(
            vol_regime > 1.2, 1, 0
        )  # High volatility cluster

        # Volatility trend (expanding vs contracting volatility)
        dataframe["%-vol_trend"] = (atr_14.rolling(5).mean() / atr_14.rolling(20).mean()) - 1

        # ================================================================
        # 4. TREND STRENGTH (KEEP BEST FROM V2.0)
        # ================================================================

        # SMA slope (was good in V2.0, keep it)
        daily_sma_20 = dataframe["close"].rolling(20).mean()
        daily_sma_50 = dataframe["close"].rolling(50).mean()
        dataframe["%-daily_sma_slope"] = (daily_sma_20 - daily_sma_50) / daily_sma_50

        # ================================================================
        # 5. VOLUME MOMENTUM (SIMPLIFIED)
        # ================================================================

        # OBV momentum (was consistent in V2.0)
        if "volume" in dataframe.columns:
            daily_obv = self._calculate_obv(dataframe)
            dataframe["%-obv_momentum"] = daily_obv.diff(5)  # 5-day change
        else:
            dataframe["%-obv_momentum"] = 0

        # ================================================================
        # 6. LAGGED FEATURES PLACEHOLDER (FILLED IN populate_indicators)
        # ================================================================

        # These will be properly populated after FreqAI predictions
        dataframe["%-prob_diff_lag1"] = 0  # Lag-1 of probability difference
        dataframe["%-vol_lag1"] = 0  # Lag-1 of main volatility factor
        dataframe["%-rsi_lag2"] = 0  # Lag-2 of RSI for mean reversion
        dataframe["%-macd_lag1"] = 0  # Lag-1 of MACD momentum

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

        # Clean temporary columns
        temp_cols = ["atr_7_pct", "atr_14_pct", "atr_21_pct"]
        for col in temp_cols:
            if col in dataframe.columns:
                dataframe.drop(col, axis=1, inplace=True)

        # Log feature summary for monitoring
        logger.info(f"V2.1 Strategy Features: {len(feature_columns)} total features")
        logger.info(f"Feature columns: {feature_columns}")

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        Create target variable for 3-day direction prediction.
        Same as V2.0 but with more conservative threshold based on analysis.
        """

        # Set class names for 2-class classification
        self.freqai.class_names = ["down", "up"]

        # Calculate 3-day forward return
        forward_return = dataframe["close"].shift(-3) / dataframe["close"] - 1

        # Use 2.5% threshold (slightly higher than V2.0's 2%) for even cleaner signals
        threshold = 0.025  # 2.5% threshold for cleaner signals
        dataframe["&s-direction"] = np.where(
            forward_return > threshold,
            "up",
            "down",
        )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Required method for FreqTrade strategy.
        Enhanced with proper lag feature implementation.
        """
        # Get FreqAI predictions
        dataframe = self.freqai.start(dataframe, metadata, self)

        # ================================================================
        # PROPER LAGGED FEATURES IMPLEMENTATION
        # ================================================================

        # Create lag-1 of probability difference for target relevance
        if "up" in dataframe.columns and "down" in dataframe.columns:
            # Calculate probability difference
            dataframe["prob_diff"] = abs(dataframe["up"] - dataframe["down"])

            # Create lag-1 feature
            dataframe["%-prob_diff_lag1"] = dataframe["prob_diff"].shift(1).fillna(0)

        # Create lagged features for key indicators
        if "%-vol_pca_1" in dataframe.columns:
            dataframe["%-vol_lag1"] = dataframe["%-vol_pca_1"].shift(1).fillna(0)

        if "%-daily_rsi" in dataframe.columns:
            dataframe["%-rsi_lag2"] = dataframe["%-daily_rsi"].shift(2).fillna(0)

        if "%-macd_momentum" in dataframe.columns:
            dataframe["%-macd_lag1"] = dataframe["%-macd_momentum"].shift(1).fillna(0)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Enhanced entry logic using V2.1 focused features.
        Higher confidence thresholds due to better feature quality.
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

            # Higher confidence thresholds due to better features
            min_confidence = 0.70  # Increased from 0.65
            min_prob_diff = 0.35  # Increased from 0.30

            # Base prediction mask
            confident_mask = (
                (dataframe["do_predict"] == 1)
                & (dataframe["max_prob"] > min_confidence)
                & (dataframe["prob_diff"] > min_prob_diff)
            )

            # Enhanced regime-based filtering using V2.1 features

            # Volatility-based regime (better than SMA regime)
            low_vol_regime = (
                dataframe["%-vol_regime_cluster"] == 0
            )  # Not in high volatility cluster

            # Momentum alignment
            strong_momentum = dataframe["%-daily_rsi"] > 55
            weak_momentum = dataframe["%-daily_rsi"] < 45

            # Trend confirmation
            uptrend = dataframe["%-daily_sma_slope"] > 0.01  # 1% slope threshold
            downtrend = dataframe["%-daily_sma_slope"] < -0.01

            # Breakout persistence
            sustained_breakout = dataframe["%-breakout_persistence"] >= 2  # 2+ consecutive days

            # Long entry: Strong bullish prediction + supportive conditions
            long_mask = (
                confident_mask
                & (dataframe["up"] > dataframe["down"])
                & (strong_momentum | sustained_breakout)  # Either momentum OR sustained breakout
                & (uptrend | low_vol_regime)  # Either uptrend OR stable volatility
                & (dataframe["%-vol_trend"] > -0.1)  # Not in severe volatility contraction
            )

            # Short entry: Strong bearish prediction + bearish conditions
            short_mask = (
                confident_mask
                & (dataframe["down"] > dataframe["up"])
                & weak_momentum  # Weak momentum
                & (
                    downtrend | (dataframe["%-vol_regime_cluster"] == 1)
                )  # Downtrend OR high volatility
                & (dataframe["%-breakout_persistence"] == 0)  # No recent breakouts
            )

            dataframe.loc[long_mask, "enter_long"] = 1
            dataframe.loc[short_mask, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Enhanced exit logic using V2.1 features.
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
            min_confidence = 0.70
            min_prob_diff = 0.35

            confident_mask = (
                (dataframe["do_predict"] == 1)
                & (dataframe[["up", "down"]].max(axis=1) > min_confidence)
                & (abs(dataframe["up"] - dataframe["down"]) > min_prob_diff)
            )

            # Enhanced exit conditions with momentum reversal detection
            momentum_reversal = (
                abs(dataframe["%-macd_momentum"]) > dataframe["%-macd_momentum"].rolling(10).std()
            )

            # Exit long when strong bearish prediction OR momentum reversal
            exit_long_mask = (
                (confident_mask & (dataframe["down"] > dataframe["up"]))
                | (
                    momentum_reversal & (dataframe["%-daily_rsi"] > 70)
                )  # Overbought with momentum reversal
            )

            # Exit short when strong bullish prediction OR momentum reversal
            exit_short_mask = (
                (confident_mask & (dataframe["up"] > dataframe["down"]))
                | (
                    momentum_reversal & (dataframe["%-daily_rsi"] < 30)
                )  # Oversold with momentum reversal
            )

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
        Enhanced custom stoploss using multi-scale volatility.
        """
        # Get the dataframe for this pair to access volatility features
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe is not None and len(dataframe) > 0:
            # Use primary volatility factor for dynamic stoploss
            current_vol = (
                dataframe["%-vol_pca_1"].iloc[-1] if "%-vol_pca_1" in dataframe.columns else 0.02
            )

            # Volatility-scaled stoploss (1.5-3x volatility distance based on regime)
            vol_regime = (
                dataframe["%-vol_regime_cluster"].iloc[-1]
                if "%-vol_regime_cluster" in dataframe.columns
                else 0
            )

            if vol_regime == 1:  # High volatility regime
                vol_scaled_sl = -3.0 * abs(current_vol)  # Wider stops in high vol
            else:  # Normal volatility regime
                vol_scaled_sl = -2.0 * abs(current_vol)  # Tighter stops in low vol

            # Ensure reasonable bounds
            vol_scaled_sl = max(vol_scaled_sl, -0.25)  # Max 25% stoploss
            vol_scaled_sl = min(vol_scaled_sl, -0.05)  # Min 5% stoploss
        else:
            # Fallback to base stoploss
            vol_scaled_sl = -0.12

        # Enhanced profit-based tightening
        if current_profit > 0.15:  # If profit > 15%
            return max(vol_scaled_sl, -0.05)  # Tight stoploss at 5%
        elif current_profit > 0.10:  # If profit > 10%
            return max(vol_scaled_sl, -0.08)  # Medium stoploss at 8%
        elif current_profit > 0.05:  # If profit > 5%
            return max(vol_scaled_sl, -0.12)  # Loose stoploss at 12%
        else:
            return vol_scaled_sl  # Use volatility-scaled stoploss

    # ================================================================
    # FEATURE ANALYSIS METHODS (ENHANCED)
    # ================================================================

    def analyze_feature_groups(self, dataframe: DataFrame) -> dict:
        """
        Analyze feature groups for V2.1 iteration testing.
        Group features into volatility, momentum, and lag buckets.
        """
        feature_groups = {
            "volatility": [
                "%-vol_pca_1",
                "%-vol_pca_2",
                "%-vol_ratio_7_14",
                "%-vol_ratio_14_21",
                "%-atr_momentum_3d",
                "%-vol_regime_cluster",
                "%-vol_trend",
            ],
            "momentum": [
                "%-daily_rsi",
                "%-daily_macd_hist",
                "%-macd_momentum",
                "%-breakout_persistence",
                "%-daily_sma_slope",
            ],
            "lag": ["%-prob_diff_lag1", "%-vol_lag1", "%-rsi_lag2", "%-macd_lag1"],
            "volume": ["%-obv_momentum"],
        }

        # Check which features are available
        available_features = [col for col in dataframe.columns if col.startswith("%-")]

        analysis = {}
        for group, features in feature_groups.items():
            available_in_group = [f for f in features if f in available_features]
            analysis[group] = {
                "features": available_in_group,
                "count": len(available_in_group),
                "coverage": len(available_in_group) / len(features) if features else 0,
            }

        logger.info(f"V2.1 Feature Group Analysis: {analysis}")
        return analysis
