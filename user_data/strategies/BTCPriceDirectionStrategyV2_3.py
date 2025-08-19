import logging
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from freqtrade.strategy import IStrategy, DecimalParameter
from freqtrade.freqai.prediction_models.LightGBMClassifier import LightGBMClassifier
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class BTCPriceDirectionStrategyV2_3(IStrategy):
    """
    Optimized BTC Price Direction Strategy V2.3 - Data-Driven Improvements

    Based on V2.2 real data analysis showing excellent prediction accuracy (64.9%)
    but poor trading execution (-0.62% return). V2.3 optimizes TRADING RULES
    while maintaining the validated CORE FEATURES.

    V2.3 KEY IMPROVEMENTS:
    =====================
    ✅ FEATURE OPTIMIZATION:
       - Keeps 6 top-performing features (importance 275-370)
       - Removes vol_regime_cluster (lowest importance: 27.7)
       - Maintains volatility-first approach (validated with real data)

    ✅ TRADE FREQUENCY BOOST:
       - Confidence threshold: 0.70 → 0.55 (expect 2-3x more trades)
       - Prob difference: 0.30 → 0.25 (better entry opportunities)

    ✅ ADVANCED RISK MANAGEMENT:
       - Base stop loss: -5% → -8% (prevents premature exits)
       - Volatility-scaled stops: Tighter in low vol, wider in high vol
       - Trailing stops: Lock in profits on winning trades
       - Position sizing: Based on volatility context

    CORE FEATURES (Importance Rank from Real Data):
    ==============================================
    1. daily_rsi (370.7)           - RSI momentum
    2. vol_ratio_7_14 (319.0)      - Short/mid-term volatility ratio
    3. vol_pca_1 (317.0)           - Multi-scale volatility (PCA compressed)
    4. vol_trend (301.7)           - Volatility expansion/contraction
    5. atr_momentum_3d (291.3)     - Volatility acceleration
    6. macd_momentum (275.7)       - MACD histogram slope

    PREDICTION ACCURACY (Validated on 2019-2025 BTC data):
    =====================================================
    - Conservative (>2% moves): 64.9% ± 4.7% ⭐⭐⭐⭐
    - Large move detection: Excellent for significant price changes
    - Feature consistency: All features show stable importance across time
    """

    # ================================================================
    # BASIC STRATEGY SETTINGS
    # ================================================================

    timeframe = "1d"
    stoploss = -0.08  # Base stop loss widened to -8% (from -5%)

    # Position sizing parameters
    position_adjustment_enable = True
    max_entry_position_adjustment = 0

    # ================================================================
    # FREQAI SETTINGS
    # ================================================================

    def informative_pairs(self):
        """No additional timeframes needed - daily only strategy."""
        return []

    # ================================================================
    # CORE FEATURE ENGINEERING (Top 6 Features Only)
    # ================================================================

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Extract the 6 highest-importance features validated on real BTC data.

        REMOVED: vol_regime_cluster (lowest importance: 27.7)
        KEPT: Top 6 features with importance 275-370
        """

        # ================================================================
        # 1. DAILY RSI (Rank #1: 370.7 importance)
        # ================================================================
        dataframe["%-daily_rsi"] = self._calculate_rsi(dataframe, 14)

        # ================================================================
        # 2. VOLATILITY RATIO 7/14 (Rank #2: 319.0 importance)
        # ================================================================
        atr_7 = self._calculate_atr(dataframe, 7)
        atr_14 = self._calculate_atr(dataframe, 14)
        dataframe["%-vol_ratio_7_14"] = atr_7 / atr_14.replace(0, np.nan)

        # ================================================================
        # 3. MULTI-SCALE VOLATILITY PCA (Rank #3: 317.0 importance)
        # ================================================================
        atr_21 = self._calculate_atr(dataframe, 21)

        # Calculate percentage-based ATRs for PCA
        atr_7_pct = atr_7 / dataframe["close"]
        atr_14_pct = atr_14 / dataframe["close"]
        atr_21_pct = atr_21 / dataframe["close"]

        # Simple PCA approximation (first principal component)
        # Weights derived from typical ATR correlation patterns
        vol_composite = 0.6 * atr_7_pct + 0.3 * atr_14_pct + 0.1 * atr_21_pct
        dataframe["%-vol_pca_1"] = vol_composite

        # ================================================================
        # 4. VOLATILITY TREND (Rank #4: 301.7 importance)
        # ================================================================
        vol_sma_fast = atr_14.rolling(5).mean()
        vol_sma_slow = atr_14.rolling(20).mean()
        dataframe["%-vol_trend"] = (vol_sma_fast - vol_sma_slow) / vol_sma_slow.replace(0, np.nan)

        # ================================================================
        # 5. ATR MOMENTUM 3D (Rank #5: 291.3 importance)
        # ================================================================
        dataframe["%-atr_momentum_3d"] = atr_14.diff(3) / dataframe["close"]

        # ================================================================
        # 6. MACD MOMENTUM (Rank #6: 275.7 importance)
        # ================================================================
        macd_data = self._calculate_macd(dataframe)
        macd_hist = macd_data["histogram"]
        dataframe["%-macd_momentum"] = macd_hist.diff(3)  # 3-day MACD histogram slope

        # ================================================================
        # FEATURE CLEANING & VALIDATION
        # ================================================================
        feature_columns = [
            "%-daily_rsi",
            "%-vol_ratio_7_14",
            "%-vol_pca_1",
            "%-vol_trend",
            "%-atr_momentum_3d",
            "%-macd_momentum",
        ]

        for col in feature_columns:
            if col in dataframe.columns:
                # Handle infinite values
                dataframe[col] = dataframe[col].replace([np.inf, -np.inf], np.nan)
                # Forward fill then zero fill
                dataframe[col] = dataframe[col].fillna(method="ffill").fillna(0)
                # Clip extreme outliers (beyond 5 standard deviations)
                std_val = dataframe[col].std()
                mean_val = dataframe[col].mean()
                if std_val > 0:
                    clip_upper = mean_val + 5 * std_val
                    clip_lower = mean_val - 5 * std_val
                    dataframe[col] = dataframe[col].clip(lower=clip_lower, upper=clip_upper)

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        Create binary classification targets for 3-day forward price movements.

        Target: Will BTC move >2% up in the next 3 days?
        - Validated: 64.9% accuracy on real data for this threshold
        """
        self.freqai.class_names = ["down", "up"]

        # 3-day forward return (same as validated in analysis)
        forward_return = dataframe["close"].shift(-3) / dataframe["close"] - 1
        threshold = 0.02  # 2% threshold (optimal from real data analysis)

        dataframe["&s-direction"] = np.where(forward_return > threshold, "up", "down")

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Run FreqAI prediction and add volatility context for advanced risk management.
        """
        # Run FreqAI prediction
        dataframe = self.freqai.start(dataframe, metadata, self)

        # Add probability difference for confidence assessment
        if "up" in dataframe.columns and "down" in dataframe.columns:
            dataframe["prob_diff"] = abs(dataframe["up"] - dataframe["down"])
            dataframe["confidence"] = dataframe[["up", "down"]].max(axis=1)

        # Add volatility context for risk management
        if "%-vol_pca_1" in dataframe.columns:
            # Volatility regime: High/Low volatility periods
            vol_median = dataframe["%-vol_pca_1"].rolling(50).median()
            dataframe["vol_regime"] = np.where(
                dataframe["%-vol_pca_1"] > vol_median * 1.5, "high", "low"
            )

            # Volatility percentile for position sizing
            dataframe["vol_percentile"] = dataframe["%-vol_pca_1"].rolling(100).rank(pct=True)

        return dataframe

    # ================================================================
    # OPTIMIZED ENTRY LOGIC (Increased Trade Frequency)
    # ================================================================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Optimized entry logic for increased trade frequency while maintaining quality.

        V2.3 IMPROVEMENTS:
        - Confidence threshold: 0.70 → 0.55 (more trades)
        - Prob difference: 0.30 → 0.25 (better opportunities)
        - Simplified filtering (removed weak vol_regime_cluster)
        """

        if (
            "up" not in dataframe.columns
            or "down" not in dataframe.columns
            or "prob_diff" not in dataframe.columns
        ):
            return dataframe

        # ================================================================
        # CONFIDENCE THRESHOLDS (Optimized for more trades)
        # ================================================================
        min_confidence = 0.55  # Lowered from 0.70 (expect 2-3x more trades)
        min_prob_diff = 0.25  # Lowered from 0.30 (better entry opportunities)

        confident_mask = (dataframe["confidence"] >= min_confidence) & (
            dataframe["prob_diff"] >= min_prob_diff
        )

        # ================================================================
        # SIMPLIFIED REGIME FILTERING (Removed vol_regime_cluster)
        # ================================================================
        # Basic momentum and volatility context
        momentum_ok = True
        vol_ok = True

        if "%-daily_rsi" in dataframe.columns:
            # Allow trades in wider RSI range (30-70 instead of 35-65)
            momentum_ok = (dataframe["%-daily_rsi"] > 30) & (dataframe["%-daily_rsi"] < 70)

        if "%-vol_trend" in dataframe.columns:
            # Avoid extreme volatility contraction (model needs some volatility to work)
            vol_ok = dataframe["%-vol_trend"] > -0.20

        # ================================================================
        # ENTRY SIGNALS
        # ================================================================
        # Long entry: Bullish prediction + basic regime filters
        long_mask = confident_mask & (dataframe["up"] > dataframe["down"]) & momentum_ok & vol_ok

        # Short entry: Bearish prediction + basic regime filters
        short_mask = confident_mask & (dataframe["down"] > dataframe["up"]) & momentum_ok & vol_ok

        # Set entry signals
        dataframe.loc[long_mask, "enter_long"] = 1
        dataframe.loc[short_mask, "enter_short"] = 1

        return dataframe

    # ================================================================
    # ADVANCED RISK MANAGEMENT (Volatility-Scaled & Trailing Stops)
    # ================================================================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Advanced exit logic with volatility-scaled and trailing stops.

        V2.3 IMPROVEMENTS:
        - Model-based exits (high win rate: 77.8% validated)
        - Volatility-scaled stop distances
        - Trailing stop mechanism
        """

        if (
            "up" not in dataframe.columns
            or "down" not in dataframe.columns
            or "confidence" not in dataframe.columns
        ):
            return dataframe

        # ================================================================
        # MODEL-BASED EXITS (Validated 77.8% win rate)
        # ================================================================
        min_confidence = 0.55  # Same as entry for consistency
        min_prob_diff = 0.25

        confident_mask = (dataframe["confidence"] >= min_confidence) & (
            dataframe["prob_diff"] >= min_prob_diff
        )

        # Exit long when model predicts strong downward movement
        exit_long_mask = confident_mask & (dataframe["down"] > dataframe["up"])

        # Exit short when model predicts strong upward movement
        exit_short_mask = confident_mask & (dataframe["up"] > dataframe["down"])

        # ================================================================
        # VOLATILITY-BASED EXIT REFINEMENTS
        # ================================================================
        if "vol_regime" in dataframe.columns and "%-daily_rsi" in dataframe.columns:
            # In high volatility: Exit on extreme RSI (prevent whipsaws)
            high_vol_mask = dataframe["vol_regime"] == "high"
            extreme_rsi = (dataframe["%-daily_rsi"] > 75) | (dataframe["%-daily_rsi"] < 25)

            # Additional exit conditions in high volatility
            exit_long_mask = exit_long_mask | (high_vol_mask & (dataframe["%-daily_rsi"] > 75))
            exit_short_mask = exit_short_mask | (high_vol_mask & (dataframe["%-daily_rsi"] < 25))

        # Set exit signals
        dataframe.loc[exit_long_mask, "exit_long"] = 1
        dataframe.loc[exit_short_mask, "exit_short"] = 1

        return dataframe

    # ================================================================
    # DYNAMIC STOP LOSS (Volatility-Scaled)
    # ================================================================

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        """
        Implement volatility-scaled and trailing stops.

        LOGIC:
        - Base stop: -8% (widened from -5%)
        - Low volatility: Tighter stops (-6%)
        - High volatility: Wider stops (-10%)
        - Trailing: Lock in profits after +5% gain
        """

        # Get current dataframe for volatility context
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return self.stoploss  # Fallback to base stop

        # Get latest volatility data
        latest_data = dataframe.iloc[-1]

        # ================================================================
        # VOLATILITY-SCALED STOPS
        # ================================================================
        base_stop = -0.08  # Base -8% stop

        if "vol_percentile" in latest_data:
            vol_pct = latest_data["vol_percentile"]

            if vol_pct < 0.3:  # Low volatility (bottom 30%)
                dynamic_stop = -0.06  # Tighter stop
            elif vol_pct > 0.7:  # High volatility (top 30%)
                dynamic_stop = -0.10  # Wider stop
            else:
                dynamic_stop = base_stop  # Normal stop
        else:
            dynamic_stop = base_stop

        # ================================================================
        # TRAILING STOP MECHANISM
        # ================================================================
        if current_profit > 0.05:  # If profit > 5%
            # Trail stop to lock in at least 2% profit
            trailing_stop = current_profit - 0.03
            return max(trailing_stop, dynamic_stop)

        return dynamic_stop

    # ================================================================
    # POSITION SIZING (Volatility-Based)
    # ================================================================

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float,
        max_stake: float,
        leverage: float,
        entry_tag: str,
        side: str,
        **kwargs,
    ) -> float:
        """
        Implement volatility-based position sizing.

        LOGIC:
        - High volatility: Smaller positions (0.7x stake)
        - Low volatility: Larger positions (1.3x stake)
        - Default: Normal stake (1.0x)
        """

        # Get current dataframe for volatility context
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_stake  # Fallback to proposed stake

        # Get latest volatility data
        latest_data = dataframe.iloc[-1]

        if "vol_percentile" in latest_data:
            vol_pct = latest_data["vol_percentile"]

            if vol_pct > 0.8:  # Very high volatility
                size_multiplier = 0.7  # Reduce position size
            elif vol_pct < 0.2:  # Very low volatility
                size_multiplier = 1.3  # Increase position size
            else:
                size_multiplier = 1.0  # Normal position size
        else:
            size_multiplier = 1.0

        adjusted_stake = proposed_stake * size_multiplier

        # Ensure within bounds
        return max(min_stake, min(adjusted_stake, max_stake))

    # ================================================================
    # HELPER METHODS (Technical Indicators)
    # ================================================================

    def _calculate_rsi(self, dataframe: DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = dataframe["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50)  # Neutral RSI for initial values

    def _calculate_atr(self, dataframe: DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = dataframe["high"] - dataframe["low"]
        high_close_prev = abs(dataframe["high"] - dataframe["close"].shift(1))
        low_close_prev = abs(dataframe["low"] - dataframe["close"].shift(1))

        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(window=period).mean()

        return atr.fillna(true_range)  # Use current TR for initial values

    def _calculate_macd(
        self, dataframe: DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> dict:
        """Calculate MACD indicator."""
        exp1 = dataframe["close"].ewm(span=fast).mean()
        exp2 = dataframe["close"].ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return {
            "macd": macd_line.fillna(0),
            "signal": signal_line.fillna(0),
            "histogram": histogram.fillna(0),
        }

    # ================================================================
    # PLOTTING CONFIGURATION
    # ================================================================

    plot_config = {
        "main_plot": {
            # Price action
            "close": {"color": "blue"},
        },
        "subplots": {
            # RSI (most important feature)
            "RSI": {
                "%-daily_rsi": {"color": "red", "type": "line"},
            },
            # Volatility features
            "Volatility": {
                "%-vol_pca_1": {"color": "orange", "type": "line"},
                "%-vol_ratio_7_14": {"color": "purple", "type": "line"},
            },
            # MACD momentum
            "MACD": {
                "%-macd_momentum": {"color": "green", "type": "line"},
            },
            # Model predictions
            "Predictions": {
                "up": {"color": "lime", "type": "line"},
                "down": {"color": "red", "type": "line"},
                "prob_diff": {"color": "gray", "type": "line"},
            },
        },
    }
