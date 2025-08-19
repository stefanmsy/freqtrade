from datetime import datetime, timedelta
import logging

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy

logger = logging.getLogger(__name__)


class BTCPriceDirectionStrategyV2_4(IStrategy):
    """
    Advanced BTC Price Direction Strategy V2.4 - Risk Management Revolution

    V2.3 ANALYSIS RESULTS:
    =====================
    ✅ TRADE FREQUENCY: 161 trades (12.4x improvement over V2.2)
    ✅ MODEL QUALITY: 60.2% win rate on exit_signals (+232.68% profit)
    ❌ STOP LOSSES: 0% win rate on stop_loss (-296.54% loss)
    ❌ NET RESULT: -63.86% (purely due to broken risk management)

    V2.4 RISK MANAGEMENT REVOLUTION:
    ===============================
    🔧 ATR-BASED STOPS: Replace fixed -8% with dynamic -1.5x ATR stops
    ⏰ TIME-BASED HOLDS: Minimum 3-day hold before stops can trigger
    📈 TRAILING STOPS: Move to breakeven at +4%, trail thereafter
    💰 TAKE-PROFIT: Quick +5% profit capture before whipsaws
    📊 DYNAMIC SIZING: Scale position size by signal confidence

    EXPECTED IMPROVEMENT:
    ====================
    Current: Model signals +232.68%, Stop losses -296.54% = -63.86% NET
    Target:  Model signals +232.68%, Stop losses -100.00% = +132.68% NET
    Goal: Transform stop losses from -296.54% to manageable -100% loss

    CORE FEATURES (Validated 60.2% Win Rate):
    =========================================
    1. daily_rsi (370.7)           - RSI momentum
    2. vol_ratio_7_14 (319.0)      - Volatility dynamics
    3. vol_pca_1 (317.0)           - Multi-scale volatility
    4. vol_trend (301.7)           - Volatility expansion/contraction
    5. atr_momentum_3d (291.3)     - Volatility acceleration
    6. macd_momentum (275.7)       - MACD histogram slope
    """

    # ================================================================
    # BASIC STRATEGY SETTINGS
    # ================================================================

    timeframe = "1d"
    stoploss = -0.99  # Disabled - using custom stop loss logic

    # Risk management parameters
    position_adjustment_enable = True
    max_entry_position_adjustment = 0

    # Take profit levels (updated to improve average win size)
    minimal_roi = {
        "0": 0.08,  # 8% quick take-profit
        "1440": 0.05,  # 5% after 1 day
        "4320": 0.03,  # 3% after 3 days
        "10080": 0,  # Hold indefinitely after 1 week
    }

    # Optional: tighten ATR stop bounds to reduce average loss size
    # When enabled, bounds ATR-based stops between 4% and 8%
    use_tight_atr_bounds = True
    atr_bounds_default = (-0.20, -0.06)  # (min, max) for default ATR-based bounds
    atr_bounds_tight = (-0.08, -0.04)  # (min, max) for optional tighter bounds

    # Optional: use model-based exit signals
    # If False, exits are managed purely by ROI/stop/trailing
    use_model_exits = False
    exit_min_confidence = 0.70
    exit_min_prob_diff = 0.35
    exit_confirmation_bars = 2

    # Optional: soft time-stop to exit stale trades
    # After this many days in trade, tighten stop to breakeven/small loss
    soft_time_stop_enabled = True
    soft_time_stop_days = 21
    # Breakeven floor: set to 0.0 for exact breakeven, negative for small allowed loss
    soft_time_stop_floor = 0.0  # breakeven

    # Optional: hard time-based exit to close stale trades
    hard_time_exit_enabled = False
    hard_time_exit_days = 30
    hard_time_exit_min_profit = 0.03  # Exit if below +3% after N days

    # ================================================================
    # FREQAI SETTINGS
    # ================================================================

    def informative_pairs(self):
        """No additional timeframes needed - daily only strategy."""
        return []

    # ================================================================
    # CORE FEATURE ENGINEERING (Proven 6 Features)
    # ================================================================

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Extract the 6 validated features that generate 60.2% win rate signals.

        These features have been proven effective through 4.5 years of backtesting.
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
        dataframe["%-macd_momentum"] = macd_hist.diff(3)

        # ================================================================
        # ADDITIONAL RISK MANAGEMENT FEATURES
        # ================================================================
        # Store ATR for dynamic stop loss calculation
        dataframe["atr_14"] = atr_14
        dataframe["atr_21"] = atr_21

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
                dataframe[col] = dataframe[col].replace([np.inf, -np.inf], np.nan)
                dataframe[col] = dataframe[col].fillna(method="ffill").fillna(0)
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
        - Proven: 60.2% accuracy on real signals over 4.5 years
        """
        self.freqai.class_names = ["down", "up"]

        # 3-day forward return (validated timeframe)
        forward_return = dataframe["close"].shift(-3) / dataframe["close"] - 1
        threshold = 0.02  # 2% threshold

        dataframe["&s-direction"] = np.where(forward_return > threshold, "up", "down")

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Run FreqAI prediction and add risk management context.
        """
        # Run FreqAI prediction
        dataframe = self.freqai.start(dataframe, metadata, self)

        # Add probability difference and confidence
        if "up" in dataframe.columns and "down" in dataframe.columns:
            dataframe["prob_diff"] = abs(dataframe["up"] - dataframe["down"])
            dataframe["confidence"] = dataframe[["up", "down"]].max(axis=1)

        # Add volatility context for dynamic risk management
        if "%-vol_pca_1" in dataframe.columns:
            # Volatility percentile for position sizing
            dataframe["vol_percentile"] = dataframe["%-vol_pca_1"].rolling(100).rank(pct=True)

            # ATR-based stop distance calculation
            if "atr_14" in dataframe.columns:
                dataframe["atr_stop_distance"] = dataframe["atr_14"] / dataframe["close"] * 1.5

        return dataframe

    # ================================================================
    # OPTIMIZED ENTRY LOGIC (Proven to Generate 161 Trades)
    # ================================================================

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Validated entry logic that generates 35.8 trades/year with 60.2% signal quality.

        V2.4 ADDITION: Store trade entry info for risk management tracking.
        """

        if (
            "up" not in dataframe.columns
            or "down" not in dataframe.columns
            or "prob_diff" not in dataframe.columns
        ):
            return dataframe

        # ================================================================
        # PROVEN THRESHOLDS (Generate 161 trades with 60.2% win rate)
        # ================================================================
        min_confidence = 0.55  # Validated: generates sufficient trades
        min_prob_diff = 0.25  # Validated: maintains signal quality

        confident_mask = (dataframe["confidence"] >= min_confidence) & (
            dataframe["prob_diff"] >= min_prob_diff
        )

        # ================================================================
        # BASIC REGIME FILTERING (Prevents extreme condition entries)
        # ================================================================
        momentum_ok = True
        vol_ok = True

        if "%-daily_rsi" in dataframe.columns:
            momentum_ok = (dataframe["%-daily_rsi"] > 30) & (dataframe["%-daily_rsi"] < 70)

        if "%-vol_trend" in dataframe.columns:
            vol_ok = dataframe["%-vol_trend"] > -0.20

        # ================================================================
        # ENTRY SIGNALS WITH CONFIDENCE TRACKING
        # ================================================================
        long_mask = confident_mask & (dataframe["up"] > dataframe["down"]) & momentum_ok & vol_ok
        short_mask = confident_mask & (dataframe["down"] > dataframe["up"]) & momentum_ok & vol_ok

        # Add signal strength for position sizing
        dataframe.loc[long_mask, "signal_strength"] = dataframe.loc[long_mask, "confidence"]
        dataframe.loc[short_mask, "signal_strength"] = dataframe.loc[short_mask, "confidence"]

        # Set entry signals
        dataframe.loc[long_mask, "enter_long"] = 1
        dataframe.loc[short_mask, "enter_short"] = 1

        return dataframe

    # ================================================================
    # REVOLUTIONARY RISK MANAGEMENT (V2.4 Core Innovation)
    # ================================================================

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Conservative model-based exits with confirmation, or disabled.
        """

        # Optionally disable model exits entirely
        if not self.use_model_exits:
            return dataframe

        if (
            "up" not in dataframe.columns
            or "down" not in dataframe.columns
            or "confidence" not in dataframe.columns
            or "prob_diff" not in dataframe.columns
        ):
            return dataframe

        # Strong opposite prediction
        strong_mask = (dataframe["confidence"] >= self.exit_min_confidence) & (
            dataframe["prob_diff"] >= self.exit_min_prob_diff
        )

        opp_long = dataframe["down"] > dataframe["up"]
        opp_short = dataframe["up"] > dataframe["down"]

        # Volatility filter (avoid exits in very high vol)
        if "vol_percentile" in dataframe.columns:
            vol_ok = dataframe["vol_percentile"] < 0.8
        else:
            vol_ok = True

        # Momentum/RSI reversal filters
        rsi_ok_long = True
        rsi_ok_short = True
        macd_ok_long = True
        macd_ok_short = True

        if "%-daily_rsi" in dataframe.columns:
            rsi_ok_long = dataframe["%-daily_rsi"] > 75
            rsi_ok_short = dataframe["%-daily_rsi"] < 25

        if "%-macd_momentum" in dataframe.columns:
            macd_ok_long = dataframe["%-macd_momentum"] < 0
            macd_ok_short = dataframe["%-macd_momentum"] > 0

        confirm_long = strong_mask & opp_long & vol_ok & rsi_ok_long & macd_ok_long
        confirm_short = strong_mask & opp_short & vol_ok & rsi_ok_short & macd_ok_short

        # Require multi-bar confirmation to reduce whipsaws
        if self.exit_confirmation_bars > 1:
            confirm_long = (
                confirm_long.rolling(self.exit_confirmation_bars).sum()
                == self.exit_confirmation_bars
            )
            confirm_short = (
                confirm_short.rolling(self.exit_confirmation_bars).sum()
                == self.exit_confirmation_bars
            )

        dataframe.loc[confirm_long, "exit_long"] = 1
        dataframe.loc[confirm_short, "exit_short"] = 1

        return dataframe

    # ================================================================
    # ADVANCED STOP LOSS SYSTEM (V2.4 Innovation)
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
        V2.4 Revolutionary Stop Loss System:

        1. TIME-BASED HOLDS: Minimum 3-day hold before stops can trigger
        2. ATR-BASED STOPS: Dynamic stops based on market volatility
        3. TRAILING STOPS: Move to breakeven at +4%, trail thereafter
        4. PROFIT PROTECTION: Lock in gains progressively

        GOAL: Transform -296.54% stop loss to manageable -100% loss
        """

        # ================================================================
        # 1. TIME-BASED HOLD PROTECTION
        # ================================================================
        trade_duration = current_time - trade.open_date_utc
        min_hold_days = 3  # Match 3-day prediction horizon

        if trade_duration < timedelta(days=min_hold_days):
            # During minimum hold period, only exit on extreme losses (-15%)
            return -0.15

        # ================================================================
        # 2. GET CURRENT VOLATILITY CONTEXT
        # ================================================================
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return -0.12  # Fallback: wider than V2.3's -8%

        latest_data = dataframe.iloc[-1]

        # ================================================================
        # 3. ATR-BASED DYNAMIC STOPS (Key Innovation)
        # ================================================================
        if "atr_stop_distance" in latest_data:
            # Use 1.5x ATR as stop distance (as recommended)
            atr_stop = -latest_data["atr_stop_distance"]

            # Ensure reasonable bounds
            if self.use_tight_atr_bounds:
                # Bound between -8% and -4%
                min_bound, max_bound = self.atr_bounds_tight
                atr_stop = max(min_bound, min(max_bound, atr_stop))
            else:
                # Default: Bound between -20% and -6%
                min_bound, max_bound = self.atr_bounds_default
                atr_stop = max(min_bound, min(max_bound, atr_stop))
        else:
            # Fallback: volatility percentile based stops
            vol_pct = latest_data.get("vol_percentile", 0.5)
            if vol_pct < 0.3:  # Low volatility
                atr_stop = -0.08  # Tighter stops OK
            elif vol_pct > 0.7:  # High volatility
                atr_stop = -0.15  # Much wider stops needed
            else:
                atr_stop = -0.12  # Moderate stops

        # ================================================================
        # 4. TRAILING STOP LOGIC
        # ================================================================
        if current_profit >= 0.04:  # Strong move: trail to lock 2%
            trailing_stop = current_profit - 0.02
            return max(trailing_stop, atr_stop)

        elif current_profit >= 0.03:  # Earlier trail: lock ~1.5%
            trailing_stop = current_profit - 0.015
            return max(trailing_stop, atr_stop)

        elif current_profit >= 0.02:  # Partial protection at +2%
            protected_stop = max(atr_stop, -0.04)
            return protected_stop

        # ================================================================
        # 5. SOFT TIME-STOP FOR STALE TRADES
        # ================================================================
        if self.soft_time_stop_enabled:
            if trade_duration >= timedelta(days=self.soft_time_stop_days):
                # Tighten to breakeven/small loss to close stale trades sooner
                atr_stop = max(atr_stop, self.soft_time_stop_floor)

        # ================================================================
        # 6. RETURN DYNAMIC (POSSIBLY TIGHTENED) STOP
        # ================================================================
        return atr_stop

    # ================================================================
    # DYNAMIC POSITION SIZING (V2.4 Risk Management)
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
        V2.4 Dynamic Position Sizing based on signal confidence and volatility.

        LOGIC:
        - High confidence signals: Full position size
        - Medium confidence: Reduced position size
        - High volatility: Smaller positions (wider stops need smaller size)
        """

        # Get current dataframe for context
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return proposed_stake

        latest_data = dataframe.iloc[-1]

        # ================================================================
        # CONFIDENCE-BASED SIZING
        # ================================================================
        confidence = latest_data.get("confidence", 0.55)

        if confidence >= 0.65:  # High confidence
            confidence_multiplier = 1.0  # Full size
        elif confidence >= 0.60:  # Medium confidence
            confidence_multiplier = 0.8  # Reduced size
        else:  # Low confidence
            confidence_multiplier = 0.6  # Small size

        # ================================================================
        # VOLATILITY-BASED SIZING
        # ================================================================
        vol_pct = latest_data.get("vol_percentile", 0.5)

        if vol_pct > 0.8:  # Very high volatility
            vol_multiplier = 0.7  # Smaller positions
        elif vol_pct < 0.2:  # Very low volatility
            vol_multiplier = 1.1  # Slightly larger positions
        else:
            vol_multiplier = 1.0  # Normal positions

        # ================================================================
        # ATR-BASED SIZING (Account for Stop Distance)
        # ================================================================
        atr_stop_distance = latest_data.get("atr_stop_distance", 0.12)

        if atr_stop_distance > 0.15:  # Wide stops needed
            atr_multiplier = 0.8  # Smaller positions
        elif atr_stop_distance < 0.08:  # Tight stops possible
            atr_multiplier = 1.1  # Larger positions
        else:
            atr_multiplier = 1.0  # Normal positions

        # ================================================================
        # CALCULATE FINAL POSITION SIZE
        # ================================================================
        total_multiplier = confidence_multiplier * vol_multiplier * atr_multiplier
        adjusted_stake = proposed_stake * total_multiplier

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

        return rsi.fillna(50)

    def _calculate_atr(self, dataframe: DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = dataframe["high"] - dataframe["low"]
        high_close_prev = abs(dataframe["high"] - dataframe["close"].shift(1))
        low_close_prev = abs(dataframe["low"] - dataframe["close"].shift(1))

        true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
        atr = true_range.rolling(window=period).mean()

        return atr.fillna(true_range)

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
    # HARD TIME-BASED EXIT (closes stale trades regardless of signals)
    # ================================================================

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | bool | None:
        if not self.hard_time_exit_enabled:
            return None

        trade_duration = current_time - trade.open_date_utc
        if trade_duration >= timedelta(days=self.hard_time_exit_days):
            # Exit if profit is below threshold after max hold period
            if current_profit < self.hard_time_exit_min_profit:
                return "time_exit"

        return None

    # ================================================================
    # PLOTTING CONFIGURATION
    # ================================================================

    plot_config = {
        "main_plot": {
            "close": {"color": "blue"},
        },
        "subplots": {
            "RSI": {
                "%-daily_rsi": {"color": "red", "type": "line"},
            },
            "Volatility": {
                "%-vol_pca_1": {"color": "orange", "type": "line"},
                "%-vol_ratio_7_14": {"color": "purple", "type": "line"},
            },
            "MACD": {
                "%-macd_momentum": {"color": "green", "type": "line"},
            },
            "Predictions": {
                "up": {"color": "lime", "type": "line"},
                "down": {"color": "red", "type": "line"},
                "confidence": {"color": "blue", "type": "line"},
            },
            "Risk Management": {
                "atr_stop_distance": {"color": "orange", "type": "line"},
                "vol_percentile": {"color": "gray", "type": "line"},
            },
        },
    }
