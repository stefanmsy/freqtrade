import logging
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from freqtrade.strategy import IStrategy, DecimalParameter
from freqtrade.freqai.prediction_models.LightGBMClassifier import LightGBMClassifier
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class BTCPriceDirectionStrategyV2_2(IStrategy):
    """
    Volatility-Focused BTC Price Direction Strategy V2.2

    Based on comprehensive V2.1 analysis showing volatility features are the clear winners:
    - Volatility group: 58.8% accuracy (best performing)
    - Momentum group: 58.3% accuracy (second best)
    - Lag group: 56.8% accuracy (underperformed)
    - Volume group: 55.3% accuracy (weakest)

    V2.2 Strategy: "LESS IS MORE"
    - 7 high-quality features vs V2.1's 16 noisy features
    - 5 core volatility features + 2 selective momentum features
    - Removed: lag features, volume features, redundant indicators
    - Focus: Multi-scale volatility dynamics with minimal momentum support

    Expected improvement: Volatility group alone nearly matched full V2.1 performance,
    suggesting focused approach should outperform feature-bloated versions.
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
    stoploss = -0.10  # Slightly tighter for focused strategy

    use_exit_signal = True
    startup_candle_count: int = 75  # Reduced from 100 - no lag features needed
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
            "v2_2_volatility": {
                "%-vol_pca_1": {"color": "purple"},
                "%-vol_regime_cluster": {"color": "red"},
                "%-atr_momentum_3d": {"color": "orange"},
            },
            "v2_2_momentum": {
                "%-daily_rsi": {"color": "blue"},
                "%-macd_momentum": {"color": "green"},
            },
        },
    }

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Build V2.2 volatility-focused feature set.
        Only 7 high-quality features: 5 volatility + 2 momentum.
        """

        # ================================================================
        # CORE VOLATILITY FEATURES (5 features - THE FOUNDATION)
        # ================================================================

        # Calculate multi-scale ATRs for volatility analysis
        atr_7 = self._calculate_atr(dataframe, 7)
        atr_14 = self._calculate_atr(dataframe, 14)
        atr_21 = self._calculate_atr(dataframe, 21)

        # Normalized ATRs (percent of price)
        atr_7_pct = atr_7 / dataframe["close"]
        atr_14_pct = atr_14 / dataframe["close"]
        atr_21_pct = atr_21 / dataframe["close"]

        # 1. PRIMARY VOLATILITY FACTOR (PCA compression of multi-scale ATRs)
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler

            # Prepare ATR data for PCA
            atr_features = (
                pd.DataFrame(
                    {"atr_7_pct": atr_7_pct, "atr_14_pct": atr_14_pct, "atr_21_pct": atr_21_pct}
                )
                .fillna(method="ffill")
                .fillna(0)
            )

            if len(atr_features.dropna()) > 50:  # Enough data for PCA
                scaler = StandardScaler()
                atr_scaled = scaler.fit_transform(atr_features)

                pca = PCA(n_components=1)  # Only need first component
                atr_pca = pca.fit_transform(atr_scaled)

                dataframe["%-vol_pca_1"] = atr_pca[:, 0]  # Primary volatility factor

                logger.info(
                    f"Volatility PCA explained variance: {pca.explained_variance_ratio_[0]:.3f}"
                )
            else:
                # Fallback if insufficient data
                dataframe["%-vol_pca_1"] = atr_14_pct

        except ImportError:
            # Fallback if sklearn not available
            dataframe["%-vol_pca_1"] = atr_14_pct

        # 2. VOLATILITY RATIO (short-term vs medium-term volatility relationship)
        dataframe["%-vol_ratio_7_14"] = atr_7 / (atr_14 + 1e-8)

        # 3. ATR MOMENTUM (volatility acceleration/deceleration over 3 days)
        dataframe["%-atr_momentum_3d"] = atr_14.diff(3) / (atr_14.shift(3) + 1e-8)

        # 4. VOLATILITY CLUSTERING REGIME (high volatility periods)
        vol_regime_ratio = atr_14 / atr_14.rolling(50).mean()
        dataframe["%-vol_regime_cluster"] = np.where(vol_regime_ratio > 1.2, 1, 0)

        # 5. VOLATILITY TREND (expanding vs contracting volatility)
        dataframe["%-vol_trend"] = (atr_14.rolling(5).mean() / atr_14.rolling(20).mean()) - 1

        # ================================================================
        # SELECTIVE MOMENTUM FEATURES (2 features - MINIMAL SUPPORT)
        # ================================================================

        # 6. DAILY RSI (interpretable momentum indicator)
        dataframe["%-daily_rsi"] = self._calculate_rsi(dataframe, 14)

        # 7. MACD MOMENTUM (momentum divergence via histogram slope)
        macd_data = self._calculate_macd(dataframe)
        dataframe["%-macd_momentum"] = macd_data["histogram"].diff(3)  # 3-day slope

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

            # Clip extreme values (winsorize at 2% and 98% for more aggressive outlier removal)
            if dataframe[col].dtype in ["float64", "float32"]:
                q_low = dataframe[col].quantile(0.02)
                q_high = dataframe[col].quantile(0.98)
                dataframe[col] = dataframe[col].clip(lower=q_low, upper=q_high)

        # Log feature summary for monitoring
        logger.info(f"V2.2 Volatility-Focused Strategy: {len(feature_columns)} total features")
        logger.info(f"Feature columns: {feature_columns}")
        logger.info("✅ V2.2: Focused on volatility dynamics with minimal momentum support")

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        Create target variable for 3-day direction prediction.
        Back to 2% threshold (V2.0 level) since 2.5% may have been too conservative.
        """

        # Set class names for 2-class classification
        self.freqai.class_names = ["down", "up"]

        # Calculate 3-day forward return
        forward_return = dataframe["close"].shift(-3) / dataframe["close"] - 1

        # Use 2% threshold (back to V2.0 level for better balance)
        threshold = 0.02  # 2% threshold
        dataframe["&s-direction"] = np.where(
            forward_return > threshold,
            "up",
            "down",
        )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Required method for FreqTrade strategy.
        Simple implementation - no lag features needed.
        """
        # Get FreqAI predictions (no additional processing needed)
        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Simplified entry logic focused on volatility-based signals.
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

            # Balanced confidence thresholds (not too high, not too low)
            min_confidence = 0.60  # Reduced from V2.1's 0.70
            min_prob_diff = 0.25  # Reduced from V2.1's 0.35

            # Base prediction mask
            confident_mask = (
                (dataframe["do_predict"] == 1)
                & (dataframe["max_prob"] > min_confidence)
                & (dataframe["prob_diff"] > min_prob_diff)
            )

            # Volatility-based regime filtering (check if columns exist)
            stable_vol_regime = True  # Default fallback
            expanding_vol = True  # Default fallback
            strong_momentum = True  # Default fallback
            weak_momentum = False  # Default fallback

            if "%-vol_regime_cluster" in dataframe.columns:
                stable_vol_regime = (
                    dataframe["%-vol_regime_cluster"] == 0
                )  # Not in high vol cluster
            if "%-vol_trend" in dataframe.columns:
                expanding_vol = dataframe["%-vol_trend"] > 0  # Volatility expanding
            if "%-daily_rsi" in dataframe.columns:
                strong_momentum = dataframe["%-daily_rsi"] > 55
                weak_momentum = dataframe["%-daily_rsi"] < 45

            # Long entry: Strong bullish prediction + favorable volatility conditions
            long_mask = (
                confident_mask
                & (dataframe["up"] > dataframe["down"])
                & (
                    stable_vol_regime | strong_momentum
                )  # Either stable volatility OR strong momentum
            )

            # Add volatility contraction filter if available
            if "%-vol_trend" in dataframe.columns:
                long_mask = long_mask & (dataframe["%-vol_trend"] > -0.15)

            # Short entry: Strong bearish prediction + bearish conditions
            short_mask = (
                confident_mask
                & (dataframe["down"] > dataframe["up"])
                & weak_momentum  # Weak momentum only
            )

            # Add volatility regime filter if available
            if "%-vol_regime_cluster" in dataframe.columns:
                short_mask = short_mask & (expanding_vol | (dataframe["%-vol_regime_cluster"] == 1))

            dataframe.loc[long_mask, "enter_long"] = 1
            dataframe.loc[short_mask, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Clean exit logic based on signal reversal and volatility spikes.
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
            min_confidence = 0.60
            min_prob_diff = 0.25

            confident_mask = (
                (dataframe["do_predict"] == 1)
                & (dataframe[["up", "down"]].max(axis=1) > min_confidence)
                & (abs(dataframe["up"] - dataframe["down"]) > min_prob_diff)
            )

            # Volatility spike exit condition (check if columns exist)
            vol_spike = False  # Default fallback
            momentum_extreme = False  # Default fallback

            if "%-vol_regime_cluster" in dataframe.columns:
                vol_spike = dataframe["%-vol_regime_cluster"] == 1  # High volatility cluster
            if "%-daily_rsi" in dataframe.columns:
                momentum_extreme = (dataframe["%-daily_rsi"] > 75) | (dataframe["%-daily_rsi"] < 25)

            # Exit long when strong bearish prediction OR volatility spike with extreme RSI
            exit_long_mask = confident_mask & (dataframe["down"] > dataframe["up"])
            if "%-vol_regime_cluster" in dataframe.columns and "%-daily_rsi" in dataframe.columns:
                exit_long_mask = exit_long_mask | (vol_spike & (dataframe["%-daily_rsi"] > 75))

            # Exit short when strong bullish prediction OR volatility spike with extreme RSI
            exit_short_mask = confident_mask & (dataframe["up"] > dataframe["down"])
            if "%-vol_regime_cluster" in dataframe.columns and "%-daily_rsi" in dataframe.columns:
                exit_short_mask = exit_short_mask | (vol_spike & (dataframe["%-daily_rsi"] < 25))

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
        Volatility-focused custom stoploss using primary volatility factor.
        """
        # Get the dataframe for this pair to access volatility features
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe is not None and len(dataframe) > 0:
            # Use primary volatility factor for dynamic stoploss
            current_vol = (
                dataframe["%-vol_pca_1"].iloc[-1] if "%-vol_pca_1" in dataframe.columns else 0.02
            )

            # Volatility regime for stoploss adjustment
            vol_regime = (
                dataframe["%-vol_regime_cluster"].iloc[-1]
                if "%-vol_regime_cluster" in dataframe.columns
                else 0
            )

            # Dynamic stoploss based on volatility
            if vol_regime == 1:  # High volatility regime
                vol_scaled_sl = -2.5 * abs(current_vol)  # Wider stops in high vol
            else:  # Normal volatility regime
                vol_scaled_sl = -1.8 * abs(current_vol)  # Tighter stops in normal vol

            # Ensure reasonable bounds
            vol_scaled_sl = max(vol_scaled_sl, -0.20)  # Max 20% stoploss
            vol_scaled_sl = min(vol_scaled_sl, -0.06)  # Min 6% stoploss
        else:
            # Fallback to base stoploss
            vol_scaled_sl = -0.10

        # Simple profit-based tightening
        if current_profit > 0.12:  # If profit > 12%
            return max(vol_scaled_sl, -0.06)  # Tight stoploss at 6%
        elif current_profit > 0.08:  # If profit > 8%
            return max(vol_scaled_sl, -0.08)  # Medium stoploss at 8%
        else:
            return vol_scaled_sl  # Use volatility-scaled stoploss

    # ================================================================
    # VOLATILITY ANALYSIS METHODS
    # ================================================================

    def get_volatility_summary(self, dataframe: DataFrame) -> dict:
        """
        Get current volatility state summary for monitoring.
        """
        if len(dataframe) == 0:
            return {}

        latest = dataframe.iloc[-1]

        return {
            "primary_vol_factor": latest.get("%-vol_pca_1", 0),
            "vol_ratio_7_14": latest.get("%-vol_ratio_7_14", 0),
            "atr_momentum": latest.get("%-atr_momentum_3d", 0),
            "vol_regime": "HIGH" if latest.get("%-vol_regime_cluster", 0) == 1 else "NORMAL",
            "vol_trend": "EXPANDING" if latest.get("%-vol_trend", 0) > 0 else "CONTRACTING",
            "momentum_rsi": latest.get("%-daily_rsi", 50),
            "macd_momentum": latest.get("%-macd_momentum", 0),
        }
