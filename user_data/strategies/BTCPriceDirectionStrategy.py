import logging
import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from freqtrade.strategy import IStrategy, DecimalParameter
from freqtrade.freqai.prediction_models.LightGBMClassifier import LightGBMClassifier
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class BTCPriceDirectionStrategy(IStrategy):
    """
    Simple BTC Price Direction Prediction Strategy using FreqAI
    This strategy uses only price-based technical indicators to predict
    BTC price direction (up/down) for the next 3 days.

    Key Features:
    - Classification model (direction prediction)
    - Minimal price-based features only
    - 3-day forward prediction horizon
    - Daily timeframe for swing trading
    - Simple, clean implementation
    """

    # Use LightGBM Classifier for direction prediction
    freqai_model = LightGBMClassifier

    # Daily timeframe for swing trading
    timeframe = "1d"

    def informative_pairs(self):
        """Define additional informative pairs if needed."""
        return []

    # No ROI - rely entirely on AI signal for exits
    minimal_roi = {
        "0": 0.99,  # Effectively disabled
    }

    # Use custom stoploss for risk management
    stoploss = -0.15  # 15% base stoploss

    use_exit_signal = True
    startup_candle_count: int = 50  # Enough for features + 3d label
    can_short = True  # Enable shorting for futures trading

    # Disable trailing stop
    trailing_stop = False
    process_only_new_candles = True

    # Direction classification threshold (can be optimized via hyperopt)
    direction_threshold = DecimalParameter(0.005, 0.05, default=0.01, space="buy", optimize=True)

    plot_config = {
        "main_plot": {
            "close": {"color": "blue"},
        },
        "subplots": {
            "&-target_3d": {"&-target_3d": {"color": "green"}},
            "do_predict": {"do_predict": {"color": "orange"}},
            "technical_indicators": {
                "%-rsi_14": {"color": "purple"},
                "%-macd": {"color": "red"},
                "%-bb_upper": {"color": "gray"},
                "%-bb_lower": {"color": "gray"},
            },
        },
    }

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Build minimal price-based features for direction prediction.
        Focus on proven technical indicators with minimal noise.
        Enhanced with comprehensive volatility indicators.
        """

        # 1. RSI (Relative Strength Index) - momentum indicator
        dataframe["%-rsi_14"] = self.rsi(dataframe, 14)

        # 2. MACD (Moving Average Convergence Divergence) - trend indicator
        macd = self.macd(dataframe)
        dataframe["%-macd"] = macd["macd"]
        dataframe["%-macd_signal"] = macd["macdsignal"]
        dataframe["%-macd_histogram"] = macd["macdhist"]

        # 3. Bollinger Bands - volatility indicator
        bb = self.bb(dataframe, 20, 2)
        dataframe["%-bb_upper"] = bb["upper"]
        dataframe["%-bb_lower"] = bb["lower"]
        dataframe["%-bb_percent"] = bb["percent"]

        # 4. Moving Averages - trend indicators
        dataframe["%-sma_20"] = self.sma(dataframe, 20)
        dataframe["%-sma_50"] = self.sma(dataframe, 50)
        dataframe["%-ema_12"] = self.ema(dataframe, 12)
        dataframe["%-ema_26"] = self.ema(dataframe, 26)

        # 5. Price-based ratios and differences
        dataframe["%-price_sma20_ratio"] = dataframe["close"] / dataframe["%-sma_20"]
        dataframe["%-price_sma50_ratio"] = dataframe["close"] / dataframe["%-sma_50"]
        dataframe["%-sma20_sma50_ratio"] = dataframe["%-sma_20"] / dataframe["%-sma_50"]

        # 6. Enhanced Volatility Indicators
        # ATR (Average True Range) - multiple periods
        dataframe["%-atr_14"] = self.atr(dataframe, 14)
        dataframe["%-atr_21"] = self.atr(dataframe, 21)
        dataframe["%-atr_percent_14"] = dataframe["%-atr_14"] / dataframe["close"]
        dataframe["%-atr_percent_21"] = dataframe["%-atr_21"] / dataframe["close"]

        # ATR ratio (short vs long term volatility)
        dataframe["%-atr_ratio"] = dataframe["%-atr_14"] / dataframe["%-atr_21"]

        # Historical Volatility (rolling standard deviation of returns)
        dataframe["%-returns"] = dataframe["close"].pct_change()
        dataframe["%-volatility_14"] = dataframe["%-returns"].rolling(14).std()
        dataframe["%-volatility_21"] = dataframe["%-returns"].rolling(21).std()
        dataframe["%-volatility_ratio"] = (
            dataframe["%-volatility_14"] / dataframe["%-volatility_21"]
        )

        # Realized Volatility (sum of squared returns)
        dataframe["%-realized_vol_14"] = (dataframe["%-returns"] ** 2).rolling(14).sum()
        dataframe["%-realized_vol_21"] = (dataframe["%-returns"] ** 2).rolling(21).sum()

        # Parkinson Volatility (using high-low range)
        dataframe["%-parkinson_vol_14"] = np.sqrt(
            (1 / (4 * np.log(2)))
            * ((np.log(dataframe["high"] / dataframe["low"]) ** 2).rolling(14).mean())
        )

        # Garman-Klass Volatility (more efficient than Parkinson)
        dataframe["%-gk_vol_14"] = np.sqrt(
            (
                0.5 * (np.log(dataframe["high"] / dataframe["low"]) ** 2)
                - (2 * np.log(2) - 1) * (np.log(dataframe["close"] / dataframe["open"]) ** 2)
            )
            .rolling(14)
            .mean()
        )

        # Volatility of Volatility (volatility clustering)
        dataframe["%-vol_of_vol_14"] = dataframe["%-volatility_14"].rolling(14).std()

        # Volatility Regime Indicators
        dataframe["%-vol_regime_14"] = np.where(
            dataframe["%-volatility_14"] > dataframe["%-volatility_14"].rolling(50).mean(),
            1,
            0,  # 1 for high volatility, 0 for low volatility
        )

        # Volatility Breakout Indicators
        dataframe["%-vol_breakout_14"] = np.where(
            dataframe["%-volatility_14"] > dataframe["%-volatility_14"].rolling(50).quantile(0.8),
            1,
            0,
        )

        # Enhanced Market Regime Features
        # VIX-like volatility indicator for crypto
        dataframe["%-crypto_vix"] = dataframe["%-volatility_14"].rolling(30).mean()

        # Market regime classification (bull/bear)
        dataframe["%-bull_bear_regime"] = np.where(
            dataframe["%-sma_20"] > dataframe["%-sma_50"], 1, 0
        )

        # Trend strength indicator
        dataframe["%-trend_strength"] = abs(dataframe["%-price_sma20_ratio"] - 1)

        # Volatility clustering indicator
        dataframe["%-vol_clustering"] = dataframe["%-volatility_14"].rolling(7).std()

        # Market momentum indicator
        dataframe["%-momentum_14"] = dataframe["close"].pct_change(14)

        # Price acceleration
        dataframe["%-acceleration"] = dataframe["%-momentum_14"].diff()

        # 7. Volume indicators (if available)
        if "volume" in dataframe.columns:
            dataframe["%-volume_sma_20"] = self.sma(dataframe, 20, "volume")
            dataframe["%-volume_ratio"] = dataframe["volume"] / dataframe["%-volume_sma_20"]

            # Volume-weighted volatility
            dataframe["%-vwap"] = (dataframe["close"] * dataframe["volume"]).rolling(
                20
            ).sum() / dataframe["volume"].rolling(20).sum()
            dataframe["%-price_vwap_ratio"] = dataframe["close"] / dataframe["%-vwap"]

        # 8. Price momentum
        dataframe["%-price_change_1d"] = dataframe["close"].pct_change(1)
        dataframe["%-price_change_3d"] = dataframe["close"].pct_change(3)
        dataframe["%-price_change_7d"] = dataframe["close"].pct_change(7)

        # 9. Enhanced High-Low range indicators
        dataframe["%-hl_range"] = (dataframe["high"] - dataframe["low"]) / dataframe["close"]
        dataframe["%-hl_range_sma_14"] = self.sma(dataframe, 14, "%-hl_range")
        dataframe["%-hl_range_ratio"] = dataframe["%-hl_range"] / dataframe["%-hl_range_sma_14"]

        # Body size (open-close range)
        dataframe["%-body_size"] = abs(dataframe["close"] - dataframe["open"]) / dataframe["close"]
        dataframe["%-body_size_sma_14"] = self.sma(dataframe, 14, "%-body_size")

        # Shadow ratios (upper and lower shadows)
        dataframe["%-upper_shadow"] = (
            dataframe["high"] - np.maximum(dataframe["open"], dataframe["close"])
        ) / dataframe["close"]
        dataframe["%-lower_shadow"] = (
            np.minimum(dataframe["open"], dataframe["close"]) - dataframe["low"]
        ) / dataframe["close"]

        # 10. Volatility-based momentum indicators
        # Volatility-adjusted RSI
        dataframe["%-vol_adjusted_rsi"] = dataframe["%-rsi_14"] * dataframe["%-volatility_14"]

        # Volatility-adjusted price momentum
        dataframe["%-vol_adjusted_momentum_3d"] = (
            dataframe["%-price_change_3d"] / dataframe["%-volatility_14"]
        )
        dataframe["%-vol_adjusted_momentum_7d"] = (
            dataframe["%-price_change_7d"] / dataframe["%-volatility_14"]
        )

        # Clean and clip features to reduce noise
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

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        Create target variable for 3-day direction prediction.
        For classification, we need to set class_names and use string labels.
        """

        # Set class names for 2-class classification (up/down only)
        self.freqai.class_names = ["down", "up"]

        # Calculate 3-day forward return
        forward_return = dataframe["close"].shift(-3) / dataframe["close"] - 1

        # Create 2-class classification target with string labels
        # Use higher threshold for clearer signals (1.5% instead of 1%)
        threshold = 0.015  # Increased from 0.01 to 0.015 for clearer signals
        dataframe["&s-direction"] = np.where(
            forward_return > threshold,
            "up",
            "down",
        )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Required method for FreqTrade strategy.
        This is called by FreqTrade to populate indicators.
        """
        # This is exactly how the working strategy does it
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry signal based on AI prediction with confidence threshold.
        """
        # Initialize entry signals
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        # Debug: Log available columns
        import logging

        logger = logging.getLogger(__name__)
        logger.info(
            f"Available columns: {[col for col in dataframe.columns if '&' in col or 'do_predict' in col or 'DI_' in col]}"
        )

        # Check for FreqAI prediction columns
        if (
            "do_predict" in dataframe.columns
            and "up" in dataframe.columns
            and "down" in dataframe.columns
        ):
            # Calculate max and second max probabilities (2-class only)
            dataframe["max_prob"] = dataframe[["up", "down"]].max(axis=1)
            dataframe["second_max_prob"] = dataframe[["up", "down"]].apply(
                lambda x: x.nlargest(2).iloc[1], axis=1
            )
            dataframe["prob_diff"] = dataframe["max_prob"] - dataframe["second_max_prob"]

            # Asymmetric thresholds by direction (v1.14 - more balanced)
            # Require higher confidence to short (avoid false bearish signals)
            short_conf_thresh = 0.55
            short_prob_thresh = 0.60

            # Allow much lower thresholds to catch more bullish moves
            long_conf_thresh = 0.35
            long_prob_thresh = 0.50

            # Only trade when confidence AND probability are above thresholds
            confident_mask = dataframe["do_predict"] == 1

            # Regime-based signal filtering (v1.14 - less restrictive)
            # Calculate SMA indicators for regime detection
            sma_20 = self.sma(dataframe, 20)
            sma_50 = self.sma(dataframe, 50)

            # Define market regime using SMA crossover with more flexibility
            bull_regime = sma_20 > sma_50
            bear_regime = ~bull_regime

            # Add trend strength filter
            trend_strength = abs(sma_20 - sma_50) / sma_50
            strong_trend = trend_strength > 0.02  # 2% difference

            # Long entry: up is highest AND confidence AND probability are high (lower thresholds) AND (bullish regime OR strong trend)
            long_mask = (
                confident_mask
                & (dataframe["prob_diff"] > long_conf_thresh)
                & (dataframe["max_prob"] > long_prob_thresh)
                & (dataframe["up"] > dataframe["down"])
                & (bull_regime | strong_trend)  # Long in bullish regime OR strong trend
            )

            # Short entry: down is highest AND confidence AND probability are high (higher thresholds) AND bearish regime
            short_mask = (
                confident_mask
                & (dataframe["prob_diff"] > short_conf_thresh)
                & (dataframe["max_prob"] > short_prob_thresh)
                & (dataframe["down"] > dataframe["up"])
                & bear_regime  # Only short in bearish regime
            )

            dataframe.loc[long_mask, "enter_long"] = 1
            dataframe.loc[short_mask, "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit signal based on AI prediction with confidence threshold.
        """
        # Initialize exit signals
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0

        # Use classification predictions for exit signals
        if (
            "do_predict" in dataframe.columns
            and "up" in dataframe.columns
            and "down" in dataframe.columns
        ):
            # Calculate max and second max probabilities (if not already calculated)
            if "prob_diff" not in dataframe.columns:
                dataframe["max_prob"] = dataframe[["up", "down"]].max(axis=1)
                dataframe["second_max_prob"] = dataframe[["up", "down"]].apply(
                    lambda x: x.nlargest(2).iloc[1], axis=1
                )
                dataframe["prob_diff"] = dataframe["max_prob"] - dataframe["second_max_prob"]

            # Asymmetric exit thresholds (v1.14 - updated to match entry)
            short_conf_thresh = 0.55
            short_prob_thresh = 0.60
            long_conf_thresh = 0.35
            long_prob_thresh = 0.50

            # Only exit when confidence AND probability are above thresholds
            confident_mask = dataframe["do_predict"] == 1

            # Exit long when AI predicts down with high confidence AND probability (higher thresholds for shorts)
            exit_long_mask = (
                confident_mask
                & (dataframe["prob_diff"] > short_conf_thresh)
                & (dataframe["max_prob"] > short_prob_thresh)
                & (dataframe["down"] > dataframe["up"])
            )

            # Exit short when AI predicts up with high confidence AND probability (lower thresholds for longs)
            exit_short_mask = (
                confident_mask
                & (dataframe["prob_diff"] > long_conf_thresh)
                & (dataframe["max_prob"] > long_prob_thresh)
                & (dataframe["up"] > dataframe["down"])
            )

            dataframe.loc[exit_long_mask, "exit_long"] = 1
            dataframe.loc[exit_short_mask, "exit_short"] = 1

        return dataframe

    # Technical indicator helper methods
    def rsi(self, dataframe: DataFrame, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = dataframe["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def macd(self, dataframe: DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> dict:
        """Calculate MACD."""
        exp1 = dataframe["close"].ewm(span=fast).mean()
        exp2 = dataframe["close"].ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return {"macd": macd_line, "macdsignal": signal_line, "macdhist": histogram}

    def bb(self, dataframe: DataFrame, period: int = 20, std: float = 2) -> dict:
        """Calculate Bollinger Bands."""
        sma = dataframe["close"].rolling(window=period).mean()
        std_dev = dataframe["close"].rolling(window=period).std()
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        percent = (dataframe["close"] - lower_band) / (upper_band - lower_band)
        return {"upper": upper_band, "lower": lower_band, "percent": percent}

    def sma(self, dataframe: DataFrame, period: int, column: str = "close") -> pd.Series:
        """Calculate Simple Moving Average."""
        return dataframe[column].rolling(window=period).mean()

    def ema(self, dataframe: DataFrame, period: int, column: str = "close") -> pd.Series:
        """Calculate Exponential Moving Average."""
        return dataframe[column].ewm(span=period).mean()

    def atr(self, dataframe: DataFrame, period: int = 14) -> pd.Series:
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
        Enhanced custom stoploss with volatility adjustment and profit-based tightening.
        """
        # Get the dataframe for this pair to access ATR
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe is not None and len(dataframe) > 0:
            # Get current ATR for volatility-based stoploss
            current_atr = dataframe["%-atr_14"].iloc[-1]

            # Calculate volatility-scaled stoploss (1.5x ATR distance)
            vol_scaled_sl = -1.5 * current_atr / current_rate

            # Ensure minimum and maximum bounds
            vol_scaled_sl = max(vol_scaled_sl, -0.25)  # Max 25% stoploss
            vol_scaled_sl = min(vol_scaled_sl, -0.05)  # Min 5% stoploss
        else:
            # Fallback to base stoploss if no data available
            vol_scaled_sl = -0.15

        # Profit-based tightening (v1.14 - less aggressive)
        if current_profit > 0.15:  # If profit > 15%
            return max(vol_scaled_sl, -0.05)  # Tight stoploss at 5%
        elif current_profit > 0.10:  # If profit > 10%
            return max(vol_scaled_sl, -0.08)  # Medium stoploss at 8%
        elif current_profit > 0.05:  # If profit > 5%
            return max(vol_scaled_sl, -0.12)  # Loose stoploss at 12%
        else:
            return vol_scaled_sl  # Use volatility-scaled stoploss
