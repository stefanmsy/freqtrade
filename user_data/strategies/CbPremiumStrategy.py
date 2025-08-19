import logging
from functools import reduce
import numpy as np  # Add this back

import talib.abstract as ta
import pandas as pd
from pandas import DataFrame
from technical import qtpylib

from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
from freqtrade.freqai.prediction_models.PyTorchMLPRegressor import PyTorchMLPRegressor

logger = logging.getLogger(__name__)


class CbPremiumStrategy(IStrategy):
    """
    Coinbase Premium Strategy using FreqAI
    This strategy uses close, volume, and Coinbase premium as features to predict price movements.

    IMPORTANT: This strategy requires Coinbase premium data to be available in the dataframe.
    The data should be included via custom data loading or informative pairs.
    """

    # Specify the PyTorch model
    freqai_model = PyTorchMLPRegressor

    # Set timeframe to match downloaded data
    timeframe = "1h"

    def informative_pairs(self):
        """
        Define additional, informative pair/interval combinations to be cached from the exchange.
        These pairs will be merged with the original dataframe before feature engineering.
        """
        pairs = []
        return pairs

    # No ROI - rely entirely on AI signal for exits
    minimal_roi = {
        "0": 0.99,  # Effectively disabled (99% profit would never happen)
    }

    plot_config = {
        "main_plot": {},
        "subplots": {
            "&-s_close": {"&-s_close": {"color": "blue"}},
            "do_predict": {
                "do_predict": {"color": "brown"},
            },
        },
    }

    process_only_new_candles = True
    # No stop loss - rely entirely on AI signal
    stoploss = -0.99  # Effectively disabled (99% loss would never happen)
    use_exit_signal = True
    startup_candle_count: int = 40
    can_short = False  # Disable short trading - cb_premium works for longs only

    # Disable trailing stop
    trailing_stop = False

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `indicator_periods_candles`, `include_timeframes`, `include_shifted_candles`, and
        `include_corr_pairs`. In other words, a single feature defined in this function
        will automatically expand to a total of
        `indicator_periods_candles` * `include_timeframes` * `include_shifted_candles` *
        `include_corr_pairs` numbers of features added to the model.

        All features must be prepended with `%` to be recognized by FreqAI internals.

        # :param dataframe: strategy dataframe which will receive the features
        # :param period: period of the indicator
        # :param metadata: metadata of current pair
        #"""
        # # Core technical indicators
        # dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=period)
        # dataframe["%-mfi-period"] = ta.MFI(dataframe, timeperiod=period)
        # dataframe["%-adx-period"] = ta.ADX(dataframe, timeperiod=period)
        # dataframe["%-sma-period"] = ta.SMA(dataframe, timeperiod=period)
        # dataframe["%-ema-period"] = ta.EMA(dataframe, timeperiod=period)

        # Volume indicators
        dataframe["%-relative_volume-period"] = (
            dataframe["volume"] / dataframe["volume"].rolling(period).mean()
        )

        # Add volume momentum
        dataframe["%-volume_momentum-period"] = (
            dataframe["volume"] - dataframe["volume"].shift(period)
        ) / dataframe["volume"].shift(period).replace(0, 1)

        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `include_timeframes`, `include_shifted_candles`, and `include_corr_pairs`.

        Features defined here will *not* be automatically duplicated on user defined
        `indicator_periods_candles`

        All features must be prepended with `%` to be recognized by FreqAI internals.

        :param dataframe: strategy dataframe which will receive the features
        :param metadata: metadata of current pair
        """
        # Percentage change in price
        # dataframe["%-pct-change"] = dataframe["close"].pct_change()

        # Raw features
        dataframe["%-raw_volume"] = dataframe["volume"]

        # Coinbase premium (if available in the data)
        # This feature expects cb_premium data to be merged into the dataframe
        # You can provide this data via custom data loading or informative pairs
        if "cb_premium" in dataframe.columns:
            dataframe["%-cb_premium"] = dataframe["cb_premium"]

            # Create z-score feature here so it's available for entry/exit logic
            cb_premium_mean = dataframe["cb_premium"].rolling(10, min_periods=1).mean()
            cb_premium_std = dataframe["cb_premium"].rolling(10, min_periods=1).std()
            dataframe["%-cb_premium_zscore_10"] = (
                dataframe["cb_premium"] - cb_premium_mean
            ) / cb_premium_std.replace(0, 1)

            # Fill NaN values
            dataframe["%-cb_premium_zscore_10"].fillna(0, inplace=True)
        else:
            dataframe["%-cb_premium"] = 0.0
            dataframe["%-cb_premium_zscore_10"] = 0.0

        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Enhanced preprocessing for cb_premium to address autocorrelation and non-stationarity
        """
        # Time-based features
        # dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour

        # Add Coinbase Premium as a feature (must be prefixed with %)
        if "cb_premium" in dataframe.columns:
            # 1. Raw cb_premium (keep original for comparison)
            dataframe["%-cb_premium"] = dataframe["cb_premium"]

            # 2. First-order differencing (simplified)
            # dataframe["%-cb_premium_diff"] = dataframe["cb_premium"].diff()

            # 3. Rolling statistics on original series (more stable)
            dataframe["%-cb_premium_ma_10"] = dataframe["cb_premium"].rolling(10).mean()
            dataframe["%-cb_premium_ma_20"] = dataframe["cb_premium"].rolling(20).mean()
            dataframe["%-cb_premium_std_10"] = dataframe["cb_premium"].rolling(10).std()

            # 4. Z-score using rolling statistics (already created in expand_basic)
            if "%-cb_premium_zscore_10" not in dataframe.columns:
                dataframe["%-cb_premium_zscore_10"] = (
                    dataframe["cb_premium"] - dataframe["%-cb_premium_ma_10"]
                ) / dataframe["%-cb_premium_std_10"].replace(0, 1)

            # 5. Simplified lagged features (fewer lags)
            for lag in [1, 2, 6, 12]:  # Reduced number of lags
                dataframe[f"%-cb_premium_lag_{lag}"] = dataframe["cb_premium"].shift(lag)

            # 6. Momentum features
            dataframe["%-cb_premium_momentum_10"] = dataframe["cb_premium"] - dataframe[
                "cb_premium"
            ].shift(10)
            dataframe["%-cb_premium_momentum_20"] = dataframe["cb_premium"] - dataframe[
                "cb_premium"
            ].shift(20)

            # 7. Volatility features
            dataframe["%-cb_premium_volatility_10"] = dataframe["cb_premium"].rolling(10).std()
            dataframe["%-cb_premium_volatility_20"] = dataframe["cb_premium"].rolling(20).std()

            # 8. Percentile features (more robust)
            dataframe["%-cb_premium_percentile_25"] = (
                dataframe["cb_premium"].rolling(20).quantile(0.25)
            )
            dataframe["%-cb_premium_percentile_75"] = (
                dataframe["cb_premium"].rolling(20).quantile(0.75)
            )
            dataframe["%-cb_premium_percentile_10"] = (
                dataframe["cb_premium"].rolling(20).quantile(0.10)
            )
            dataframe["%-cb_premium_percentile_90"] = (
                dataframe["cb_premium"].rolling(20).quantile(0.90)
            )

            # 9. Fill NaN values more carefully
            cb_premium_features = [
                col for col in dataframe.columns if col.startswith("%-cb_premium")
            ]
            for col in cb_premium_features:
                # Replace infinite values first
                dataframe[col] = dataframe[col].replace([np.inf, -np.inf], np.nan)
                # Forward fill first, then backward fill, then zero
                dataframe[col].fillna(method="ffill", inplace=True)
                dataframe[col].fillna(method="bfill", inplace=True)
                dataframe[col].fillna(0, inplace=True)

        else:
            # Fallback if cb_premium is not available
            dataframe["%-cb_premium"] = 0.0
            dataframe["%-cb_premium_diff"] = 0.0
            dataframe["%-cb_premium_ma_10"] = 0.0
            dataframe["%-cb_premium_std_10"] = 0.0
            dataframe["%-cb_premium_zscore_10"] = 0.0

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        Required function to set the targets for the model.
        All targets must be prepended with `&` to be recognized by the FreqAI internals.

        :param dataframe: strategy dataframe which will receive the targets
        :param metadata: metadata of current pair
        """
        # Our target is the future price change over the label period
        dataframe["&-s_close"] = (
            dataframe["close"]
            .shift(-self.freqai_info["feature_parameters"]["label_period_candles"])
            .rolling(self.freqai_info["feature_parameters"]["label_period_candles"])
            .mean()
            / dataframe["close"]
            - 1
        )

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # All indicators must be populated by feature_engineering_*() functions

        # Load Coinbase premium data directly and align with trading data timestamps
        try:
            import os

            # Try 1H data first, then fallback to 1D
            cb_premium_paths = [
                "user_data/data/cb_premium/cb_premium_1H.csv",
                "user_data/data/cb_premium/cb_premium_1D.csv",
            ]

            cb_data = None
            for path in cb_premium_paths:
                if os.path.exists(path):
                    cb_data = pd.read_csv(path)
                    cb_data["Open time"] = pd.to_datetime(cb_data["Open time"])
                    logger.info(f"Loaded Coinbase premium data from {path}")
                    break

            if cb_data is not None:
                # Create a temporary merged dataframe to ensure proper timestamp alignment
                temp_df = dataframe[["date"]].copy()
                temp_df = temp_df.merge(
                    cb_data[["Open time", "cb_premium"]],
                    left_on="date",
                    right_on="Open time",
                    how="left",
                )
                dataframe["cb_premium"] = temp_df["cb_premium"]

                logger.info(f"Merged Coinbase premium data for {metadata['pair']}")
                logger.info(
                    f"Coinbase premium data coverage: {dataframe['cb_premium'].count()}/{len(dataframe)}"
                )
            else:
                logger.warning(f"Coinbase premium data not found in any of the expected paths")
                dataframe["cb_premium"] = 0.0
        except Exception as e:
            logger.error(f"Error loading Coinbase premium data: {e}")
            dataframe["cb_premium"] = 0.0

        # The model will return all labels created by user in `set_freqai_targets()`
        # (& appended targets), an indication of whether or not the prediction should be accepted,
        # the target mean/std values for each of the labels created by user in
        # `set_freqai_targets()` for each training period.

        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # More selective long entry conditions - only trade strong signals
        enter_long_conditions = [
            df["do_predict"] == 1,
            df["&-s_close"] > 0.015,  # Increased from 0.01 to 0.015 for stronger signals
        ]

        # Add volume confirmation
        if "%-relative_volume-period" in df.columns:
            enter_long_conditions.append(
                df["%-relative_volume-period"] > 1.5  # Increased from 1.1 to 1.5
            )

        # Safely check if cb_premium_zscore_10 exists and add condition
        if "%-cb_premium_zscore_10" in df.columns:
            enter_long_conditions.append(
                df["%-cb_premium_zscore_10"].abs() > 0.8  # Increased from 0.4 to 0.8
            )

        if enter_long_conditions:
            combined_condition = enter_long_conditions[0]
            for condition in enter_long_conditions[1:]:
                combined_condition = combined_condition & condition

            df.loc[combined_condition, "enter_long"] = 1

        # No short trading - cb_premium works for longs only
        df["enter_short"] = 0

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # More conservative exit conditions
        exit_long_conditions = [
            df["do_predict"] == 1,
            (df["&-s_close"] < -0.008),  # Reduced from -0.01 to -0.008 for faster exit
        ]

        # Safely check if cb_premium_zscore_10 exists and add condition
        if "%-cb_premium_zscore_10" in df.columns:
            exit_long_conditions.append(
                df["%-cb_premium_zscore_10"] < -1.2
            )  # Reduced from -1.5 to -1.2

        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        # Short exit conditions
        exit_short_conditions = [
            df["do_predict"] == 1,
            (df["&-s_close"] > 0.008),  # Reduced from 0.01 to 0.008 for faster exit
        ]

        # Safely check if cb_premium_zscore_10 exists and add condition
        if "%-cb_premium_zscore_10" in df.columns:
            exit_short_conditions.append(
                df["%-cb_premium_zscore_10"] > 1.2
            )  # Reduced from 1.5 to 1.2

        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return df
