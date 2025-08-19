import logging
from functools import reduce
import numpy as np
import os

import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy
from freqtrade.freqai.prediction_models.LightGBMRegressor import LightGBMRegressor

try:
    # Optional sklearn wrappers if configured
    from user_data.strategies.sklearn_regressors import (
        ElasticNetRegressor as SkElasticNetRegressor,
        LinearRegressionRegressor as SkLinearRegressionRegressor,
    )
except Exception:  # pragma: no cover
    SkElasticNetRegressor = None
    SkLinearRegressionRegressor = None

logger = logging.getLogger(__name__)


class GlassnodeSwingStrategy(IStrategy):
    """
    Glassnode On-Chain ML Swing Trading Strategy using FreqAI
    This strategy uses on-chain metrics from Glassnode combined with technical indicators
    to predict 5-day forward returns for swing trading.

    IMPORTANT: This strategy requires Glassnode data to be available in the data folder.
    The data should be in user_data/data/glassnode/ directory.
    """

    # Default model
    freqai_model = LightGBMRegressor

    # Set timeframe to daily to match on-chain cadence
    timeframe = "1d"

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
            "&-target_3d": {"&-target_3d": {"color": "blue"}},
            "do_predict": {"do_predict": {"color": "brown"}},
            "onchain_metrics": {
                "%-sopr_dz_7d": {"color": "orange"},
                "%-exchange_netflow_log_dz_7d": {"color": "green"},
                "%-percent_lth_dz_7d": {"color": "purple"},
            },
        },
    }

    process_only_new_candles = True
    # No stop loss - rely entirely on AI signal
    stoploss = -0.99  # Effectively disabled (99% loss would never happen)
    use_exit_signal = True
    # With 1d timeframe, a month is enough for features + 5d label
    startup_candle_count: int = 60
    can_short = False  # Disable shorts (shorts underperformed in latest runs)

    # Disable trailing stop
    trailing_stop = False

    def feature_engineering_expand_all(
        self, dataframe: DataFrame, period: int, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Keep this function minimal to avoid introducing non-glassnode features.
        Intentionally no extra features here.
        """
        return dataframe

    def feature_engineering_expand_basic(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        This function will automatically expand the defined features on the config defined
        `include_timeframes`, `include_shifted_candles`, and `include_corr_pairs`.

        All features must be prepended with `%` to be recognized by FreqAI internals.

        :param dataframe: strategy dataframe which will receive the features
        :param metadata: metadata of current pair
        """
        return dataframe

    def feature_engineering_standard(
        self, dataframe: DataFrame, metadata: dict, **kwargs
    ) -> DataFrame:
        """
        Build reduced on-chain features to avoid overfitting.
        Only selected metrics, windows, and feature types are produced.
        """

        def build_features_for_metric(
            df: DataFrame,
            series: pd.Series,
            prefix: str,
            allowed_windows: list[int],
            include_types: set[str],  # subset of {"dz", "dpct", "dvol"}
            include_vol_ratio: bool = True,
        ) -> dict[str, pd.Series]:
            s = series.copy()
            daily = (
                pd.DataFrame({"date": df["date"], "v": s})
                .dropna(subset=["v"])
                .set_index("date")
                .resample("1D")
                .ffill()
            )
            if daily.empty:
                return {}

            out: dict[str, pd.Series] = {}
            date_index = pd.DatetimeIndex(df["date"])

            # Always compute 7d and 30d vol to enable vol_ratio if requested
            need_vol_for_ratio = include_vol_ratio
            vol_cache: dict[int, pd.Series] = {}

            def map_back(series_to_map: pd.Series) -> pd.Series:
                cleaned = series_to_map.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
                mapped = cleaned.reindex(date_index, method="ffill")
                mapped.index = df.index
                return mapped

            def winsorize(s: pd.Series, low_q: float = 0.01, high_q: float = 0.99) -> pd.Series:
                try:
                    q_low = s.quantile(low_q)
                    q_high = s.quantile(high_q)
                    if pd.isna(q_low) or pd.isna(q_high):
                        return s
                    return s.clip(lower=q_low, upper=q_high)
                except Exception:
                    return s

            windows_to_compute = sorted(
                set(list(allowed_windows) + ([7, 30] if need_vol_for_ratio else []))
            )
            for window in windows_to_compute:
                mean_roll = daily["v"].rolling(window).mean()
                std_roll = daily["v"].rolling(window).std()

                if "dz" in include_types and window in allowed_windows:
                    z_score = (daily["v"] - mean_roll) / std_roll.replace(0, 1)
                    out[f"%-{prefix}_dz_{window}d"] = map_back(z_score)

                if "dpct" in include_types and window in allowed_windows:
                    pct_change = daily["v"].pct_change(window)
                    pct_change = winsorize(pct_change)
                    out[f"%-{prefix}_dpct_{window}d"] = map_back(pct_change)

                # compute volatility always if requested for ratio, or if dvol explicitly included
                if need_vol_for_ratio or ("dvol" in include_types and window in allowed_windows):
                    volatility = daily["v"].rolling(window).std() / mean_roll.replace(0, 1)
                    volatility = winsorize(volatility)
                    vol_cache[window] = volatility
                    if "dvol" in include_types and window in allowed_windows:
                        out[f"%-{prefix}_dvol_{window}d"] = map_back(volatility)

            if include_vol_ratio and 7 in vol_cache and 30 in vol_cache:
                vol_ratio = vol_cache[7] / vol_cache[30].replace(0, 1)
                vol_ratio = winsorize(vol_ratio)
                out[f"%-{prefix}_vol_ratio_7_30"] = map_back(vol_ratio)

            return out

        # Collect all features first to avoid DataFrame fragmentation
        all_features: dict[str, pd.Series] = {}

        # SOPR: windows 7,14; keep only z-scores; no vol ratio
        if "sopr" in dataframe.columns:
            all_features.update(
                build_features_for_metric(
                    dataframe, dataframe["sopr"], "sopr", [7, 14], {"dz"}, False
                )
            )

        # Exchange netflow (log transform), windows 7,14; keep only z-scores; no vol ratio
        if "exchange_netflow" in dataframe.columns:
            ex = dataframe["exchange_netflow"].copy()
            ex_log = np.sign(ex) * np.log1p(np.abs(ex))
            all_features.update(
                build_features_for_metric(
                    dataframe, ex_log, "exchange_netflow_log", [7, 14], {"dz"}, False
                )
            )

        # Realized PnL ratio, windows 7,14; keep only z-scores; no vol ratio
        if "realized_pnl_ratio" in dataframe.columns:
            all_features.update(
                build_features_for_metric(
                    dataframe,
                    dataframe["realized_pnl_ratio"],
                    "realized_pnl",
                    [7, 14],
                    {"dz"},
                    False,
                )
            )

        # NVT ratio and MVRV metrics (from full CSV) - z-scores only; no vol ratio
        if "nvt_ratio" in dataframe.columns:
            all_features.update(
                build_features_for_metric(
                    dataframe, dataframe["nvt_ratio"], "nvt_ratio", [7, 14], {"dz"}, False
                )
            )
        if "mvrv" in dataframe.columns:
            all_features.update(
                build_features_for_metric(
                    dataframe, dataframe["mvrv"], "mvrv", [7, 14], {"dz"}, False
                )
            )
        # Backwards compatibility if separate NVT price series exist - z-scores only
        if "nvt_price" in dataframe.columns:
            all_features.update(
                build_features_for_metric(
                    dataframe, dataframe["nvt_price"], "nvt_price", [7, 14], {"dz"}, False
                )
            )
        if "nvt_price_90D" in dataframe.columns:
            all_features.update(
                build_features_for_metric(
                    dataframe, dataframe["nvt_price_90D"], "nvt_price_90D", [7], {"dz"}, False
                )
            )

        # 90D NVT premium - removed per reduction plan (skip)

        # Whale netflow (log), windows 7,14; z-scores only; no vol ratio
        if "whale_exchange_netflow" in dataframe.columns:
            w = dataframe["whale_exchange_netflow"].copy()
            w_log = np.sign(w) * np.log1p(np.abs(w))
            all_features.update(
                build_features_for_metric(
                    dataframe, w_log, "whale_netflow_log", [7, 14], {"dz"}, False
                )
            )

        # Percent LTH/STH
        if (
            "long_term_holder_supply" in dataframe.columns
            and "short_term_holder_supply" in dataframe.columns
        ):
            total_supply = (
                dataframe["long_term_holder_supply"] + dataframe["short_term_holder_supply"]
            )
            percent_lth = dataframe["long_term_holder_supply"] / total_supply.replace(0, 1)
            percent_sth = dataframe["short_term_holder_supply"] / total_supply.replace(0, 1)
            all_features.update(
                build_features_for_metric(
                    dataframe, percent_lth, "percent_lth", [7, 14], {"dz"}, False
                )
            )
            all_features.update(
                build_features_for_metric(dataframe, percent_sth, "percent_sth", [7], {"dz"}, False)
            )

        if all_features:
            feature_df = pd.DataFrame(all_features, index=dataframe.index)
            dataframe = pd.concat([dataframe, feature_df], axis=1)

        # Calendar features (non-price): day-of-week and month as cyclical encodings
        try:
            dt_index = pd.DatetimeIndex(dataframe["date"])  # type: ignore[index]
            dow = dt_index.dayofweek
            month = dt_index.month
            dataframe["%-dow_sin"] = np.sin(2 * np.pi * dow / 7)
            dataframe["%-dow_cos"] = np.cos(2 * np.pi * dow / 7)
            dataframe["%-month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
            dataframe["%-month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)
        except Exception:
            pass

        # Clip all *_dz z-scores
        for col in [c for c in dataframe.columns if c.startswith("%-") and "_dz_" in c]:
            dataframe[col] = dataframe[col].clip(lower=-5, upper=5)

        # Clean engineered features
        engineered_features = [col for col in dataframe.columns if col.startswith("%-")]
        for col in list(engineered_features):
            if dataframe[col].notna().sum() == 0:
                dataframe.drop(columns=[col], inplace=True)
        for col in [c for c in dataframe.columns if c.startswith("%-")]:
            dataframe[col] = (
                dataframe[col].replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0)
            )

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: dict, **kwargs) -> DataFrame:
        """
        *Only functional with FreqAI enabled strategies*
        Required function to set the targets for the model.
        All targets must be prepended with `&` to be recognized by the FreqAI internals.

        :param dataframe: strategy dataframe which will receive the targets
        :param metadata: metadata of current pair
        """
        # Target: forward return over label_period; winsorize to reduce outliers
        label_period = self.freqai_info["feature_parameters"]["label_period_candles"]
        raw = dataframe["close"].shift(-label_period) / dataframe["close"] - 1
        # Clip extreme returns to ±10%
        target_series = raw.clip(lower=-0.10, upper=0.10)
        dataframe = pd.concat([dataframe, pd.DataFrame({"&-target_3d": target_series})], axis=1)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Load and merge Glassnode on-chain data with price data
        """
        try:
            # Load Glassnode data (prefer daily, fallback to hourly and resample to daily)
            glassnode_full_daily_path = "user_data/data/glassnode/btc-glassnode_full_1d.csv"
            glassnode_daily_path = "user_data/data/glassnode/btc-glassnode_1d.csv"
            glassnode_hourly_path = "user_data/data/glassnode/btc-glassnode_1h.csv"

            merged_onchain = None
            if os.path.exists(glassnode_full_daily_path):
                # Preferred: use the full daily export and map columns
                df_gn = pd.read_csv(glassnode_full_daily_path)
                ts = pd.to_datetime(df_gn["timestamp"], errors="coerce", dayfirst=True, utc=True)
                missing = ts.isna()
                if missing.any():
                    ts2 = pd.to_datetime(df_gn.loc[missing, "timestamp"], errors="coerce")
                    ts.loc[missing] = ts2
                df_gn["timestamp"] = ts
                df_gn = df_gn.dropna(subset=["timestamp"]).copy()

                # Drop unnamed/blank columns that may appear due to CSV formatting
                drop_cols = [
                    c
                    for c in df_gn.columns
                    if not isinstance(c, str) or c.strip() == "" or c.startswith("Unnamed")
                ]
                if drop_cols:
                    df_gn.drop(columns=drop_cols, inplace=True)

                # Rename verbose Glassnode headers to internal names
                col_map = {
                    "BTC: Price": "gn_price",
                    "BTC: Realized Profit/Loss Ratio": "realized_pnl_ratio",
                    "BTC: Spent Output Profit Ratio (SOPR)": "sopr",
                    "BTC: Network Value to Transactions Ratio (NVT)": "nvt_ratio",
                    "BTC: Market Value to Realized Value Ratio (MVRV)": "mvrv",
                    "BTC: Net Transfer Volume from/to Exchanges": "exchange_netflow",
                    "Exchange Whale Net-flow (Volume)": "whale_exchange_netflow",
                    "BTC: Total Supply Held by Long-Term Holders": "long_term_holder_supply",
                    "BTC: Total Supply Held by Short-Term Holders": "short_term_holder_supply",
                }
                df_gn.rename(columns=col_map, inplace=True)

                # Ensure numeric for all on-chain columns except timestamp
                for col in df_gn.columns:
                    if col == "timestamp":
                        continue
                    df_gn[col] = pd.to_numeric(df_gn[col], errors="coerce")

                logger.info(f"Loaded Glassnode full daily data from {glassnode_full_daily_path}")

                # Merge with main dataframe using asof (backward) with 1D tolerance
                left_df = dataframe[["date"]].reset_index().sort_values("date")
                right_df = df_gn.sort_values("timestamp")
                merged_onchain = (
                    pd.merge_asof(
                        left_df,
                        right_df,
                        left_on="date",
                        right_on="timestamp",
                        direction="backward",
                        tolerance=pd.Timedelta("1D"),
                    )
                    .set_index("index")
                    .reindex(dataframe.index)
                )
            elif os.path.exists(glassnode_daily_path):
                df_gn = pd.read_csv(glassnode_daily_path)
                # Robust datetime parsing supporting multiple formats
                ts = pd.to_datetime(df_gn["timestamp"], errors="coerce", dayfirst=True, utc=True)
                missing = ts.isna()
                if missing.any():
                    ts2 = pd.to_datetime(df_gn.loc[missing, "timestamp"], errors="coerce")
                    ts.loc[missing] = ts2
                df_gn["timestamp"] = ts
                df_gn = df_gn.dropna(subset=["timestamp"])
                # Ensure numeric for all on-chain columns
                for col in df_gn.columns:
                    if col == "timestamp":
                        continue
                    df_gn[col] = pd.to_numeric(df_gn[col], errors="coerce")
                logger.info(f"Loaded Glassnode daily data from {glassnode_daily_path}")

                # Merge with main dataframe using asof (backward) with 1D tolerance
                left_df = dataframe[["date"]].reset_index().sort_values("date")
                right_df = df_gn.sort_values("timestamp")
                merged_onchain = (
                    pd.merge_asof(
                        left_df,
                        right_df,
                        left_on="date",
                        right_on="timestamp",
                        direction="backward",
                        tolerance=pd.Timedelta("1D"),
                    )
                    .set_index("index")
                    .reindex(dataframe.index)
                )
            elif os.path.exists(glassnode_hourly_path):
                df_gn_h = pd.read_csv(glassnode_hourly_path)
                ts = pd.to_datetime(df_gn_h["timestamp"], errors="coerce", dayfirst=True, utc=True)
                missing = ts.isna()
                if missing.any():
                    ts2 = pd.to_datetime(df_gn_h.loc[missing, "timestamp"], errors="coerce")
                    ts.loc[missing] = ts2
                df_gn_h["timestamp"] = ts
                df_gn_h = df_gn_h.dropna(subset=["timestamp"])
                # Ensure numeric for all on-chain columns
                for col in df_gn_h.columns:
                    if col == "timestamp":
                        continue
                    df_gn_h[col] = pd.to_numeric(df_gn_h[col], errors="coerce")

                # Resample hourly to daily using aggregation rules:
                # - Sum for netflow-related columns
                # - Last observation for all other columns
                netflow_columns = {"exchange_netflow", "whale_exchange_netflow"}
                agg_rules: dict[str, str] = {}
                for col in df_gn_h.columns:
                    if col == "timestamp":
                        continue
                    agg_rules[col] = "sum" if col in netflow_columns else "last"

                df_gn_h = df_gn_h.set_index("timestamp").resample("1D").agg(agg_rules).reset_index()
                logger.info(
                    f"Loaded Glassnode hourly data and resampled to daily (sum netflows, last others) from {glassnode_hourly_path}"
                )

                left_df = dataframe[["date"]].reset_index().sort_values("date")
                right_df = df_gn_h.sort_values("timestamp")
                merged_onchain = (
                    pd.merge_asof(
                        left_df,
                        right_df,
                        left_on="date",
                        right_on="timestamp",
                        direction="backward",
                        tolerance=pd.Timedelta("1D"),
                    )
                    .set_index("index")
                    .reindex(dataframe.index)
                )
            else:
                logger.warning(
                    f"Glassnode data not found at {glassnode_full_daily_path}, {glassnode_daily_path} or {glassnode_hourly_path}"
                )

            if merged_onchain is not None:
                # Add all on-chain columns to main dataframe
                for col in merged_onchain.columns:
                    if col == "timestamp" or col == "date":
                        continue
                    dataframe[col] = merged_onchain[col]

                logger.info(f"Merged Glassnode on-chain data for {metadata['pair']}")
                # Compute coverage across available on-chain columns
                onchain_cols = [
                    "sopr",
                    "exchange_netflow",
                    "realized_pnl_ratio",
                    "nvt_price",
                    "nvt_price_90D",
                    "90D_NVT_premium",
                    "nvt_ratio",
                    "mvrv",
                    "whale_exchange_netflow",
                    "long_term_holder_supply",
                    "short_term_holder_supply",
                ]
                present_cols = [c for c in onchain_cols if c in dataframe.columns]
                coverage = dataframe[present_cols].notna().any(axis=1).sum() if present_cols else 0
                logger.info(
                    f"Glassnode data coverage (rows with any on-chain): {coverage}/{len(dataframe)}"
                )

            # Load daily LTH/STH data and align to daily timeframe
            lth_sth_path = "user_data/data/glassnode/lth_sth_supply_1d.csv"
            if os.path.exists(lth_sth_path):
                lth_sth_data = pd.read_csv(lth_sth_path)
                # Robust datetime parsing supporting multiple formats
                ts = pd.to_datetime(
                    lth_sth_data["timestamp"], errors="coerce", dayfirst=True, utc=True
                )
                missing = ts.isna()
                if missing.any():
                    ts2 = pd.to_datetime(lth_sth_data.loc[missing, "timestamp"], errors="coerce")
                    ts.loc[missing] = ts2
                lth_sth_data["timestamp"] = ts
                lth_sth_data = lth_sth_data.dropna(subset=["timestamp"])
                # Ensure numeric for supply columns if present
                for col in ["LTH_supply", "STH_supply"]:
                    if col in lth_sth_data.columns:
                        lth_sth_data[col] = pd.to_numeric(lth_sth_data[col], errors="coerce")

                # Ensure daily sampling and forward fill
                lth_sth_data.set_index("timestamp", inplace=True)
                lth_sth_daily = lth_sth_data.resample("1D").ffill().reset_index()

                # Merge with main dataframe using asof (backward) with 1D tolerance
                left_df = dataframe[["date"]].reset_index().sort_values("date")
                right_df = lth_sth_daily.sort_values("timestamp")
                merged = pd.merge_asof(
                    left_df,
                    right_df,
                    left_on="date",
                    right_on="timestamp",
                    direction="backward",
                    tolerance=pd.Timedelta("1D"),
                )
                merged = merged.set_index("index").reindex(dataframe.index)

                # Add LTH/STH columns
                if "LTH_supply" in merged.columns:
                    dataframe["long_term_holder_supply"] = merged["LTH_supply"]
                if "STH_supply" in merged.columns:
                    dataframe["short_term_holder_supply"] = merged["STH_supply"]

                logger.info(f"Merged LTH/STH data for {metadata['pair']}")
            else:
                logger.warning(f"LTH/STH data not found at {lth_sth_path}")

            # Drop raw on-chain columns that are entirely NaN to avoid constant-zero features later
            onchain_cols_dropcheck = [
                "sopr",
                "exchange_netflow",
                "realized_pnl_ratio",
                "nvt_price",
                "nvt_price_90D",
                "90D_NVT_premium",
                "nvt_ratio",
                "mvrv",
                "whale_exchange_netflow",
                "long_term_holder_supply",
                "short_term_holder_supply",
                "gn_price",
            ]
            for c in onchain_cols_dropcheck:
                if c in dataframe.columns and dataframe[c].notna().sum() == 0:
                    dataframe.drop(columns=[c], inplace=True)

            # Log non-null counts per on-chain column for diagnostics
            nn_info = {
                c: int(dataframe[c].notna().sum())
                for c in onchain_cols_dropcheck
                if c in dataframe.columns
            }
            logger.info(f"On-chain non-null counts over window: {nn_info}")

        except Exception as e:
            logger.error(f"Error loading Glassnode data: {e}")
            # Set default values for on-chain features
            onchain_features = [
                "sopr",
                "exchange_netflow",
                "realized_pnl_ratio",
                "nvt_price",
                "nvt_price_90D",
                "90D_NVT_premium",
                "whale_exchange_netflow",
                "long_term_holder_supply",
                "short_term_holder_supply",
            ]
            for feature in onchain_features:
                dataframe[feature] = 0.0

        # Log engineered features before starting FreqAI, to diagnose feature availability
        engineered_cols = [
            c for c in dataframe.columns if isinstance(c, str) and c.startswith("%-")
        ]
        if engineered_cols:
            logger.info(
                f"Engineered features available: {len(engineered_cols)} (showing up to 10): "
                f"{engineered_cols[:10]}"
            )
        else:
            # Expected: engineered features are generated inside FreqAI's feature_engineering_standard during training
            logger.info(
                "Engineered features (prefix '%-') will be generated during FreqAI feature engineering."
            )

        # The model will return all labels created by user in `set_freqai_targets()`
        dataframe = self.freqai.start(dataframe, metadata, self)

        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry logic for swing trading based on AI predictions and on-chain signals
        """
        # Long entry conditions (relaxed threshold, no DI gating)
        enter_long_conditions = [
            df["&-target_3d"] > 0.005,
        ]

        # Add volume confirmation
        if "%-relative_volume-period" in df.columns:
            enter_long_conditions.append(df["%-relative_volume-period"] > 1.2)

        # (Temporarily remove on-chain confirmations to assess pure model signal)

        # Combine conditions
        if enter_long_conditions:
            combined_condition = enter_long_conditions[0]
            for condition in enter_long_conditions[1:]:
                combined_condition = combined_condition & condition

            df.loc[combined_condition, "enter_long"] = 1

        # Short entry conditions (not used since can_short=False, keep for completeness)
        enter_short_conditions = [
            df["&-target_3d"] < -0.005,
        ]

        # Add volume confirmation for shorts
        if "%-relative_volume-period" in df.columns:
            enter_short_conditions.append(df["%-relative_volume-period"] > 1.2)

        # (Temporarily remove on-chain confirmations for shorts)

        # Combine short conditions
        if enter_short_conditions:
            combined_short_condition = enter_short_conditions[0]
            for condition in enter_short_conditions[1:]:
                combined_short_condition = combined_short_condition & condition

            df.loc[combined_short_condition, "enter_short"] = 1

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit logic for swing trading
        """
        # Long exit conditions
        exit_long_conditions = [
            (df["&-target_3d"] < -0.002),  # Exit earlier when model turns slightly negative
        ]

        # (Remove on-chain exit confirmations for assessment)

        # Combine long exit conditions
        if exit_long_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_long_conditions), "exit_long"] = 1

        # Short exit conditions
        exit_short_conditions = [
            (df["&-target_3d"] > 0.002),
        ]

        # (Remove on-chain exit confirmations for shorts)

        # Combine short exit conditions
        if exit_short_conditions:
            df.loc[reduce(lambda x, y: x & y, exit_short_conditions), "exit_short"] = 1

        return df


class GlassnodeSwingElasticNet(GlassnodeSwingStrategy):
    """
    Same strategy, but using sklearn ElasticNet via wrapper for FreqAI.
    """

    # Fall back to LightGBM if wrapper not available
    if SkElasticNetRegressor is not None:
        freqai_model = SkElasticNetRegressor
    else:  # pragma: no cover
        freqai_model = LightGBMRegressor


class GlassnodeSwingLinear(GlassnodeSwingStrategy):
    """
    Same strategy, but using sklearn LinearRegression via wrapper for FreqAI.
    """

    if SkLinearRegressionRegressor is not None:
        freqai_model = SkLinearRegressionRegressor
    else:  # pragma: no cover
        freqai_model = LightGBMRegressor
