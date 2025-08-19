"""
Enhanced Analysis Script for BTCPriceDirectionStrategyV2.1

This script tests the V2.1 improvements:
1. Compares V2.0 vs V2.1 feature quality and performance
2. Tests feature group isolation (volatility, momentum, lag buckets)
3. Validates volatility-focused approach vs traditional indicators
4. Analyzes lag feature effectiveness
5. Tests volatility regime vs SMA regime performance

Key Focus Areas:
- Multi-scale volatility features with PCA compression
- Enhanced momentum divergence (MACD slope, breakout persistence)
- Proper lagged features for persistence/mean-reversion
- Volatility clustering vs SMA-based regime detection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Try importing ML libraries
try:
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    ML_AVAILABLE = True
except ImportError as e:
    print(f"ML libraries not available: {e}")
    ML_AVAILABLE = False


class StrategyV2_1_Analyzer:
    """Enhanced analyzer for V2.1 strategy with feature group testing."""

    def __init__(self):
        self.results = {}
        self.v2_0_features = {}
        self.v2_1_features = {}
        self.feature_group_analysis = {}

    def load_data(self, days=1000):
        """Load or generate data for analysis."""
        print("📊 Loading data for V2.1 analysis...")

        # Generate enhanced synthetic data with more realistic patterns
        dates = pd.date_range(start="2022-01-01", periods=days, freq="D")

        # Generate more realistic BTC price movements with volatility clustering
        np.random.seed(42)

        # Create volatility regimes
        regime_changes = np.random.choice(
            [0, 1], size=days, p=[0.85, 0.15]
        )  # 15% chance of regime change
        vol_base = 0.02  # Base volatility
        vol_regimes = []
        current_vol = vol_base

        for change in regime_changes:
            if change == 1:  # Regime change
                current_vol = np.random.uniform(0.01, 0.08)  # New volatility level
            vol_regimes.append(current_vol)

        # Generate returns with volatility clustering
        returns = []
        for i, vol in enumerate(vol_regimes):
            if i == 0:
                returns.append(np.random.normal(0.001, vol))
            else:
                # Add some persistence to returns
                persistence = 0.1 * returns[-1]  # 10% persistence
                returns.append(persistence + np.random.normal(0.001, vol))

        # Create prices from returns
        price = 45000  # Starting price
        prices = [price]
        for ret in returns[1:]:
            price = price * (1 + ret)
            prices.append(price)

        # Create realistic OHLCV data
        data = pd.DataFrame(
            {
                "date": dates,
                "open": prices,
                "high": [
                    p * (1 + abs(np.random.normal(0, vol_regimes[i] * 0.5)))
                    for i, p in enumerate(prices)
                ],
                "low": [
                    p * (1 - abs(np.random.normal(0, vol_regimes[i] * 0.5)))
                    for i, p in enumerate(prices)
                ],
                "close": prices,
                "volume": np.random.uniform(1000, 10000, days),
            }
        )

        data.set_index("date", inplace=True)
        self.data = data

        print(f"✅ Generated {len(data)} candles with realistic volatility clustering")
        return True

    def extract_v2_0_features(self, dataframe):
        """Extract V2.0 features for comparison."""
        print("🔧 Extracting V2.0 features...")

        df = dataframe.copy()

        # V2.0 Features (from previous analysis)
        daily_sma_20 = df["close"].rolling(20).mean()
        daily_sma_50 = df["close"].rolling(50).mean()
        df["daily_sma_regime"] = np.where(daily_sma_20 > daily_sma_50, 1, 0)
        df["daily_sma_slope"] = (daily_sma_20 - daily_sma_50) / daily_sma_50

        df["daily_rsi"] = self._calculate_rsi(df, 14)
        macd_data = self._calculate_macd(df)
        df["daily_macd_hist"] = macd_data["histogram"]

        daily_high_20 = df["high"].rolling(20).max()
        df["close_high_ratio"] = df["close"] / daily_high_20

        daily_atr = self._calculate_atr(df, 14)
        df["daily_atr_percent"] = daily_atr / df["close"]
        df["daily_atr_raw"] = daily_atr

        if "volume" in df.columns:
            daily_obv = self._calculate_obv(df)
            df["obv_momentum"] = daily_obv.diff(5)
        else:
            df["obv_momentum"] = 0

        daily_returns = df["close"].pct_change()
        daily_vol = daily_returns.rolling(14).std()
        daily_vol_mean = daily_vol.rolling(50).mean()
        df["vol_regime"] = daily_vol / daily_vol_mean

        v2_0_columns = [
            "daily_sma_regime",
            "daily_sma_slope",
            "daily_rsi",
            "daily_macd_hist",
            "close_high_ratio",
            "daily_atr_percent",
            "daily_atr_raw",
            "obv_momentum",
            "vol_regime",
        ]

        # Clean features
        for col in v2_0_columns:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).ffill().bfill()

        print(f"✅ Extracted {len(v2_0_columns)} V2.0 features")
        return df[v2_0_columns].dropna()

    def extract_v2_1_features(self, dataframe):
        """Extract V2.1 enhanced features."""
        print("🔧 Extracting V2.1 enhanced features...")

        df = dataframe.copy()

        # ================================================================
        # V2.1 ENHANCED FEATURES
        # ================================================================

        # Multi-scale volatility with PCA
        atr_7 = self._calculate_atr(df, 7)
        atr_14 = self._calculate_atr(df, 14)
        atr_21 = self._calculate_atr(df, 21)

        atr_7_pct = atr_7 / df["close"]
        atr_14_pct = atr_14 / df["close"]
        atr_21_pct = atr_21 / df["close"]

        # Volatility ratios
        df["vol_ratio_7_14"] = atr_7 / (atr_14 + 1e-8)
        df["vol_ratio_14_21"] = atr_14 / (atr_21 + 1e-8)
        df["atr_momentum_3d"] = atr_14.diff(3) / (atr_14.shift(3) + 1e-8)

        # PCA compression of multi-scale ATRs
        if ML_AVAILABLE:
            atr_features = (
                pd.DataFrame(
                    {"atr_7_pct": atr_7_pct, "atr_14_pct": atr_14_pct, "atr_21_pct": atr_21_pct}
                )
                .fillna(method="ffill")
                .fillna(0)
            )

            if len(atr_features.dropna()) > 50:
                scaler = StandardScaler()
                atr_scaled = scaler.fit_transform(atr_features)

                pca = PCA(n_components=2)
                atr_pca = pca.fit_transform(atr_scaled)

                df["vol_pca_1"] = atr_pca[:, 0]
                df["vol_pca_2"] = atr_pca[:, 1]
            else:
                df["vol_pca_1"] = atr_14_pct
                df["vol_pca_2"] = df["vol_ratio_14_21"]
        else:
            df["vol_pca_1"] = atr_14_pct
            df["vol_pca_2"] = df["vol_ratio_14_21"]

        # Enhanced momentum features
        df["daily_rsi"] = self._calculate_rsi(df, 14)
        macd_data = self._calculate_macd(df)
        df["daily_macd_hist"] = macd_data["histogram"]
        df["macd_momentum"] = macd_data["histogram"].diff(3)

        # Breakout persistence
        daily_high_20 = df["high"].rolling(20).max()
        breakout_count = []
        for i in range(len(df)):
            if i < 20:
                breakout_count.append(0)
            else:
                consecutive_days = 0
                for j in range(i, max(0, i - 10), -1):
                    if df["close"].iloc[j] > daily_high_20.iloc[j - 1]:
                        consecutive_days += 1
                    else:
                        break
                breakout_count.append(consecutive_days)
        df["breakout_persistence"] = breakout_count

        # Volatility-based regime (replace SMA regime)
        vol_regime = atr_14 / atr_14.rolling(50).mean()
        df["vol_regime_cluster"] = np.where(vol_regime > 1.2, 1, 0)
        df["vol_trend"] = (atr_14.rolling(5).mean() / atr_14.rolling(20).mean()) - 1

        # Keep best from V2.0
        daily_sma_20 = df["close"].rolling(20).mean()
        daily_sma_50 = df["close"].rolling(50).mean()
        df["daily_sma_slope"] = (daily_sma_20 - daily_sma_50) / daily_sma_50

        # Volume momentum
        if "volume" in df.columns:
            daily_obv = self._calculate_obv(df)
            df["obv_momentum"] = daily_obv.diff(5)
        else:
            df["obv_momentum"] = 0

        # Lag features (simulated)
        df["vol_lag1"] = df["vol_pca_1"].shift(1).fillna(0)
        df["rsi_lag2"] = df["daily_rsi"].shift(2).fillna(0)
        df["macd_lag1"] = df["macd_momentum"].shift(1).fillna(0)

        v2_1_columns = [
            "vol_pca_1",
            "vol_pca_2",
            "vol_ratio_7_14",
            "vol_ratio_14_21",
            "atr_momentum_3d",
            "daily_rsi",
            "daily_macd_hist",
            "macd_momentum",
            "breakout_persistence",
            "vol_regime_cluster",
            "vol_trend",
            "daily_sma_slope",
            "obv_momentum",
            "vol_lag1",
            "rsi_lag2",
            "macd_lag1",
        ]

        # Clean features
        for col in v2_1_columns:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).ffill().bfill()

        print(f"✅ Extracted {len(v2_1_columns)} V2.1 enhanced features")
        return df[v2_1_columns].dropna()

    def create_targets(self, dataframe, threshold=0.025):
        """Create targets with V2.1's higher threshold."""
        print(f"🎯 Creating targets with {threshold * 100}% threshold...")

        forward_return = dataframe["close"].shift(-3) / dataframe["close"] - 1
        targets = np.where(forward_return > threshold, 1, 0)
        actual_direction = np.where(forward_return > 0, 1, 0)

        return pd.Series(targets, index=dataframe.index), pd.Series(
            actual_direction, index=dataframe.index
        )

    def compare_v2_0_vs_v2_1(self):
        """Compare V2.0 vs V2.1 performance."""
        print("\n" + "=" * 60)
        print("🔍 COMPARING V2.0 vs V2.1 STRATEGIES")
        print("=" * 60)

        # Extract features
        v2_0_features = self.extract_v2_0_features(self.data)
        v2_1_features = self.extract_v2_1_features(self.data)
        conservative_targets, actual_targets = self.create_targets(self.data)

        # Align data
        common_index = v2_0_features.index.intersection(v2_1_features.index).intersection(
            conservative_targets.index
        )
        v2_0_features = v2_0_features.loc[common_index]
        v2_1_features = v2_1_features.loc[common_index]
        conservative_targets = conservative_targets.loc[common_index]
        actual_targets = actual_targets.loc[common_index]

        print(f"\n📊 Comparison Overview:")
        print(f"  V2.0 Features: {len(v2_0_features.columns)}")
        print(f"  V2.1 Features: {len(v2_1_features.columns)}")
        print(f"  Common samples: {len(common_index)}")

        if not ML_AVAILABLE:
            print("⚠️ ML libraries not available, skipping model comparison")
            return

        # Test both versions
        results = {}

        for version, features in [("V2.0", v2_0_features), ("V2.1", v2_1_features)]:
            print(f"\n🧪 Testing {version}...")

            # Remove NaN values
            mask = ~(
                features.isna().any(axis=1) | conservative_targets.isna() | actual_targets.isna()
            )
            X = features[mask]
            y_conservative = conservative_targets[mask]
            y_actual = actual_targets[mask]

            if len(X) < 100:
                print(f"⚠️ Insufficient data for {version}: {len(X)} samples")
                continue

            # Time series split evaluation
            tscv = TimeSeriesSplit(n_splits=3)
            conservative_accuracies = []
            actual_accuracies = []
            feature_importances = []

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train = y_conservative.iloc[train_idx]
                y_test_conservative = y_conservative.iloc[test_idx]
                y_test_actual = y_actual.iloc[test_idx]

                # Train model
                model = lgb.LGBMClassifier(
                    n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42, verbose=-1
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Calculate accuracies
                acc_conservative = accuracy_score(y_test_conservative, y_pred)
                acc_actual = accuracy_score(y_test_actual, y_pred)

                conservative_accuracies.append(acc_conservative)
                actual_accuracies.append(acc_actual)

                # Store feature importance
                importance = dict(zip(X.columns, model.feature_importances_))
                feature_importances.append(importance)

            results[version] = {
                "conservative_accuracy_mean": np.mean(conservative_accuracies),
                "conservative_accuracy_std": np.std(conservative_accuracies),
                "actual_accuracy_mean": np.mean(actual_accuracies),
                "actual_accuracy_std": np.std(actual_accuracies),
                "feature_importances": feature_importances,
            }

            print(
                f"  Conservative Accuracy: {results[version]['conservative_accuracy_mean']:.3f} ± {results[version]['conservative_accuracy_std']:.3f}"
            )
            print(
                f"  Actual Accuracy: {results[version]['actual_accuracy_mean']:.3f} ± {results[version]['actual_accuracy_std']:.3f}"
            )

        # Compare results
        if "V2.0" in results and "V2.1" in results:
            print(f"\n📈 COMPARISON RESULTS:")

            cons_improvement = (
                results["V2.1"]["conservative_accuracy_mean"]
                - results["V2.0"]["conservative_accuracy_mean"]
            )
            actual_improvement = (
                results["V2.1"]["actual_accuracy_mean"] - results["V2.0"]["actual_accuracy_mean"]
            )

            print(f"  Conservative Accuracy Improvement: {cons_improvement:+.3f}")
            print(f"  Actual Accuracy Improvement: {actual_improvement:+.3f}")

            if cons_improvement > 0.02:  # 2% improvement
                print(f"  ✅ V2.1 shows significant improvement!")
            elif cons_improvement > 0:
                print(f"  ⚡ V2.1 shows modest improvement")
            else:
                print(f"  ⚠️ V2.1 needs further refinement")

        return results

    def test_feature_groups(self):
        """Test individual feature groups (volatility, momentum, lag)."""
        print("\n" + "=" * 60)
        print("🔬 FEATURE GROUP ISOLATION TESTING")
        print("=" * 60)

        if not ML_AVAILABLE:
            print("⚠️ ML libraries not available, skipping feature group testing")
            return

        # Extract all V2.1 features
        all_features = self.extract_v2_1_features(self.data)
        conservative_targets, _ = self.create_targets(self.data)

        # Define feature groups
        feature_groups = {
            "Volatility": [
                "vol_pca_1",
                "vol_pca_2",
                "vol_ratio_7_14",
                "vol_ratio_14_21",
                "atr_momentum_3d",
                "vol_regime_cluster",
                "vol_trend",
            ],
            "Momentum": [
                "daily_rsi",
                "daily_macd_hist",
                "macd_momentum",
                "breakout_persistence",
                "daily_sma_slope",
            ],
            "Lag": ["vol_lag1", "rsi_lag2", "macd_lag1"],
            "Volume": ["obv_momentum"],
        }

        group_results = {}

        for group_name, group_features in feature_groups.items():
            print(f"\n🧪 Testing {group_name} Group...")

            # Select features for this group
            available_features = [f for f in group_features if f in all_features.columns]

            if not available_features:
                print(f"  ⚠️ No features available for {group_name} group")
                continue

            group_data = all_features[available_features]

            # Align data
            common_index = group_data.index.intersection(conservative_targets.index)
            X = group_data.loc[common_index]
            y = conservative_targets.loc[common_index]

            # Remove NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]

            if len(X) < 100:
                print(f"  ⚠️ Insufficient data for {group_name}: {len(X)} samples")
                continue

            # Test this group
            tscv = TimeSeriesSplit(n_splits=3)
            accuracies = []

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                model = lgb.LGBMClassifier(
                    n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)

            group_accuracy = np.mean(accuracies)
            group_std = np.std(accuracies)

            group_results[group_name] = {
                "accuracy_mean": group_accuracy,
                "accuracy_std": group_std,
                "feature_count": len(available_features),
                "features": available_features,
            }

            print(f"  Features: {available_features}")
            print(f"  Accuracy: {group_accuracy:.3f} ± {group_std:.3f}")

        # Rank feature groups
        print(f"\n🏆 FEATURE GROUP RANKINGS:")
        sorted_groups = sorted(
            group_results.items(), key=lambda x: x[1]["accuracy_mean"], reverse=True
        )

        for i, (group, results) in enumerate(sorted_groups):
            print(
                f"  {i + 1}. {group:12s}: {results['accuracy_mean']:.3f} ± {results['accuracy_std']:.3f} ({results['feature_count']} features)"
            )

        return group_results

    def test_regime_detection(self):
        """Test volatility clustering vs SMA-based regime detection."""
        print("\n" + "=" * 60)
        print("📊 REGIME DETECTION COMPARISON")
        print("=" * 60)

        if not ML_AVAILABLE:
            print("⚠️ ML libraries not available, skipping regime testing")
            return

        df = self.data.copy()

        # SMA-based regime (V2.0 approach)
        daily_sma_20 = df["close"].rolling(20).mean()
        daily_sma_50 = df["close"].rolling(50).mean()
        sma_regime = np.where(daily_sma_20 > daily_sma_50, 1, 0)

        # Volatility clustering regime (V2.1 approach)
        atr_14 = self._calculate_atr(df, 14)
        vol_regime = atr_14 / atr_14.rolling(50).mean()
        vol_cluster_regime = np.where(vol_regime > 1.2, 1, 0)

        # Test both regime types
        conservative_targets, _ = self.create_targets(df)

        regimes = {
            "SMA Regime": pd.Series(sma_regime, index=df.index),
            "Volatility Cluster": pd.Series(vol_cluster_regime, index=df.index),
        }

        regime_results = {}

        for regime_name, regime_series in regimes.items():
            print(f"\n🧪 Testing {regime_name}...")

            # Align data
            common_index = regime_series.index.intersection(conservative_targets.index)
            X = regime_series.loc[common_index].to_frame()
            y = conservative_targets.loc[common_index]

            # Remove NaN values
            mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[mask]
            y = y[mask]

            if len(X) < 100:
                print(f"  ⚠️ Insufficient data for {regime_name}: {len(X)} samples")
                continue

            # Test regime predictive power
            tscv = TimeSeriesSplit(n_splits=3)
            accuracies = []

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                model = lgb.LGBMClassifier(
                    n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42, verbose=-1
                )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)

            regime_accuracy = np.mean(accuracies)
            regime_std = np.std(accuracies)

            # Calculate regime statistics
            regime_stats = {
                "accuracy_mean": regime_accuracy,
                "accuracy_std": regime_std,
                "regime_frequency": regime_series.mean(),
                "regime_transitions": (regime_series.diff() != 0).sum(),
            }

            regime_results[regime_name] = regime_stats

            print(f"  Accuracy: {regime_accuracy:.3f} ± {regime_std:.3f}")
            print(f"  Regime Frequency: {regime_stats['regime_frequency']:.3f}")
            print(f"  Transitions: {regime_stats['regime_transitions']}")

        return regime_results

    def generate_comprehensive_report(self):
        """Generate comprehensive V2.1 analysis report."""
        print("\n" + "=" * 80)
        print("📋 COMPREHENSIVE V2.1 STRATEGY ANALYSIS REPORT")
        print("=" * 80)

        # Load data
        if not self.load_data():
            print("❌ Failed to load data. Exiting.")
            return

        # Run all analyses
        comparison_results = self.compare_v2_0_vs_v2_1()
        group_results = self.test_feature_groups()
        regime_results = self.test_regime_detection()

        # Generate executive summary
        print("\n" + "=" * 80)
        print("📈 V2.1 EXECUTIVE SUMMARY")
        print("=" * 80)

        print(f"\n🎯 KEY FINDINGS:")

        if comparison_results and "V2.0" in comparison_results and "V2.1" in comparison_results:
            v2_0_acc = comparison_results["V2.0"]["conservative_accuracy_mean"]
            v2_1_acc = comparison_results["V2.1"]["conservative_accuracy_mean"]
            improvement = v2_1_acc - v2_0_acc

            print(f"  • V2.1 vs V2.0 Improvement: {improvement:+.3f} ({improvement * 100:+.1f}%)")

            if improvement > 0.02:
                print(f"  ✅ Significant improvement achieved!")
            elif improvement > 0:
                print(f"  ⚡ Modest improvement, continue refinement")
            else:
                print(f"  ⚠️ No improvement, investigate feature engineering")

        if group_results:
            best_group = max(group_results.items(), key=lambda x: x[1]["accuracy_mean"])
            print(
                f"  • Best Feature Group: {best_group[0]} ({best_group[1]['accuracy_mean']:.3f} accuracy)"
            )

        if regime_results:
            if "Volatility Cluster" in regime_results and "SMA Regime" in regime_results:
                vol_acc = regime_results["Volatility Cluster"]["accuracy_mean"]
                sma_acc = regime_results["SMA Regime"]["accuracy_mean"]
                regime_improvement = vol_acc - sma_acc
                print(f"  • Volatility vs SMA Regime: {regime_improvement:+.3f} improvement")

        print(f"\n📋 RECOMMENDATIONS:")
        recommendations = []

        if comparison_results and "V2.1" in comparison_results:
            if comparison_results["V2.1"]["conservative_accuracy_mean"] > 0.55:
                recommendations.append("✅ V2.1 shows predictive power - proceed with backtesting")
            else:
                recommendations.append("⚠️ Continue feature engineering - accuracy still too low")

        if group_results:
            best_groups = sorted(
                group_results.items(), key=lambda x: x[1]["accuracy_mean"], reverse=True
            )[:2]
            recommendations.append(
                f"🎯 Focus on top groups: {', '.join([g[0] for g in best_groups])}"
            )

        if not recommendations:
            recommendations.append("🔬 Continue iterative feature engineering")

        for rec in recommendations:
            print(f"  {rec}")

        print("\n" + "=" * 80)

    # Helper methods for technical indicators
    def _calculate_rsi(self, dataframe, period=14):
        """Calculate RSI."""
        delta = dataframe["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, dataframe, fast=12, slow=26, signal=9):
        """Calculate MACD."""
        exp1 = dataframe["close"].ewm(span=fast).mean()
        exp2 = dataframe["close"].ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        return {"macd": macd_line, "signal": signal_line, "histogram": histogram}

    def _calculate_atr(self, dataframe, period=14):
        """Calculate Average True Range."""
        high_low = dataframe["high"] - dataframe["low"]
        high_close = np.abs(dataframe["high"] - dataframe["close"].shift())
        low_close = np.abs(dataframe["low"] - dataframe["close"].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def _calculate_obv(self, dataframe):
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


def main():
    """Run comprehensive V2.1 analysis."""
    print("🚀 Starting V2.1 Enhanced Strategy Analysis")
    print("=" * 60)

    # Initialize analyzer
    analyzer = StrategyV2_1_Analyzer()

    # Run comprehensive analysis
    analyzer.generate_comprehensive_report()

    print("\n🎉 V2.1 Analysis complete! Review the results above.")


if __name__ == "__main__":
    main()
