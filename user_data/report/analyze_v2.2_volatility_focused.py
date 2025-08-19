"""
V2.2 Volatility-Focused Strategy Analysis

This script tests the V2.2 "less is more" approach:
1. Compares V2.0 vs V2.1 vs V2.2 performance
2. Validates that 7 focused features outperform 16 scattered features
3. Tests volatility-first hypothesis against momentum and lag approaches
4. Measures feature quality vs quantity impact

V2.2 Design:
- 5 core volatility features (multi-scale, clustering, momentum)
- 2 selective momentum features (RSI, MACD slope)
- Removed: all lag features, volume features, redundant indicators
- Hypothesis: Focused volatility approach > feature bloat
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


class StrategyV2_2_Analyzer:
    """Analyzer for V2.2 volatility-focused strategy validation."""

    def __init__(self):
        self.results = {}
        self.comparison_data = {}

    def load_data(self, days=1000):
        """Load enhanced synthetic data with strong volatility patterns."""
        print("📊 Loading data for V2.2 volatility-focused analysis...")

        # Generate enhanced synthetic data with MORE realistic volatility clustering
        dates = pd.date_range(start="2022-01-01", periods=days, freq="D")
        np.random.seed(42)

        # Create more pronounced volatility regimes
        regime_switches = []
        current_regime = 0  # 0=low vol, 1=high vol
        switch_prob = 0.05  # 5% chance of regime switch per day

        for i in range(days):
            if np.random.random() < switch_prob:
                current_regime = 1 - current_regime  # Switch regime
            regime_switches.append(current_regime)

        # Generate volatility levels based on regime
        vol_regimes = []
        for regime in regime_switches:
            if regime == 0:  # Low volatility regime
                vol = np.random.uniform(0.015, 0.035)  # 1.5-3.5% daily vol
            else:  # High volatility regime
                vol = np.random.uniform(0.045, 0.085)  # 4.5-8.5% daily vol
            vol_regimes.append(vol)

        # Generate returns with strong volatility clustering and momentum
        returns = []
        momentum_factor = 0

        for i, vol in enumerate(vol_regimes):
            if i == 0:
                ret = np.random.normal(0.001, vol)
            else:
                # Add momentum persistence (stronger in high vol regimes)
                persistence = 0.15 * returns[-1] if regime_switches[i] == 1 else 0.08 * returns[-1]

                # Add volatility momentum (volatility tends to cluster)
                vol_momentum = 0.1 * (vol_regimes[i] - vol_regimes[i - 1]) if i > 0 else 0

                ret = (
                    persistence
                    + vol_momentum * np.random.normal(0, 0.5)
                    + np.random.normal(0.001, vol)
                )

            returns.append(ret)

        # Create prices from returns
        price = 45000  # Starting price
        prices = [price]
        for ret in returns[1:]:
            price = price * (1 + ret)
            prices.append(price)

        # Create realistic OHLCV data with volatility-dependent ranges
        data = pd.DataFrame(
            {
                "date": dates,
                "open": prices,
                "high": [
                    p * (1 + abs(np.random.normal(0, vol_regimes[i] * 0.6)))
                    for i, p in enumerate(prices)
                ],
                "low": [
                    p * (1 - abs(np.random.normal(0, vol_regimes[i] * 0.6)))
                    for i, p in enumerate(prices)
                ],
                "close": prices,
                "volume": [
                    np.random.uniform(1000, 5000)
                    if regime_switches[i] == 0
                    else np.random.uniform(3000, 15000)  # Higher volume in high vol regimes
                    for i in range(days)
                ],
            }
        )

        data.set_index("date", inplace=True)
        self.data = data

        print(f"✅ Generated {len(data)} candles with enhanced volatility clustering")
        print(f"    Low vol periods: {regime_switches.count(0)} days")
        print(f"    High vol periods: {regime_switches.count(1)} days")
        return True

    def extract_v2_0_features(self, dataframe):
        """Extract V2.0 features."""
        print("🔧 Extracting V2.0 features...")
        df = dataframe.copy()

        # V2.0 Feature set (9 features)
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

        for col in v2_0_columns:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).ffill().bfill()

        print(f"✅ Extracted {len(v2_0_columns)} V2.0 features")
        return df[v2_0_columns].dropna()

    def extract_v2_1_features(self, dataframe):
        """Extract V2.1 features (feature bloat version)."""
        print("🔧 Extracting V2.1 features...")
        df = dataframe.copy()

        # V2.1 Feature set (16 features - the bloated version)
        # Multi-scale volatility
        atr_7 = self._calculate_atr(df, 7)
        atr_14 = self._calculate_atr(df, 14)
        atr_21 = self._calculate_atr(df, 21)

        df["vol_ratio_7_14"] = atr_7 / (atr_14 + 1e-8)
        df["vol_ratio_14_21"] = atr_14 / (atr_21 + 1e-8)
        df["atr_momentum_3d"] = atr_14.diff(3) / (atr_14.shift(3) + 1e-8)

        # PCA compression
        if ML_AVAILABLE:
            atr_7_pct = atr_7 / df["close"]
            atr_14_pct = atr_14 / df["close"]
            atr_21_pct = atr_21 / df["close"]

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
            df["vol_pca_1"] = atr_14 / df["close"]
            df["vol_pca_2"] = df["vol_ratio_14_21"]

        # Enhanced momentum
        df["daily_rsi"] = self._calculate_rsi(df, 14)
        macd_data = self._calculate_macd(df)
        df["daily_macd_hist"] = macd_data["histogram"]
        df["macd_momentum"] = macd_data["histogram"].diff(3)

        # Breakout persistence (complex feature)
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

        # Volatility regime
        vol_regime = atr_14 / atr_14.rolling(50).mean()
        df["vol_regime_cluster"] = np.where(vol_regime > 1.2, 1, 0)
        df["vol_trend"] = (atr_14.rolling(5).mean() / atr_14.rolling(20).mean()) - 1

        # Keep some V2.0 features
        daily_sma_20 = df["close"].rolling(20).mean()
        daily_sma_50 = df["close"].rolling(50).mean()
        df["daily_sma_slope"] = (daily_sma_20 - daily_sma_50) / daily_sma_50

        # Volume
        if "volume" in df.columns:
            daily_obv = self._calculate_obv(df)
            df["obv_momentum"] = daily_obv.diff(5)
        else:
            df["obv_momentum"] = 0

        # Lag features (the problematic ones)
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

        for col in v2_1_columns:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).ffill().bfill()

        print(f"✅ Extracted {len(v2_1_columns)} V2.1 features")
        return df[v2_1_columns].dropna()

    def extract_v2_2_features(self, dataframe):
        """Extract V2.2 focused volatility features."""
        print("🔧 Extracting V2.2 volatility-focused features...")
        df = dataframe.copy()

        # V2.2 Feature set (7 features - focused approach)

        # Core volatility features (5)
        atr_7 = self._calculate_atr(df, 7)
        atr_14 = self._calculate_atr(df, 14)
        atr_21 = self._calculate_atr(df, 21)

        # 1. Primary volatility factor (PCA of multi-scale ATRs)
        if ML_AVAILABLE:
            atr_7_pct = atr_7 / df["close"]
            atr_14_pct = atr_14 / df["close"]
            atr_21_pct = atr_21 / df["close"]

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
                pca = PCA(n_components=1)  # Only primary component
                atr_pca = pca.fit_transform(atr_scaled)
                df["vol_pca_1"] = atr_pca[:, 0]
            else:
                df["vol_pca_1"] = atr_14_pct
        else:
            df["vol_pca_1"] = atr_14 / df["close"]

        # 2. Volatility ratio
        df["vol_ratio_7_14"] = atr_7 / (atr_14 + 1e-8)

        # 3. ATR momentum
        df["atr_momentum_3d"] = atr_14.diff(3) / (atr_14.shift(3) + 1e-8)

        # 4. Volatility clustering regime
        vol_regime_ratio = atr_14 / atr_14.rolling(50).mean()
        df["vol_regime_cluster"] = np.where(vol_regime_ratio > 1.2, 1, 0)

        # 5. Volatility trend
        df["vol_trend"] = (atr_14.rolling(5).mean() / atr_14.rolling(20).mean()) - 1

        # Selective momentum features (2)
        # 6. Daily RSI
        df["daily_rsi"] = self._calculate_rsi(df, 14)

        # 7. MACD momentum
        macd_data = self._calculate_macd(df)
        df["macd_momentum"] = macd_data["histogram"].diff(3)

        v2_2_columns = [
            "vol_pca_1",
            "vol_ratio_7_14",
            "atr_momentum_3d",
            "vol_regime_cluster",
            "vol_trend",
            "daily_rsi",
            "macd_momentum",
        ]

        for col in v2_2_columns:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan).ffill().bfill()

        print(f"✅ Extracted {len(v2_2_columns)} V2.2 focused features")
        return df[v2_2_columns].dropna()

    def create_targets(self, dataframe, threshold=0.02):
        """Create targets with 2% threshold (V2.2 setting)."""
        print(f"🎯 Creating targets with {threshold * 100}% threshold...")

        forward_return = dataframe["close"].shift(-3) / dataframe["close"] - 1
        targets = np.where(forward_return > threshold, 1, 0)
        actual_direction = np.where(forward_return > 0, 1, 0)

        return pd.Series(targets, index=dataframe.index), pd.Series(
            actual_direction, index=dataframe.index
        )

    def compare_all_versions(self):
        """Compare V2.0 vs V2.1 vs V2.2 performance."""
        print("\n" + "=" * 70)
        print("🔍 COMPREHENSIVE VERSION COMPARISON: V2.0 vs V2.1 vs V2.2")
        print("=" * 70)

        # Extract all feature sets
        v2_0_features = self.extract_v2_0_features(self.data)
        v2_1_features = self.extract_v2_1_features(self.data)
        v2_2_features = self.extract_v2_2_features(self.data)
        conservative_targets, actual_targets = self.create_targets(self.data)

        # Align data
        common_index = (
            v2_0_features.index.intersection(v2_1_features.index)
            .intersection(v2_2_features.index)
            .intersection(conservative_targets.index)
        )

        v2_0_features = v2_0_features.loc[common_index]
        v2_1_features = v2_1_features.loc[common_index]
        v2_2_features = v2_2_features.loc[common_index]
        conservative_targets = conservative_targets.loc[common_index]
        actual_targets = actual_targets.loc[common_index]

        print(f"\n📊 Comparison Overview:")
        print(f"  V2.0 Features: {len(v2_0_features.columns)} (baseline)")
        print(f"  V2.1 Features: {len(v2_1_features.columns)} (feature bloat)")
        print(f"  V2.2 Features: {len(v2_2_features.columns)} (volatility focused)")
        print(f"  Common samples: {len(common_index)}")

        if not ML_AVAILABLE:
            print("⚠️ ML libraries not available, skipping model comparison")
            return

        # Test all versions
        results = {}
        versions = [("V2.0", v2_0_features), ("V2.1", v2_1_features), ("V2.2", v2_2_features)]

        for version, features in versions:
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

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train = y_conservative.iloc[train_idx]
                y_test_conservative = y_conservative.iloc[test_idx]
                y_test_actual = y_actual.iloc[test_idx]

                # Train model with version-appropriate settings
                if version == "V2.2":
                    # Optimized for fewer features
                    model = lgb.LGBMClassifier(
                        n_estimators=180,
                        learning_rate=0.07,
                        max_depth=5,
                        subsample=0.8,
                        colsample_bytree=0.9,
                        random_state=42,
                        verbose=-1,
                    )
                else:
                    # Standard settings
                    model = lgb.LGBMClassifier(
                        n_estimators=150,
                        learning_rate=0.08,
                        max_depth=6,
                        random_state=42,
                        verbose=-1,
                    )

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Calculate accuracies
                acc_conservative = accuracy_score(y_test_conservative, y_pred)
                acc_actual = accuracy_score(y_test_actual, y_pred)

                conservative_accuracies.append(acc_conservative)
                actual_accuracies.append(acc_actual)

            results[version] = {
                "conservative_accuracy_mean": np.mean(conservative_accuracies),
                "conservative_accuracy_std": np.std(conservative_accuracies),
                "actual_accuracy_mean": np.mean(actual_accuracies),
                "actual_accuracy_std": np.std(actual_accuracies),
                "feature_count": len(features.columns),
            }

            print(
                f"  Conservative Accuracy: {results[version]['conservative_accuracy_mean']:.3f} ± {results[version]['conservative_accuracy_std']:.3f}"
            )
            print(
                f"  Actual Accuracy: {results[version]['actual_accuracy_mean']:.3f} ± {results[version]['actual_accuracy_std']:.3f}"
            )

        # Compare results
        print(f"\n📈 COMPREHENSIVE COMPARISON RESULTS:")
        print("=" * 70)

        # Rank by conservative accuracy
        ranked_versions = sorted(
            results.items(), key=lambda x: x[1]["conservative_accuracy_mean"], reverse=True
        )

        for i, (version, result) in enumerate(ranked_versions):
            print(
                f"  {i + 1}. {version:4s}: {result['conservative_accuracy_mean']:.3f} ± {result['conservative_accuracy_std']:.3f} ({result['feature_count']} features)"
            )

        # Calculate improvements
        if "V2.0" in results and "V2.2" in results:
            v2_2_improvement = (
                results["V2.2"]["conservative_accuracy_mean"]
                - results["V2.0"]["conservative_accuracy_mean"]
            )
            print(
                f"\n🎯 V2.2 vs V2.0 Improvement: {v2_2_improvement:+.3f} ({v2_2_improvement * 100:+.1f}%)"
            )

        if "V2.1" in results and "V2.2" in results:
            v2_2_vs_v2_1 = (
                results["V2.2"]["conservative_accuracy_mean"]
                - results["V2.1"]["conservative_accuracy_mean"]
            )
            print(f"🎯 V2.2 vs V2.1 Improvement: {v2_2_vs_v2_1:+.3f} ({v2_2_vs_v2_1 * 100:+.1f}%)")

        return results

    def generate_v2_2_report(self):
        """Generate comprehensive V2.2 validation report."""
        print("\n" + "=" * 80)
        print("📋 V2.2 VOLATILITY-FOCUSED STRATEGY VALIDATION REPORT")
        print("=" * 80)

        # Load data
        if not self.load_data():
            print("❌ Failed to load data. Exiting.")
            return

        # Run comprehensive comparison
        comparison_results = self.compare_all_versions()

        # Generate executive summary
        print("\n" + "=" * 80)
        print("📈 V2.2 VALIDATION SUMMARY")
        print("=" * 80)

        if comparison_results:
            print(f"\n🎯 KEY VALIDATION RESULTS:")

            # Check if V2.2 is the best performer
            best_version = max(
                comparison_results.items(), key=lambda x: x[1]["conservative_accuracy_mean"]
            )

            if best_version[0] == "V2.2":
                print(
                    f"  ✅ V2.2 WINS: Best performer with {best_version[1]['conservative_accuracy_mean']:.3f} accuracy"
                )
                print(f"  🎯 Volatility-focused approach validated!")
            else:
                print(f"  ⚠️ V2.2 did not win - {best_version[0]} performed best")

            # Feature efficiency analysis
            if "V2.1" in comparison_results and "V2.2" in comparison_results:
                v2_1_efficiency = (
                    comparison_results["V2.1"]["conservative_accuracy_mean"]
                    / comparison_results["V2.1"]["feature_count"]
                )
                v2_2_efficiency = (
                    comparison_results["V2.2"]["conservative_accuracy_mean"]
                    / comparison_results["V2.2"]["feature_count"]
                )

                print(f"\n🔬 FEATURE EFFICIENCY:")
                print(f"  V2.1: {v2_1_efficiency:.4f} accuracy per feature")
                print(f"  V2.2: {v2_2_efficiency:.4f} accuracy per feature")

                if v2_2_efficiency > v2_1_efficiency:
                    print(f"  ✅ V2.2 is more feature-efficient!")
                else:
                    print(f"  ⚠️ V2.2 efficiency needs improvement")

        print(f"\n📋 FINAL RECOMMENDATIONS:")
        recommendations = []

        if comparison_results:
            best_acc = max(r["conservative_accuracy_mean"] for r in comparison_results.values())

            if best_acc > 0.60:
                recommendations.append("✅ Strategy shows strong predictive power (>60%)")
            elif best_acc > 0.55:
                recommendations.append("⚡ Strategy shows moderate predictive power (>55%)")
            else:
                recommendations.append("⚠️ Continue feature engineering - accuracy too low")

            if "V2.2" in comparison_results:
                v2_2_acc = comparison_results["V2.2"]["conservative_accuracy_mean"]
                if v2_2_acc == best_acc:
                    recommendations.append(
                        "🎯 V2.2 volatility approach is optimal - proceed to backtesting"
                    )
                elif v2_2_acc > 0.55:
                    recommendations.append("⚡ V2.2 shows promise - consider minor refinements")
                else:
                    recommendations.append("🔬 V2.2 needs significant improvement")

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
    """Run V2.2 validation analysis."""
    print("🚀 Starting V2.2 Volatility-Focused Strategy Validation")
    print("=" * 60)

    # Initialize analyzer
    analyzer = StrategyV2_2_Analyzer()

    # Run comprehensive validation
    analyzer.generate_v2_2_report()

    print("\n🎉 V2.2 Validation complete! Review the results above.")


if __name__ == "__main__":
    main()
