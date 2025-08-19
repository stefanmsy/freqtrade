"""
Comprehensive Analysis Script for BTCPriceDirectionStrategyV2 (Daily-Only)

This script addresses the key concerns about strategy effectiveness:
1. Real accuracy vs overfitting
2. Feature importance and consistency
3. Correlation analysis to identify redundant features
4. Comparison with actual price movements
5. Performance across different time periods

Key Analysis Goals:
- Verify if focused features are actually predictive
- Check consistency of feature importance across periods
- Identify any remaining overfitting issues
- Establish baseline for V2.x iterations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Try importing FreqTrade and ML libraries
try:
    from freqtrade.data.btanalysis import load_backtest_data, load_backtest_metadata
    from freqtrade.data.history import load_pair_history
    from freqtrade.configuration import Configuration
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import TimeSeriesSplit

    FREQTRADE_AVAILABLE = True
except ImportError as e:
    print(f"FreqTrade/ML libraries not available: {e}")
    FREQTRADE_AVAILABLE = False


class StrategyV2Analyzer:
    """Comprehensive analyzer for V2 strategy performance and features."""

    def __init__(self, strategy_name="BTCPriceDirectionStrategyV2", config_path=None):
        self.strategy_name = strategy_name
        self.config_path = config_path
        self.results = {}
        self.feature_analysis = {}

    def load_data(self, data_path="user_data/data/binance", pair="BTC/USDT", timeframe="1d"):
        """Load historical data for analysis."""
        print("📊 Loading historical data...")

        if FREQTRADE_AVAILABLE:
            try:
                # Load data using FreqTrade
                self.data = load_pair_history(
                    datadir=Path(data_path),
                    timeframe=timeframe,
                    pair=pair.replace("/", "_").replace(":", "_"),
                    data_format="feather",
                )
                print(
                    f"✅ Loaded {len(self.data)} candles from {self.data.index[0]} to {self.data.index[-1]}"
                )
                return True
            except Exception as e:
                print(f"❌ FreqTrade data loading failed: {e}")

        # Fallback: Try to load from any available source
        return self._load_fallback_data()

    def _load_fallback_data(self):
        """Fallback data loading if FreqTrade data not available."""
        print("🔄 Attempting fallback data loading...")

        # Try to find any CSV/feather files in user_data
        data_files = list(Path("user_data").rglob("*BTC*"))
        if data_files:
            print(f"Found data files: {data_files}")
            # Use the first available file
            try:
                if data_files[0].suffix == ".feather":
                    self.data = pd.read_feather(data_files[0])
                elif data_files[0].suffix == ".csv":
                    self.data = pd.read_csv(data_files[0], index_col=0, parse_dates=True)

                if len(self.data) > 100:
                    print(f"✅ Loaded {len(self.data)} candles from fallback source")
                    return True
            except Exception as e:
                print(f"❌ Fallback loading failed: {e}")

        # Generate synthetic data for demonstration
        print("🎲 Generating synthetic data for demonstration...")
        self.data = self._generate_synthetic_data()
        return True

    def _generate_synthetic_data(self, days=1000):
        """Generate synthetic BTC data for testing."""
        dates = pd.date_range(start="2022-01-01", periods=days, freq="D")

        # Generate realistic BTC price movements
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.04, days)  # Daily returns
        price = 45000  # Starting price
        prices = [price]

        for ret in returns[1:]:
            price = price * (1 + ret)
            prices.append(price)

        # Create OHLCV data
        data = pd.DataFrame(
            {
                "date": dates,
                "open": prices,
                "high": [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
                "low": [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
                "close": prices,
                "volume": np.random.uniform(1000, 10000, days),
            }
        )

        data.set_index("date", inplace=True)
        print(f"✅ Generated {len(data)} synthetic candles")
        return data

    def extract_v2_features(self, dataframe):
        """Extract the same features used in V2 strategy."""
        print("🔧 Extracting V2 strategy features...")

        df = dataframe.copy()

        # ================================================================
        # DAILY TREND & REGIME FEATURES (2 features)
        # ================================================================

        # 1. Daily 20-SMA vs 50-SMA crossover (binary bull/bear)
        daily_sma_20 = df["close"].rolling(20).mean()
        daily_sma_50 = df["close"].rolling(50).mean()
        df["daily_sma_regime"] = np.where(daily_sma_20 > daily_sma_50, 1, 0)

        # 2. Daily SMA slope: (SMA20 – SMA50) / SMA50
        df["daily_sma_slope"] = (daily_sma_20 - daily_sma_50) / daily_sma_50

        # ================================================================
        # DAILY MOMENTUM FEATURES (2 features)
        # ================================================================

        # 3. Daily 14-period RSI
        df["daily_rsi"] = self._calculate_rsi(df, 14)

        # 4. Daily MACD histogram
        macd_data = self._calculate_macd(df)
        df["daily_macd_hist"] = macd_data["histogram"]

        # ================================================================
        # DAILY BREAKOUT STRENGTH FEATURES (2 features)
        # ================================================================

        # 5. Days since last 20-bar high
        daily_high_20 = df["high"].rolling(20).max()
        days_since_high = []
        for i in range(len(df)):
            if i < 20:
                days_since_high.append(np.nan)
            else:
                current_high = df["high"].iloc[i]
                if current_high >= daily_high_20.iloc[i]:
                    days_since_high.append(0)
                else:
                    count = 1
                    for j in range(i - 1, max(0, i - 20), -1):
                        if df["high"].iloc[j] >= daily_high_20.iloc[j]:
                            break
                        count += 1
                    days_since_high.append(min(count, 20))

        df["days_since_high"] = days_since_high

        # 6. Daily close / 20-bar high ratio
        df["close_high_ratio"] = df["close"] / daily_high_20

        # ================================================================
        # DAILY VOLATILITY CONTEXT FEATURES (2 features)
        # ================================================================

        # 7. Daily ATR14 / daily close (percent)
        daily_atr = self._calculate_atr(df, 14)
        df["daily_atr_percent"] = daily_atr / df["close"]

        # 8. Daily ATR14 raw
        df["daily_atr_raw"] = daily_atr

        # ================================================================
        # DAILY VOLUME PRESSURE FEATURES (1 feature)
        # ================================================================

        # 9. Daily OBV momentum (ΔOBV over 5 days)
        if "volume" in df.columns:
            daily_obv = self._calculate_obv(df)
            df["obv_momentum"] = daily_obv.diff(5)
        else:
            df["obv_momentum"] = 0

        # ================================================================
        # DAILY VOLATILITY REGIME FEATURES (1 feature)
        # ================================================================

        # 10. Daily vol14 divided by its 50-bar rolling mean
        daily_returns = df["close"].pct_change()
        daily_vol = daily_returns.rolling(14).std()
        daily_vol_mean = daily_vol.rolling(50).mean()
        df["vol_regime"] = daily_vol / daily_vol_mean

        # Clean features
        feature_columns = [
            "daily_sma_regime",
            "daily_sma_slope",
            "daily_rsi",
            "daily_macd_hist",
            "days_since_high",
            "close_high_ratio",
            "daily_atr_percent",
            "daily_atr_raw",
            "obv_momentum",
            "vol_regime",
        ]

        for col in feature_columns:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].ffill().bfill()

        print(f"✅ Extracted {len(feature_columns)} V2 features")
        return df[feature_columns].dropna()

    def create_targets(self, dataframe, threshold=0.02):
        """Create 3-day forward targets matching V2 strategy."""
        print(f"🎯 Creating targets with {threshold * 100}% threshold...")

        # Calculate 3-day forward return
        forward_return = dataframe["close"].shift(-3) / dataframe["close"] - 1

        # Create binary classification target
        targets = np.where(forward_return > threshold, 1, 0)  # 1=up, 0=down

        # Also get actual direction for accuracy calculation
        actual_direction = np.where(forward_return > 0, 1, 0)  # Actual up/down

        return pd.Series(targets, index=dataframe.index), pd.Series(
            actual_direction, index=dataframe.index
        )

    def analyze_feature_importance(self, features, targets):
        """Analyze feature importance using LightGBM."""
        print("🔍 Analyzing feature importance...")

        if not FREQTRADE_AVAILABLE:
            print("⚠️ ML libraries not available, skipping feature importance analysis")
            return {}

        try:
            # Remove NaN values
            mask = ~(features.isna().any(axis=1) | targets.isna())
            X = features[mask]
            y = targets[mask]

            if len(X) < 100:
                print(f"⚠️ Insufficient data for analysis: {len(X)} samples")
                return {}

            # Train LightGBM model
            model = lgb.LGBMClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbose=-1
            )

            model.fit(X, y)

            # Get feature importance
            importance = dict(zip(X.columns, model.feature_importances_))

            print("📊 Feature Importance Rankings:")
            for i, (feature, imp) in enumerate(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)
            ):
                print(f"  {i + 1:2d}. {feature:20s} : {imp:6.1f}")

            return {"importance": importance, "model": model, "feature_data": X, "target_data": y}

        except Exception as e:
            print(f"❌ Feature importance analysis failed: {e}")
            return {}

    def analyze_feature_consistency(self, features, targets, n_splits=5):
        """Analyze feature importance consistency across time periods."""
        print(f"📈 Analyzing feature consistency across {n_splits} time periods...")

        if not FREQTRADE_AVAILABLE:
            print("⚠️ ML libraries not available, skipping consistency analysis")
            return {}

        try:
            # Remove NaN values
            mask = ~(features.isna().any(axis=1) | targets.isna())
            X = features[mask]
            y = targets[mask]

            if len(X) < 200:
                print(f"⚠️ Insufficient data for consistency analysis: {len(X)} samples")
                return {}

            # Time series split for consistency analysis
            tscv = TimeSeriesSplit(n_splits=n_splits)

            importance_across_splits = []
            accuracy_across_splits = []

            for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
                print(f"  📊 Analyzing split {i + 1}/{n_splits}...")

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Train model on this split
                model = lgb.LGBMClassifier(
                    n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbose=-1
                )

                model.fit(X_train, y_train)

                # Get predictions and accuracy
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                accuracy_across_splits.append(accuracy)

                # Store feature importance for this split
                importance = dict(zip(X.columns, model.feature_importances_))
                importance_across_splits.append(importance)

                print(f"    Accuracy: {accuracy:.3f}")

            # Analyze consistency
            consistency_analysis = self._analyze_importance_consistency(importance_across_splits)

            print(f"\n📊 Consistency Analysis Results:")
            print(
                f"  Mean Accuracy: {np.mean(accuracy_across_splits):.3f} ± {np.std(accuracy_across_splits):.3f}"
            )
            print(
                f"  Accuracy Range: {min(accuracy_across_splits):.3f} - {max(accuracy_across_splits):.3f}"
            )

            return {
                "importance_across_splits": importance_across_splits,
                "accuracy_across_splits": accuracy_across_splits,
                "consistency_metrics": consistency_analysis,
                "mean_accuracy": np.mean(accuracy_across_splits),
                "accuracy_std": np.std(accuracy_across_splits),
            }

        except Exception as e:
            print(f"❌ Consistency analysis failed: {e}")
            return {}

    def _analyze_importance_consistency(self, importance_across_splits):
        """Analyze how consistent feature importance is across time periods."""

        # Get all features
        all_features = list(importance_across_splits[0].keys())

        # Calculate coefficient of variation for each feature
        consistency_metrics = {}

        for feature in all_features:
            importances = [split[feature] for split in importance_across_splits]
            mean_imp = np.mean(importances)
            std_imp = np.std(importances)
            cv = std_imp / (mean_imp + 1e-8)  # Coefficient of variation

            consistency_metrics[feature] = {
                "mean_importance": mean_imp,
                "std_importance": std_imp,
                "coefficient_of_variation": cv,
                "min_importance": min(importances),
                "max_importance": max(importances),
            }

        # Sort by consistency (lower CV = more consistent)
        sorted_features = sorted(
            consistency_metrics.items(), key=lambda x: x[1]["coefficient_of_variation"]
        )

        print(f"\n🎯 Feature Consistency Rankings (lower CV = more consistent):")
        for i, (feature, metrics) in enumerate(sorted_features):
            print(
                f"  {i + 1:2d}. {feature:20s} : CV={metrics['coefficient_of_variation']:.3f}, Mean={metrics['mean_importance']:.1f}"
            )

        return consistency_metrics

    def analyze_correlations(self, features):
        """Analyze feature correlations to identify redundancy."""
        print("🔗 Analyzing feature correlations...")

        corr_matrix = features.corr()

        # Find highly correlated pairs
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

        print(f"\n🚨 High Correlation Pairs (>0.8):")
        if high_corr_pairs:
            for pair in high_corr_pairs:
                print(f"  {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")
        else:
            print("  ✅ No highly correlated feature pairs found")

        return {"correlation_matrix": corr_matrix, "high_correlation_pairs": high_corr_pairs}

    def calculate_real_accuracy(self, features, conservative_targets, actual_targets):
        """Calculate accuracy against both conservative and actual price movements."""
        print("📊 Calculating real accuracy metrics...")

        if not FREQTRADE_AVAILABLE:
            print("⚠️ ML libraries not available, skipping accuracy calculation")
            return {}

        try:
            # Remove NaN values
            mask = ~(
                features.isna().any(axis=1) | conservative_targets.isna() | actual_targets.isna()
            )
            X = features[mask]
            y_conservative = conservative_targets[mask]
            y_actual = actual_targets[mask]

            if len(X) < 100:
                print(f"⚠️ Insufficient data for accuracy analysis: {len(X)} samples")
                return {}

            # Train model on conservative targets (like V2 strategy)
            model = lgb.LGBMClassifier(
                n_estimators=150, learning_rate=0.08, max_depth=6, random_state=42, verbose=-1
            )

            # Use time series split for realistic evaluation
            tscv = TimeSeriesSplit(n_splits=3)
            conservative_accuracies = []
            actual_accuracies = []

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train = y_conservative.iloc[train_idx]
                y_test_conservative = y_conservative.iloc[test_idx]
                y_test_actual = y_actual.iloc[test_idx]

                # Train and predict
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Calculate accuracies
                acc_conservative = accuracy_score(y_test_conservative, y_pred)
                acc_actual = accuracy_score(y_test_actual, y_pred)

                conservative_accuracies.append(acc_conservative)
                actual_accuracies.append(acc_actual)

            results = {
                "conservative_accuracy_mean": np.mean(conservative_accuracies),
                "conservative_accuracy_std": np.std(conservative_accuracies),
                "actual_accuracy_mean": np.mean(actual_accuracies),
                "actual_accuracy_std": np.std(actual_accuracies),
                "conservative_accuracies": conservative_accuracies,
                "actual_accuracies": actual_accuracies,
            }

            print(f"📊 Accuracy Results:")
            print(
                f"  Conservative Targets (2% threshold): {results['conservative_accuracy_mean']:.3f} ± {results['conservative_accuracy_std']:.3f}"
            )
            print(
                f"  Actual Direction: {results['actual_accuracy_mean']:.3f} ± {results['actual_accuracy_std']:.3f}"
            )

            return results

        except Exception as e:
            print(f"❌ Accuracy calculation failed: {e}")
            return {}

    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report."""
        print("\n" + "=" * 60)
        print("📋 COMPREHENSIVE V2.0 STRATEGY ANALYSIS REPORT")
        print("=" * 60)

        if not hasattr(self, "data"):
            print("❌ No data loaded. Run load_data() first.")
            return

        # Extract features and targets
        features = self.extract_v2_features(self.data)
        conservative_targets, actual_targets = self.create_targets(self.data, threshold=0.02)

        # Align data
        common_index = features.index.intersection(conservative_targets.index).intersection(
            actual_targets.index
        )
        features = features.loc[common_index]
        conservative_targets = conservative_targets.loc[common_index]
        actual_targets = actual_targets.loc[common_index]

        print(f"\n📊 Dataset Overview:")
        print(f"  Total samples: {len(features)}")
        print(f"  Date range: {features.index[0].date()} to {features.index[-1].date()}")
        print(f"  Features: {len(features.columns)}")

        # Feature importance analysis
        self.feature_analysis["importance"] = self.analyze_feature_importance(
            features, conservative_targets
        )

        # Consistency analysis
        self.feature_analysis["consistency"] = self.analyze_feature_consistency(
            features, conservative_targets
        )

        # Correlation analysis
        self.feature_analysis["correlations"] = self.analyze_correlations(features)

        # Real accuracy analysis
        self.results["accuracy"] = self.calculate_real_accuracy(
            features, conservative_targets, actual_targets
        )

        # Generate summary
        self._generate_summary()

    def _generate_summary(self):
        """Generate executive summary of findings."""
        print("\n" + "=" * 60)
        print("📈 EXECUTIVE SUMMARY")
        print("=" * 60)

        # Accuracy summary
        if "accuracy" in self.results and self.results["accuracy"]:
            acc = self.results["accuracy"]
            print(f"\n🎯 ACCURACY ANALYSIS:")
            print(
                f"  • Conservative (2% threshold): {acc['conservative_accuracy_mean']:.1%} ± {acc['conservative_accuracy_std']:.1%}"
            )
            print(
                f"  • Actual direction: {acc['actual_accuracy_mean']:.1%} ± {acc['actual_accuracy_std']:.1%}"
            )

            if acc["conservative_accuracy_mean"] > 0.55:
                print(f"  ✅ Model shows predictive power above random (50%)")
            else:
                print(f"  ⚠️ Model accuracy close to random - may need improvement")

        # Consistency summary
        if "consistency" in self.feature_analysis and self.feature_analysis["consistency"]:
            cons = self.feature_analysis["consistency"]
            print(f"\n🔄 CONSISTENCY ANALYSIS:")
            print(f"  • Mean accuracy across periods: {cons['mean_accuracy']:.1%}")
            print(f"  • Accuracy standard deviation: {cons['accuracy_std']:.1%}")

            if cons["accuracy_std"] < 0.05:
                print(f"  ✅ Consistent performance across time periods")
            else:
                print(f"  ⚠️ High variance - may indicate overfitting or regime changes")

        # Feature correlation summary
        if "correlations" in self.feature_analysis and self.feature_analysis["correlations"]:
            corr = self.feature_analysis["correlations"]
            if len(corr["high_correlation_pairs"]) == 0:
                print(f"\n🔗 FEATURE CORRELATIONS:")
                print(f"  ✅ No highly correlated features found - good feature diversity")
            else:
                print(f"\n🔗 FEATURE CORRELATIONS:")
                print(f"  ⚠️ Found {len(corr['high_correlation_pairs'])} highly correlated pairs")
                print(f"  Consider removing redundant features")

        print(f"\n📋 RECOMMENDATIONS:")

        # Make recommendations based on analysis
        recommendations = []

        if "accuracy" in self.results and self.results["accuracy"]:
            acc = self.results["accuracy"]
            if acc["conservative_accuracy_mean"] < 0.55:
                recommendations.append("• Accuracy too low - consider different features or model")
            if acc["conservative_accuracy_mean"] - acc["actual_accuracy_mean"] > 0.05:
                recommendations.append(
                    "• Large gap between conservative and actual accuracy - review threshold"
                )

        if "consistency" in self.feature_analysis and self.feature_analysis["consistency"]:
            cons = self.feature_analysis["consistency"]
            if cons["accuracy_std"] > 0.05:
                recommendations.append(
                    "• High variance across periods - investigate regime changes"
                )

        if "correlations" in self.feature_analysis and self.feature_analysis["correlations"]:
            corr = self.feature_analysis["correlations"]
            if len(corr["high_correlation_pairs"]) > 0:
                recommendations.append("• Remove highly correlated features to reduce noise")

        if not recommendations:
            recommendations.append("✅ V2.0 strategy shows good baseline characteristics")
            recommendations.append(
                "✅ Ready to proceed with V2.1 optimizations or V3.0 multi-timeframe"
            )

        for rec in recommendations:
            print(f"  {rec}")

        print("\n" + "=" * 60)

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
    """Run comprehensive V2.0 analysis."""
    print("🚀 Starting V2.0 Strategy Comprehensive Analysis")
    print("=" * 60)

    # Initialize analyzer
    analyzer = StrategyV2Analyzer()

    # Load data
    if not analyzer.load_data():
        print("❌ Failed to load data. Exiting.")
        return

    # Run comprehensive analysis
    analyzer.generate_comprehensive_report()

    print("\n🎉 Analysis complete! Review the results above.")

    # Save results
    try:
        import json

        results_file = "user_data/report/v2_0_analysis_results.json"

        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Prepare results for saving (only serializable parts)
        save_results = {
            "accuracy": analyzer.results.get("accuracy", {}),
            "feature_analysis_summary": {
                "consistency_mean_accuracy": analyzer.feature_analysis.get("consistency", {}).get(
                    "mean_accuracy"
                ),
                "consistency_accuracy_std": analyzer.feature_analysis.get("consistency", {}).get(
                    "accuracy_std"
                ),
                "high_correlation_pairs_count": len(
                    analyzer.feature_analysis.get("correlations", {}).get(
                        "high_correlation_pairs", []
                    )
                ),
            },
        }

        # Convert numpy types
        save_results = json.loads(json.dumps(save_results, default=convert_numpy))

        with open(results_file, "w") as f:
            json.dump(save_results, f, indent=2)

        print(f"💾 Results saved to: {results_file}")

    except Exception as e:
        print(f"⚠️ Could not save results: {e}")


if __name__ == "__main__":
    main()
