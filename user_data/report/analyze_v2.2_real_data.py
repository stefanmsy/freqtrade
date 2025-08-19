"""
V2.2 Real Data Analysis - Using Actual BTC Price History

This script addresses the user's valid concern about using synthetic data.
Now we test V2.2 strategy features with REAL BTC historical price data.

Key Analysis:
1. Load actual BTC/USDT futures data from Binance
2. Extract V2.2 volatility-focused features on real data
3. Test real feature importance and predictive power
4. Validate V2.2 approach against actual market patterns
5. Compare with synthetic data results to see if we were overfitting

This will give us the REAL story about feature effectiveness.
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

from pathlib import Path
import sys
import os

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


class RealDataV2_2_Analyzer:
    """V2.2 Strategy analyzer using REAL BTC historical data."""

    def __init__(self):
        self.data = None
        self.results = {}

    def load_real_btc_data(self):
        """Load actual BTC historical data from FreqTrade data files."""
        print("📊 Loading REAL BTC historical data...")

        # Try to load futures data first (preferred for V2.2)
        futures_path = Path("../data/binance/futures/BTC_USDT_USDT-1d-futures.feather")
        spot_path = Path("../data/binance/BTC_USDT-1d.feather")

        try:
            if futures_path.exists():
                print(f"Loading futures data: {futures_path}")
                self.data = pd.read_feather(futures_path)
                data_source = "Futures"
            elif spot_path.exists():
                print(f"Loading spot data: {spot_path}")
                self.data = pd.read_feather(spot_path)
                data_source = "Spot"
            else:
                print("❌ No BTC data files found!")
                return False

            # Ensure we have the required columns
            required_cols = ["date", "open", "high", "low", "close", "volume"]
            if not all(col in self.data.columns for col in required_cols):
                print(f"❌ Missing required columns. Available: {list(self.data.columns)}")
                return False

            # Set date as index
            if "date" in self.data.columns:
                self.data.set_index("date", inplace=True)

            # Sort by date to ensure chronological order
            self.data.sort_index(inplace=True)

            print(f"✅ Loaded {len(self.data)} candles of REAL BTC {data_source} data")
            print(f"   Date range: {self.data.index[0]} to {self.data.index[-1]}")
            print(
                f"   Price range: ${self.data['close'].min():.0f} - ${self.data['close'].max():.0f}"
            )

            # Basic data quality checks
            print(f"   Data quality:")
            print(f"     - Missing values: {self.data.isnull().sum().sum()}")
            print(f"     - Zero volume days: {(self.data['volume'] == 0).sum()}")
            print(
                f"     - Price volatility: {self.data['close'].pct_change().std() * 100:.2f}% daily"
            )

            return True

        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False

    def extract_v2_2_features_real(self):
        """Extract V2.2 features from REAL BTC data."""
        print("🔧 Extracting V2.2 features from REAL data...")

        if self.data is None:
            print("❌ No data loaded")
            return None

        df = self.data.copy()

        # ================================================================
        # V2.2 CORE VOLATILITY FEATURES (5 features)
        # ================================================================

        # Multi-scale ATRs
        atr_7 = self._calculate_atr(df, 7)
        atr_14 = self._calculate_atr(df, 14)
        atr_21 = self._calculate_atr(df, 21)

        # Normalized ATRs
        atr_7_pct = atr_7 / df["close"]
        atr_14_pct = atr_14 / df["close"]
        atr_21_pct = atr_21 / df["close"]

        # 1. Primary volatility factor (PCA)
        if ML_AVAILABLE:
            atr_features = (
                pd.DataFrame(
                    {"atr_7_pct": atr_7_pct, "atr_14_pct": atr_14_pct, "atr_21_pct": atr_21_pct}
                )
                .fillna(method="ffill")
                .fillna(0)
            )

            if len(atr_features.dropna()) > 50:
                try:
                    scaler = StandardScaler()
                    atr_scaled = scaler.fit_transform(atr_features)
                    pca = PCA(n_components=1)
                    atr_pca = pca.fit_transform(atr_scaled)
                    df["vol_pca_1"] = atr_pca[:, 0]
                    print(f"   ✅ PCA explained variance: {pca.explained_variance_ratio_[0]:.3f}")
                except Exception as e:
                    print(f"   ⚠️ PCA failed, using fallback: {e}")
                    df["vol_pca_1"] = atr_14_pct
            else:
                df["vol_pca_1"] = atr_14_pct
        else:
            df["vol_pca_1"] = atr_14_pct

        # 2. Volatility ratio
        df["vol_ratio_7_14"] = atr_7 / (atr_14 + 1e-8)

        # 3. ATR momentum
        df["atr_momentum_3d"] = atr_14.diff(3) / (atr_14.shift(3) + 1e-8)

        # 4. Volatility clustering regime
        vol_regime_ratio = atr_14 / atr_14.rolling(50).mean()
        df["vol_regime_cluster"] = np.where(vol_regime_ratio > 1.2, 1, 0)

        # 5. Volatility trend
        df["vol_trend"] = (atr_14.rolling(5).mean() / atr_14.rolling(20).mean()) - 1

        # ================================================================
        # V2.2 SELECTIVE MOMENTUM FEATURES (2 features)
        # ================================================================

        # 6. Daily RSI
        df["daily_rsi"] = self._calculate_rsi(df, 14)

        # 7. MACD momentum
        macd_data = self._calculate_macd(df)
        df["macd_momentum"] = macd_data["histogram"].diff(3)

        # Clean features
        feature_columns = [
            "vol_pca_1",
            "vol_ratio_7_14",
            "atr_momentum_3d",
            "vol_regime_cluster",
            "vol_trend",
            "daily_rsi",
            "macd_momentum",
        ]

        for col in feature_columns:
            if col in df.columns:
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
                df[col] = df[col].ffill().bfill()

        print(f"✅ Extracted {len(feature_columns)} V2.2 features from REAL data")
        return df[feature_columns].dropna()

    def create_real_targets(self, threshold=0.02):
        """Create prediction targets from REAL price movements."""
        print(f"🎯 Creating targets from REAL price movements (threshold: {threshold * 100}%)")

        if self.data is None:
            return None, None

        # Calculate actual 3-day forward returns
        forward_return = self.data["close"].shift(-3) / self.data["close"] - 1

        # Conservative targets (2% threshold)
        conservative_targets = np.where(forward_return > threshold, 1, 0)

        # Actual direction targets (any positive movement)
        actual_targets = np.where(forward_return > 0, 1, 0)

        print(f"   Target distribution:")
        print(f"     Conservative (>{threshold * 100}%): {conservative_targets.mean():.3f}")
        print(f"     Actual direction (>0%): {actual_targets.mean():.3f}")

        return (
            pd.Series(conservative_targets, index=self.data.index),
            pd.Series(actual_targets, index=self.data.index),
        )

    def analyze_real_feature_importance(self):
        """Analyze feature importance using REAL BTC data."""
        print("\n" + "=" * 60)
        print("🔍 REAL DATA FEATURE IMPORTANCE ANALYSIS")
        print("=" * 60)

        if not ML_AVAILABLE:
            print("⚠️ ML libraries not available")
            return {}

        # Extract features and targets from real data
        features = self.extract_v2_2_features_real()
        conservative_targets, actual_targets = self.create_real_targets()

        if features is None or conservative_targets is None:
            print("❌ Failed to extract features or targets")
            return {}

        # Align data
        common_index = features.index.intersection(conservative_targets.index)
        X = features.loc[common_index]
        y_conservative = conservative_targets.loc[common_index]
        y_actual = actual_targets.loc[common_index]

        # Remove any remaining NaN values
        mask = ~(X.isna().any(axis=1) | y_conservative.isna() | y_actual.isna())
        X = X[mask]
        y_conservative = y_conservative[mask]
        y_actual = y_actual[mask]

        print(f"📊 Analysis dataset: {len(X)} samples")
        print(f"   Features: {list(X.columns)}")

        if len(X) < 100:
            print(f"⚠️ Insufficient data: {len(X)} samples")
            return {}

        # Train model and get feature importance
        model = lgb.LGBMClassifier(
            n_estimators=180,
            learning_rate=0.07,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.9,
            random_state=42,
            verbose=-1,
        )

        # Use time series split for realistic evaluation
        tscv = TimeSeriesSplit(n_splits=3)
        conservative_accuracies = []
        actual_accuracies = []
        feature_importances = []

        for i, (train_idx, test_idx) in enumerate(tscv.split(X)):
            print(f"   🧪 Testing fold {i + 1}/3...")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train = y_conservative.iloc[train_idx]
            y_test_conservative = y_conservative.iloc[test_idx]
            y_test_actual = y_actual.iloc[test_idx]

            # Train model
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

            print(f"     Conservative accuracy: {acc_conservative:.3f}")
            print(f"     Actual direction accuracy: {acc_actual:.3f}")

        # Calculate average feature importance
        avg_importance = {}
        for feature in X.columns:
            importance_values = [fi[feature] for fi in feature_importances]
            avg_importance[feature] = {
                "mean": np.mean(importance_values),
                "std": np.std(importance_values),
                "min": min(importance_values),
                "max": max(importance_values),
            }

        # Results summary
        results = {
            "conservative_accuracy_mean": np.mean(conservative_accuracies),
            "conservative_accuracy_std": np.std(conservative_accuracies),
            "actual_accuracy_mean": np.mean(actual_accuracies),
            "actual_accuracy_std": np.std(actual_accuracies),
            "feature_importance": avg_importance,
            "sample_count": len(X),
            "date_range": f"{X.index[0]} to {X.index[-1]}",
        }

        print(f"\n📈 REAL DATA RESULTS:")
        print(
            f"   Conservative accuracy: {results['conservative_accuracy_mean']:.3f} ± {results['conservative_accuracy_std']:.3f}"
        )
        print(
            f"   Actual direction accuracy: {results['actual_accuracy_mean']:.3f} ± {results['actual_accuracy_std']:.3f}"
        )

        # Feature importance ranking
        print(f"\n🏆 REAL DATA FEATURE IMPORTANCE:")
        sorted_features = sorted(avg_importance.items(), key=lambda x: x[1]["mean"], reverse=True)
        for i, (feature, stats) in enumerate(sorted_features):
            print(f"   {i + 1}. {feature:20s}: {stats['mean']:6.1f} ± {stats['std']:4.1f}")

        return results

    def generate_real_vs_synthetic_comparison(self):
        """Compare real data results with previous synthetic data results."""
        print("\n" + "=" * 70)
        print("📊 REAL vs SYNTHETIC DATA COMPARISON")
        print("=" * 70)

        real_results = self.analyze_real_feature_importance()

        if not real_results:
            print("❌ No real data results to compare")
            return

        print(f"\n🎯 KEY FINDINGS:")

        # Accuracy comparison
        real_acc = real_results["conservative_accuracy_mean"]
        print(f"   Real Data Accuracy: {real_acc:.3f}")
        print(f"   Previous Synthetic: ~0.503 (from earlier analysis)")

        accuracy_diff = real_acc - 0.503
        print(f"   Difference: {accuracy_diff:+.3f}")

        if accuracy_diff > 0.02:
            print(f"   ✅ Real data performs BETTER - synthetic underestimated!")
        elif accuracy_diff < -0.02:
            print(f"   ⚠️ Real data performs WORSE - synthetic was overly optimistic!")
        else:
            print(f"   ⚡ Real and synthetic results are similar - good validation!")

        # Feature importance insights
        if "feature_importance" in real_results:
            top_features = sorted(
                real_results["feature_importance"].items(), key=lambda x: x[1]["mean"], reverse=True
            )[:3]

            print(f"\n🔍 TOP REAL FEATURES:")
            for feature, stats in top_features:
                print(f"     {feature}: {stats['mean']:.1f}")

        print(f"\n📋 CONCLUSIONS:")
        if real_acc > 0.55:
            print(f"   ✅ V2.2 strategy shows REAL predictive power!")
            print(f"   ✅ Volatility-focused approach validated with actual data")
        elif real_acc > 0.52:
            print(f"   ⚡ V2.2 shows modest real-world performance")
            print(f"   ⚡ May be usable but needs refinement")
        else:
            print(f"   ⚠️ V2.2 struggles with real data")
            print(f"   ⚠️ Synthetic results were misleading")

        print(f"\n🎯 NEXT STEPS:")
        if real_acc > 0.55:
            print(f"   🚀 Proceed to full backtesting with FreqAI")
            print(f"   🚀 V2.2 ready for live trading consideration")
        else:
            print(f"   🔬 Continue feature engineering with real data focus")
            print(f"   🔬 Consider V2.3 with real-data optimized features")

    def run_comprehensive_real_analysis(self):
        """Run complete analysis using real BTC data."""
        print("🚀 V2.2 REAL DATA VALIDATION")
        print("=" * 60)

        # Load real data
        if not self.load_real_btc_data():
            print("❌ Failed to load real data. Exiting.")
            return

        # Run analysis
        self.generate_real_vs_synthetic_comparison()

        print("\n🎉 Real data analysis complete!")
        print("This gives us the TRUE picture of V2.2 strategy effectiveness.")

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


def main():
    """Run real data analysis to validate V2.2 strategy."""
    print("🎯 V2.2 Strategy Validation with REAL BTC Data")
    print("This addresses the synthetic data limitation!")
    print()

    analyzer = RealDataV2_2_Analyzer()
    analyzer.run_comprehensive_real_analysis()


if __name__ == "__main__":
    main()
