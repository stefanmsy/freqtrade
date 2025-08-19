#!/usr/bin/env python3
"""
Analyze data sufficiency for walk-forward validation and test new sideways threshold
"""

import pandas as pd
import numpy as np
from pathlib import Path


def analyze_data_sufficiency():
    """Check if we have sufficient data for proper walk-forward validation"""

    print("=== DATA SUFFICIENCY ANALYSIS ===")

    # Load BTC data
    btc_data_path = Path("user_data/data/binance/BTC_USDT-1d.feather")

    if not btc_data_path.exists():
        print("❌ BTC data file not found!")
        return

    btc_df = pd.read_feather(btc_data_path)
    btc_df["date"] = pd.to_datetime(btc_df["date"])

    print(f"📊 BTC Data Available:")
    print(f"   Total days: {len(btc_df)}")
    print(f"   From: {btc_df['date'].min().date()}")
    print(f"   To: {btc_df['date'].max().date()}")

    # Current FreqAI configuration
    train_period = 365  # days
    backtest_period = 180  # days
    min_required = train_period + backtest_period  # 545 days

    print(f"\n⚙️ Current FreqAI Configuration:")
    print(f"   Train period: {train_period} days")
    print(f"   Backtest period: {backtest_period} days")
    print(f"   Minimum required: {min_required} days")
    print(f"   Available: {len(btc_df)} days")

    # Check sufficiency
    is_sufficient = len(btc_df) >= min_required
    print(f"   Sufficient for walk-forward: {'✅ YES' if is_sufficient else '❌ NO'}")

    if is_sufficient:
        # Calculate possible walk-forward windows
        available_data = len(btc_df)

        # FreqAI does rolling windows: each window slides by backtest_period
        num_complete_windows = (available_data - train_period) // backtest_period

        print(f"\n📈 Walk-Forward Analysis:")
        print(f"   Possible complete windows: {num_complete_windows}")

        if num_complete_windows > 0:
            print(f"   Window structure:")
            for i in range(min(num_complete_windows, 5)):  # Show first 5 windows
                train_start_idx = i * backtest_period
                train_end_idx = train_start_idx + train_period
                test_start_idx = train_end_idx
                test_end_idx = test_start_idx + backtest_period

                if test_end_idx <= len(btc_df):
                    train_start_date = btc_df.iloc[train_start_idx]["date"].date()
                    train_end_date = btc_df.iloc[train_end_idx - 1]["date"].date()
                    test_start_date = btc_df.iloc[test_start_idx]["date"].date()
                    test_end_date = btc_df.iloc[test_end_idx - 1]["date"].date()

                    print(
                        f"     Window {i + 1}: Train {train_start_date} to {train_end_date} → Test {test_start_date} to {test_end_date}"
                    )

            if num_complete_windows > 5:
                print(f"     ... and {num_complete_windows - 5} more windows")
    else:
        shortage = min_required - len(btc_df)
        print(f"\n❌ Insufficient Data:")
        print(f"   Need {shortage} more days")
        print(f"   Suggest reducing train_period to {len(btc_df) - backtest_period} days")


def test_sideways_threshold():
    """Test impact of different sideways thresholds"""

    print("\n=== SIDEWAYS THRESHOLD ANALYSIS ===")

    # Load actual BTC data
    btc_data_path = Path("user_data/data/binance/BTC_USDT-1d.feather")
    if not btc_data_path.exists():
        print("❌ BTC data not found for threshold analysis")
        return

    btc_df = pd.read_feather(btc_data_path)
    btc_df["date"] = pd.to_datetime(btc_df["date"])
    btc_df = btc_df.set_index("date").sort_index()

    # Calculate 3-day forward returns for last 1000 days
    recent_data = btc_df.tail(1000).copy()
    returns_3d = recent_data["close"].shift(-3) / recent_data["close"] - 1
    returns_3d = returns_3d.dropna()

    print(f"📊 3-Day Forward Returns Analysis (last {len(returns_3d)} days):")
    print(f"   Mean: {returns_3d.mean():.4f} ({returns_3d.mean() * 100:.2f}%)")
    print(f"   Std: {returns_3d.std():.4f} ({returns_3d.std() * 100:.2f}%)")
    print(f"   Min: {returns_3d.min():.4f} ({returns_3d.min() * 100:.2f}%)")
    print(f"   Max: {returns_3d.max():.4f} ({returns_3d.max() * 100:.2f}%)")

    # Test different thresholds
    thresholds = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03]

    print(f"\n🎯 Threshold Impact Analysis:")
    print(f"   Threshold   Up%    Down%   Sideways%")
    print(f"   ---------   ----   -----   ---------")

    for threshold in thresholds:
        up_count = (returns_3d > threshold).sum()
        down_count = (returns_3d < -threshold).sum()
        sideways_count = ((returns_3d >= -threshold) & (returns_3d <= threshold)).sum()

        total = len(returns_3d)
        up_pct = up_count / total * 100
        down_pct = down_count / total * 100
        sideways_pct = sideways_count / total * 100

        print(f"   {threshold:.3f}       {up_pct:4.1f}%   {down_pct:4.1f}%    {sideways_pct:4.1f}%")

    print(f"\n💡 Recommendations:")
    print(
        f"   • Current 0.5% threshold → {((returns_3d >= -0.005) & (returns_3d <= 0.005)).sum() / len(returns_3d) * 100:.1f}% sideways"
    )
    print(
        f"   • Proposed 2.0% threshold → {((returns_3d >= -0.02) & (returns_3d <= 0.02)).sum() / len(returns_3d) * 100:.1f}% sideways"
    )
    print(f"   • 2.0% gives more balanced classes for ML")


if __name__ == "__main__":
    analyze_data_sufficiency()
    test_sideways_threshold()
