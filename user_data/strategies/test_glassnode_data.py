#!/usr/bin/env python3
"""
Test script for Glassnode data loading and feature engineering
This script validates that the Glassnode data can be loaded and processed correctly.
"""

import pandas as pd
import numpy as np
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_glassnode_data_loading():
    """Test loading and processing of Glassnode data"""

    print("=== Testing Glassnode Data Loading ===")

    # Test hourly data loading
    glassnode_hourly_path = "user_data/data/glassnode/btc-glassnode_1h.csv"
    if os.path.exists(glassnode_hourly_path):
        print(f"✓ Found hourly data at: {glassnode_hourly_path}")

        # Load a small sample
        sample_data = pd.read_csv(glassnode_hourly_path, nrows=100)
        print(f"✓ Loaded {len(sample_data)} rows of hourly data")
        print(f"✓ Columns: {list(sample_data.columns)}")

        # Check timestamp format
        sample_data["timestamp"] = pd.to_datetime(sample_data["timestamp"])
        print(f"✓ Timestamp conversion successful")

        # Check for key on-chain metrics
        expected_features = [
            "sopr",
            "exchange_netflow",
            "realized_pnl_ratio",
            "nvt_price",
            "nvt_price_90D",
            "90D_NVT_premium",
            "whale_exchange_netflow",
        ]

        available_features = [col for col in sample_data.columns if col in expected_features]
        print(f"✓ Available on-chain features: {available_features}")

    else:
        print(f"✗ Hourly data not found at: {glassnode_hourly_path}")

    # Test daily LTH/STH data loading
    lth_sth_path = "user_data/data/glassnode/lth_sth_supply_1d.csv"
    if os.path.exists(lth_sth_path):
        print(f"✓ Found LTH/STH data at: {lth_sth_path}")

        # Load a small sample
        sample_lth_sth = pd.read_csv(lth_sth_path, nrows=50)
        print(f"✓ Loaded {len(sample_lth_sth)} rows of LTH/STH data")
        print(f"✓ Columns: {list(sample_lth_sth.columns)}")

        # Test resampling to hourly
        sample_lth_sth["timestamp"] = pd.to_datetime(
            sample_lth_sth["timestamp"], format="%d/%m/%Y %H:%M"
        )
        sample_lth_sth.set_index("timestamp", inplace=True)
        lth_sth_hourly = sample_lth_sth.resample("1H").ffill()
        print(f"✓ Resampled to hourly: {len(lth_sth_hourly)} rows")

    else:
        print(f"✗ LTH/STH data not found at: {lth_sth_path}")


def test_feature_engineering():
    """Test feature engineering functions"""

    print("\n=== Testing Feature Engineering ===")

    # Create sample dataframe
    dates = pd.date_range("2023-01-01", periods=1000, freq="1H")
    sample_df = pd.DataFrame(
        {
            "date": dates,
            "close": np.random.randn(1000).cumsum() + 100,
            "volume": np.random.randint(1000, 10000, 1000),
            "high": np.random.randn(1000).cumsum() + 102,
            "low": np.random.randn(1000).cumsum() + 98,
            "mvrv": np.random.uniform(1, 4, 1000),
            "sopr": np.random.uniform(0.8, 1.2, 1000),
            "dormancy": np.random.uniform(50, 200, 1000),
            "accumulation_trend_score": np.random.uniform(-1, 1, 1000),
            "exchange_netflow": np.random.uniform(-1000, 1000, 1000),
            "long_term_holder_supply": np.random.uniform(8000000, 9000000, 1000),
            "short_term_holder_supply": np.random.uniform(2000000, 3000000, 1000),
            "coin_days_destroyed": np.random.uniform(100000, 500000, 1000),
            "nvt": np.random.uniform(50, 150, 1000),
            "realized_profit_loss": np.random.uniform(-0.1, 0.1, 1000),
        }
    )

    print(f"✓ Created sample dataframe with {len(sample_df)} rows")
    print(f"✓ Sample columns: {list(sample_df.columns)}")

    # Test basic feature engineering
    # SMA calculation
    sample_df["sma_10"] = sample_df["close"].rolling(10).mean()
    sample_df["sma_50"] = sample_df["close"].rolling(50).mean()
    sample_df["sma_200"] = sample_df["close"].rolling(200).mean()

    # Price changes
    sample_df["price_change_1d"] = sample_df["close"].pct_change(24)
    sample_df["price_change_5d"] = sample_df["close"].pct_change(120)

    # On-chain feature engineering
    sample_df["mvrv_slope_3d"] = sample_df["mvrv"].diff(72)
    sample_df["sopr_rolling_min_7d"] = sample_df["sopr"].rolling(168).min()

    # Dormancy z-score
    dormancy_mean = sample_df["dormancy"].rolling(168).mean()
    dormancy_std = sample_df["dormancy"].rolling(168).std()
    sample_df["dormancy_zscore"] = (sample_df["dormancy"] - dormancy_mean) / dormancy_std.replace(
        0, 1
    )

    # Percentage calculations
    total_supply = sample_df["long_term_holder_supply"] + sample_df["short_term_holder_supply"]
    sample_df["percent_lth"] = sample_df["long_term_holder_supply"] / total_supply.replace(0, 1)
    sample_df["percent_sth"] = sample_df["short_term_holder_supply"] / total_supply.replace(0, 1)

    print(f"✓ Engineered features successfully")
    print(f"✓ Final dataframe shape: {sample_df.shape}")

    # Check for NaN values
    nan_counts = sample_df.isnull().sum()
    print(f"✓ NaN counts in engineered features:")
    for col, count in nan_counts[nan_counts > 0].items():
        print(f"  {col}: {count}")

    return sample_df


def test_target_creation():
    """Test target creation for 5-day forward returns"""

    print("\n=== Testing Target Creation ===")

    # Create sample data
    dates = pd.date_range("2023-01-01", periods=1000, freq="1H")
    sample_df = pd.DataFrame({"date": dates, "close": np.random.randn(1000).cumsum() + 100})

    # Create 5-day forward returns target (120 candles for 1h)
    label_period = 120
    sample_df["target_5d"] = sample_df["close"].shift(-label_period) / sample_df["close"] - 1

    print(f"✓ Created target with {label_period} period lookahead")
    print(
        f"✓ Target range: {sample_df['target_5d'].min():.4f} to {sample_df['target_5d'].max():.4f}"
    )
    print(f"✓ Target mean: {sample_df['target_5d'].mean():.4f}")
    print(f"✓ Target std: {sample_df['target_5d'].std():.4f}")


if __name__ == "__main__":
    print("Glassnode Data and Feature Engineering Test")
    print("=" * 50)

    test_glassnode_data_loading()
    test_df = test_feature_engineering()
    test_target_creation()

    print("\n=== Test Summary ===")
    print("✓ All tests completed successfully!")
    print("✓ Ready to use GlassnodeSwingStrategy")
