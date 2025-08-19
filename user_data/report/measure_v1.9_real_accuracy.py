#!/usr/bin/env python3
"""
Measure real accuracy of v1.9 confidence threshold model against actual price data
"""

import pandas as pd
import numpy as np
from pathlib import Path


def measure_v1_9_real_accuracy():
    """Measure accuracy of v1.9 confidence threshold model against actual future price movements"""

    print("=== V1.9 CONFIDENCE THRESHOLD REAL ACCURACY MEASUREMENT ===")

    # Load prediction files
    model_id = "btc_price_direction_v1.9_confidence_threshold"
    pred_dir = Path(f"user_data/models/{model_id}/backtesting_predictions")
    pred_files = list(pred_dir.glob("*.feather"))

    all_predictions = []
    for file_path in sorted(pred_files):
        df = pd.read_feather(file_path)
        all_predictions.append(df)

    combined_df = pd.concat(all_predictions, ignore_index=True)
    combined_df["date"] = pd.to_datetime(combined_df["date"])

    # Filter for confident predictions
    confident_df = combined_df[combined_df["do_predict"] == 1].copy()

    print(f"Total confident predictions: {len(confident_df)}")

    # Load actual price data to calculate real future returns
    print("Loading price data...")
    try:
        price_file = Path("user_data/data/binance/BTC_USDT-1d.feather")
        if price_file.exists():
            price_data = pd.read_feather(price_file)
            price_data["date"] = pd.to_datetime(price_data["date"])
            price_data = price_data.set_index("date")
        else:
            print("Price data not found!")
            return None, None
    except Exception as e:
        print(f"Error loading price data: {e}")
        return None, None

    # Calculate actual future returns (3-day forward)
    price_data["future_return_3d"] = price_data["close"].shift(-3) / price_data["close"] - 1

    # Merge predictions with price data
    confident_df = confident_df.merge(
        price_data[["close", "future_return_3d"]], left_on="date", right_index=True, how="left"
    )

    # Remove rows where we don't have future data (last 3 days)
    confident_df = confident_df.dropna(subset=["future_return_3d"])

    print(f"Predictions with future data: {len(confident_df)}")

    # Calculate actual target based on 1% threshold (same as training)
    threshold = 0.01  # 1%
    confident_df["actual_target"] = np.where(
        confident_df["future_return_3d"] > threshold,
        "up",
        np.where(confident_df["future_return_3d"] < -threshold, "down", "sideways"),
    )

    print(f"\nActual target distribution (based on real future returns):")
    print(confident_df["actual_target"].value_counts())

    # Calculate confidence metrics
    confident_df["max_prob"] = confident_df[["up", "down", "sideways"]].max(axis=1)
    confident_df["second_max_prob"] = confident_df[["up", "down", "sideways"]].apply(
        lambda x: x.nlargest(2).iloc[1], axis=1
    )
    confident_df["prob_diff"] = confident_df["max_prob"] - confident_df["second_max_prob"]

    # Original classification (always pick highest)
    confident_df["original_prediction"] = confident_df[["up", "down", "sideways"]].idxmax(axis=1)

    # Apply confidence threshold (15% as implemented in strategy)
    confidence_threshold = 0.15
    confident_df["prediction_with_threshold"] = confident_df.apply(
        lambda row: "sideways"
        if row["prob_diff"] <= confidence_threshold
        else row["original_prediction"],
        axis=1,
    )

    # Calculate original accuracy (no threshold)
    original_correct = (confident_df["original_prediction"] == confident_df["actual_target"]).sum()
    original_accuracy = original_correct / len(confident_df) * 100

    print(f"\n=== ORIGINAL ACCURACY (always pick highest) ===")
    print(f"Total predictions: {len(confident_df)}")
    print(f"Correct predictions: {original_correct}")
    print(f"Accuracy: {original_accuracy:.1f}%")

    # Calculate accuracy for non-sideways predictions only (what the strategy actually trades)
    non_sideways = confident_df[confident_df["prediction_with_threshold"] != "sideways"].copy()

    if len(non_sideways) > 0:
        # Calculate accuracy for high-confidence predictions
        correct_predictions = (
            non_sideways["prediction_with_threshold"] == non_sideways["actual_target"]
        ).sum()
        accuracy = correct_predictions / len(non_sideways) * 100

        print(f"\n=== CONFIDENCE THRESHOLD ACCURACY (15%) ===")
        print(f"High-confidence predictions: {len(non_sideways)}")
        print(f"Correct predictions: {correct_predictions}")
        print(f"Accuracy: {accuracy:.1f}%")

        # Detailed accuracy breakdown
        print(f"\nAccuracy breakdown:")
        for pred_class in ["up", "down"]:
            class_predictions = non_sideways[
                non_sideways["prediction_with_threshold"] == pred_class
            ]
            if len(class_predictions) > 0:
                class_correct = (class_predictions["actual_target"] == pred_class).sum()
                class_accuracy = class_correct / len(class_predictions) * 100
                print(
                    f"  {pred_class}: {class_correct}/{len(class_predictions)} ({class_accuracy:.1f}%)"
                )

        # Compare with backtest results
        print(f"\n=== COMPARISON WITH BACKTEST RESULTS ===")
        print(f"Backtest trades: 4")
        print(f"Backtest win rate: 75% (3 wins, 1 loss)")
        print(f"High-confidence predictions: {len(non_sideways)}")
        print(f"High-confidence accuracy: {accuracy:.1f}%")

        # Show confidence distribution
        print(f"\n=== CONFIDENCE DISTRIBUTION ===")
        print(f"Mean confidence: {non_sideways['prob_diff'].mean():.3f}")
        print(f"Min confidence: {non_sideways['prob_diff'].min():.3f}")
        print(f"Max confidence: {non_sideways['prob_diff'].max():.3f}")

        # Show sample high-confidence predictions
        print(f"\n=== SAMPLE HIGH-CONFIDENCE PREDICTIONS ===")
        sample_cols = [
            "date",
            "up",
            "down",
            "sideways",
            "prob_diff",
            "prediction_with_threshold",
            "actual_target",
        ]
        print(non_sideways[sample_cols].head(10))

    else:
        print("No high-confidence predictions found!")

    return confident_df, non_sideways


if __name__ == "__main__":
    measure_v1_9_real_accuracy()
