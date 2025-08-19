#!/usr/bin/env python3
"""
Measure real accuracy by comparing predictions against actual future price movements
"""

import pandas as pd
import numpy as np
from pathlib import Path


def measure_real_accuracy():
    """Measure accuracy against actual future price movements"""

    print("=== REAL ACCURACY MEASUREMENT ===")

    # Load prediction files
    model_id = "btc_price_direction_v1.8_volatility_3class"
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
        # Try to load from user_data/data
        price_file = Path("user_data/data/binance/BTC_USDT-1d.feather")
        if price_file.exists():
            price_data = pd.read_feather(price_file)
            price_data["date"] = pd.to_datetime(price_data["date"])
            price_data = price_data.set_index("date")
        else:
            print(
                "Price data not found. Please run: freqtrade download-data --pairs BTC/USDT --timeframe 1d"
            )
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

    # Calculate original accuracy
    original_correct = (confident_df["original_prediction"] == confident_df["actual_target"]).sum()
    original_accuracy = original_correct / len(confident_df) * 100

    print(f"\n=== ORIGINAL ACCURACY (always pick highest) ===")
    print(f"Total predictions: {len(confident_df)}")
    print(f"Correct predictions: {original_correct}")
    print(f"Accuracy: {original_accuracy:.1f}%")

    # Test different confidence thresholds
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]

    print(f"\n=== CONFIDENCE THRESHOLD ACCURACY ===")

    results = []

    for threshold in thresholds:
        print(f"\n--- Threshold: {threshold:.2f} ({threshold * 100:.0f}%) ---")

        # Apply confidence threshold
        confident_df[f"prediction_threshold_{threshold}"] = confident_df.apply(
            lambda row: "sideways" if row["prob_diff"] <= threshold else row["original_prediction"],
            axis=1,
        )

        # Count predictions
        counts = confident_df[f"prediction_threshold_{threshold}"].value_counts()
        total = len(confident_df)

        print(f"Predictions:")
        for cls, count in counts.items():
            pct = count / total * 100
            print(f"  {cls}: {count} ({pct:.1f}%)")

        # Calculate accuracy for non-sideways predictions only
        non_sideways = confident_df[
            confident_df[f"prediction_threshold_{threshold}"] != "sideways"
        ].copy()

        if len(non_sideways) > 0:
            # Calculate accuracy
            correct_predictions = (
                non_sideways[f"prediction_threshold_{threshold}"] == non_sideways["actual_target"]
            ).sum()
            accuracy = correct_predictions / len(non_sideways) * 100

            print(f"Non-sideways predictions: {len(non_sideways)}")
            print(f"Correct predictions: {correct_predictions}")
            print(f"Accuracy: {accuracy:.1f}%")

            # Detailed accuracy breakdown
            print(f"Accuracy breakdown:")
            for pred_class in ["up", "down"]:
                class_predictions = non_sideways[
                    non_sideways[f"prediction_threshold_{threshold}"] == pred_class
                ]
                if len(class_predictions) > 0:
                    class_correct = (class_predictions["actual_target"] == pred_class).sum()
                    class_accuracy = class_correct / len(class_predictions) * 100
                    print(
                        f"  {pred_class}: {class_correct}/{len(class_predictions)} ({class_accuracy:.1f}%)"
                    )

            # Store results
            results.append(
                {
                    "threshold": threshold,
                    "total_predictions": total,
                    "non_sideways": len(non_sideways),
                    "sideways": total - len(non_sideways),
                    "correct": correct_predictions,
                    "accuracy": accuracy,
                    "avg_confidence": non_sideways["prob_diff"].mean(),
                }
            )
        else:
            print("No non-sideways predictions!")
            results.append(
                {
                    "threshold": threshold,
                    "total_predictions": total,
                    "non_sideways": 0,
                    "sideways": total,
                    "correct": 0,
                    "accuracy": 0,
                    "avg_confidence": 0,
                }
            )

    # Show results summary
    print(f"\n=== SUMMARY TABLE ===")
    print(
        f"{'Threshold':<10} {'Non-Sideways':<12} {'Sideways':<10} {'Accuracy':<10} {'Avg Conf':<10}"
    )
    print("-" * 60)
    print(
        f"{'Original':<10} {len(confident_df):<12} {0:<10} {original_accuracy:<10.1f} {'N/A':<10}"
    )

    for result in results:
        print(
            f"{result['threshold']:<10.2f} {result['non_sideways']:<12} {result['sideways']:<10} {result['accuracy']:<10.1f} {result['avg_confidence']:<10.3f}"
        )

    # Find optimal threshold
    if results:
        best_result = max(results, key=lambda x: x["accuracy"])
        print(
            f"\nBest accuracy: {best_result['accuracy']:.1f}% at threshold {best_result['threshold']:.2f}"
        )
        print(
            f"Trade-off: {best_result['non_sideways']} trades vs {best_result['sideways']} sideways"
        )

    return confident_df, results


if __name__ == "__main__":
    measure_real_accuracy()
