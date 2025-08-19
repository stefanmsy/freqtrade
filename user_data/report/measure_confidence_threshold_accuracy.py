#!/usr/bin/env python3
"""
Measure accuracy of confidence threshold approach against actual targets in test period
"""

import pandas as pd
import numpy as np
from pathlib import Path


def measure_confidence_threshold_accuracy():
    """Measure accuracy of different confidence thresholds against actual targets"""

    print("=== CONFIDENCE THRESHOLD ACCURACY ANALYSIS ===")

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

    # Filter for confident predictions (do_predict == 1)
    confident_df = combined_df[combined_df["do_predict"] == 1].copy()

    print(f"Total confident predictions: {len(confident_df)}")

    # Calculate confidence metrics
    confident_df["max_prob"] = confident_df[["up", "down", "sideways"]].max(axis=1)
    confident_df["second_max_prob"] = confident_df[["up", "down", "sideways"]].apply(
        lambda x: x.nlargest(2).iloc[1], axis=1
    )
    confident_df["prob_diff"] = confident_df["max_prob"] - confident_df["second_max_prob"]

    # Original classification (always pick highest)
    confident_df["original_prediction"] = confident_df[["up", "down", "sideways"]].idxmax(axis=1)

    # Get actual targets from the prediction files
    confident_df["actual_target"] = confident_df["&s-direction"]

    print(f"\nActual target distribution:")
    print(confident_df["actual_target"].value_counts())

    # Test different confidence thresholds
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]

    print(f"\n=== ACCURACY ANALYSIS BY CONFIDENCE THRESHOLD ===")

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

    # Compare with original accuracy (no threshold)
    print(f"\n=== COMPARISON WITH ORIGINAL APPROACH ===")

    # Original accuracy (always pick highest)
    original_correct = (confident_df["original_prediction"] == confident_df["actual_target"]).sum()
    original_accuracy = original_correct / len(confident_df) * 100

    print(f"Original approach (always pick highest):")
    print(f"Total predictions: {len(confident_df)}")
    print(f"Correct predictions: {original_correct}")
    print(f"Accuracy: {original_accuracy:.1f}%")

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

    # Analyze what happens to accuracy as we increase threshold
    print(f"\n=== ACCURACY TRADE-OFF ANALYSIS ===")
    print(f"As threshold increases:")
    print(f"- Fewer trades (lower non-sideways count)")
    print(f"- Higher accuracy (more confident predictions)")
    print(f"- More sideways predictions (avoiding uncertain cases)")

    return confident_df, results


if __name__ == "__main__":
    measure_confidence_threshold_accuracy()
