#!/usr/bin/env python3
"""
Analyze predictions using confidence thresholds to determine sideways classification
"""

import pandas as pd
import numpy as np
from pathlib import Path


def analyze_confidence_threshold_approach():
    """Analyze predictions using different confidence thresholds"""

    print("=== CONFIDENCE THRESHOLD ANALYSIS ===")

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

    # Calculate max and second max probabilities
    confident_df["max_prob"] = confident_df[["up", "down", "sideways"]].max(axis=1)
    confident_df["second_max_prob"] = confident_df[["up", "down", "sideways"]].apply(
        lambda x: x.nlargest(2).iloc[1], axis=1
    )
    confident_df["prob_diff"] = confident_df["max_prob"] - confident_df["second_max_prob"]

    # Original classification (always pick highest)
    confident_df["original_prediction"] = confident_df[["up", "down", "sideways"]].idxmax(axis=1)

    print(f"Total confident predictions: {len(confident_df)}")
    print(f"Original classification:")
    print(confident_df["original_prediction"].value_counts())

    # Test different confidence thresholds
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25]

    print(f"\n=== CONFIDENCE THRESHOLD ANALYSIS ===")

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

        # Calculate how many became sideways
        became_sideways = (confident_df[f"prediction_threshold_{threshold}"] == "sideways").sum()
        print(f"Became sideways: {became_sideways} ({became_sideways / total * 100:.1f}%)")

        # Show confidence distribution for non-sideways predictions
        non_sideways = confident_df[confident_df[f"prediction_threshold_{threshold}"] != "sideways"]
        if len(non_sideways) > 0:
            print(
                f"Non-sideways confidence: {non_sideways['prob_diff'].mean():.3f} mean, {non_sideways['prob_diff'].min():.3f} min"
            )

    # Show examples of predictions that would become sideways
    print(f"\n=== EXAMPLES OF PREDICTIONS THAT WOULD BECOME SIDEWAYS ===")

    for threshold in [0.05, 0.10, 0.15]:
        would_be_sideways = confident_df[confident_df["prob_diff"] <= threshold].copy()
        would_be_sideways["would_be_sideways"] = True

        print(
            f"\nThreshold {threshold:.2f}: {len(would_be_sideways)} predictions would become sideways"
        )

        if len(would_be_sideways) > 0:
            # Show sample cases
            sample_cols = ["date", "up", "down", "sideways", "prob_diff", "original_prediction"]
            print(would_be_sideways[sample_cols].head(5))

    # Analyze the closest prediction (1357)
    print(f"\n=== ANALYSIS OF PREDICTION 1357 ===")
    closest_prediction = confident_df.loc[1357]
    print(f"Date: {closest_prediction['date']}")
    print(f"Up: {closest_prediction['up']:.6f}")
    print(f"Down: {closest_prediction['down']:.6f}")
    print(f"Sideways: {closest_prediction['sideways']:.6f}")
    print(f"Probability difference: {closest_prediction['prob_diff']:.6f}")
    print(f"Original prediction: {closest_prediction['original_prediction']}")

    # What would different thresholds predict?
    for threshold in [0.05, 0.10, 0.15, 0.20]:
        if closest_prediction["prob_diff"] <= threshold:
            prediction = "sideways"
        else:
            prediction = closest_prediction["original_prediction"]
        print(f"Threshold {threshold:.2f}: {prediction}")

    return confident_df


if __name__ == "__main__":
    analyze_confidence_threshold_approach()
