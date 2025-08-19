#!/usr/bin/env python3
"""
Analysis script for BTC Price Direction Strategy v1.10 with strict filtering
Analyzes prediction distribution and performance with 50% confidence and 60% probability thresholds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime


def load_predictions():
    """Load predictions from the latest model"""
    model_path = Path("user_data/models/btc_price_direction_v1.10_strict_filtering")

    # Find the most recent prediction file
    prediction_files = list(model_path.glob("backtesting_predictions/*_prediction.feather"))
    if not prediction_files:
        print("No prediction files found!")
        return None

    # Load the most recent file
    latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
    print(f"Loading predictions from: {latest_file}")

    predictions = pd.read_feather(latest_file)
    return predictions


def load_price_data():
    """Load price data for accuracy calculation"""
    price_file = Path("user_data/data/binance/futures/BTC_USDT_USDT-1d-futures.feather")
    if not price_file.exists():
        print(f"Price file not found: {price_file}")
        return None

    price_data = pd.read_feather(price_file)
    price_data["date"] = pd.to_datetime(price_data["date"])
    return price_data


def analyze_predictions(predictions, price_data):
    """Analyze prediction distribution and quality"""

    print("=" * 80)
    print("BTC PRICE DIRECTION STRATEGY v1.10 - STRICT FILTERING ANALYSIS")
    print("=" * 80)

    # Basic statistics
    print(f"\nTotal predictions: {len(predictions)}")
    print(f"Date range: {predictions['date'].min()} to {predictions['date'].max()}")

    # Check for prediction columns
    prediction_cols = [col for col in predictions.columns if col in ["up", "down", "sideways"]]
    print(f"Prediction columns found: {prediction_cols}")

    if not prediction_cols:
        print("No prediction columns found!")
        return

    # Calculate prediction statistics
    predictions["max_prob"] = predictions[prediction_cols].max(axis=1)
    predictions["second_max_prob"] = predictions[prediction_cols].apply(
        lambda x: x.nlargest(2).iloc[1], axis=1
    )
    predictions["prob_diff"] = predictions["max_prob"] - predictions["second_max_prob"]

    # Prediction distribution
    print(f"\nPREDICTION DISTRIBUTION:")
    for col in prediction_cols:
        max_pred = (predictions[col] == predictions["max_prob"]).sum()
        print(f"  {col}: {max_pred} predictions ({max_pred / len(predictions) * 100:.1f}%)")

    # Confidence analysis
    print(f"\nCONFIDENCE ANALYSIS:")
    print(f"  Average max probability: {predictions['max_prob'].mean():.3f}")
    print(f"  Average probability difference: {predictions['prob_diff'].mean():.3f}")
    print(f"  Median probability difference: {predictions['prob_diff'].median():.3f}")

    # Threshold analysis
    confidence_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    prob_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]

    print(f"\nTHRESHOLD ANALYSIS:")
    print(f"{'Conf Threshold':<15} {'Prob Threshold':<15} {'Passing':<10} {'% Passing':<12}")
    print("-" * 60)

    for conf_thresh in confidence_thresholds:
        for prob_thresh in prob_thresholds:
            passing = (
                (predictions["prob_diff"] > conf_thresh)
                & (predictions["max_prob"] > prob_thresh)
                & (predictions["do_predict"] == 1)
            ).sum()
            pct_passing = passing / len(predictions) * 100
            print(f"{conf_thresh:<15} {prob_thresh:<15} {passing:<10} {pct_passing:<12.1f}%")

    # Current thresholds (50% confidence, 60% probability)
    current_passing = (
        (predictions["prob_diff"] > 0.50)
        & (predictions["max_prob"] > 0.60)
        & (predictions["do_predict"] == 1)
    ).sum()

    print(f"\nCURRENT THRESHOLDS (50% confidence, 60% probability):")
    print(
        f"  Predictions passing filters: {current_passing} ({current_passing / len(predictions) * 100:.1f}%)"
    )

    # Analyze passing predictions
    passing_mask = (
        (predictions["prob_diff"] > 0.50)
        & (predictions["max_prob"] > 0.60)
        & (predictions["do_predict"] == 1)
    )

    passing_predictions = predictions[passing_mask].copy()

    if len(passing_predictions) > 0:
        print(f"\nPASSING PREDICTIONS ANALYSIS:")
        print(f"  Total passing: {len(passing_predictions)}")

        # Direction distribution of passing predictions
        for col in prediction_cols:
            max_pred = (passing_predictions[col] == passing_predictions["max_prob"]).sum()
            print(
                f"  {col}: {max_pred} predictions ({max_pred / len(passing_predictions) * 100:.1f}%)"
            )

        # Confidence distribution
        print(f"\n  Average confidence (prob_diff): {passing_predictions['prob_diff'].mean():.3f}")
        print(f"  Average probability: {passing_predictions['max_prob'].mean():.3f}")

        # Time distribution
        passing_predictions["year"] = passing_predictions["date"].dt.year
        yearly_counts = passing_predictions["year"].value_counts().sort_index()
        print(f"\n  Yearly distribution:")
        for year, count in yearly_counts.items():
            print(f"    {year}: {count} predictions")

    # Calculate real accuracy if price data is available
    if price_data is not None:
        print(f"\nACCURACY ANALYSIS:")

        # Merge predictions with price data
        merged = predictions.merge(price_data[["date", "close"]], on="date", how="left")
        merged = merged.sort_values("date")

        # Calculate 3-day forward returns
        merged["future_close"] = merged["close"].shift(-3)
        merged["actual_return"] = merged["future_close"] / merged["close"] - 1

        # Determine actual direction
        threshold = 0.005  # 0.5% threshold
        merged["actual_direction"] = np.where(
            merged["actual_return"] > threshold,
            "up",
            np.where(merged["actual_return"] < -threshold, "down", "sideways"),
        )

        # Calculate accuracy for all predictions
        merged["predicted_direction"] = merged[prediction_cols].idxmax(axis=1)
        merged["correct"] = merged["predicted_direction"] == merged["actual_direction"]

        overall_accuracy = merged["correct"].mean()
        print(f"  Overall accuracy: {overall_accuracy:.1%}")

        # Accuracy by direction
        for direction in prediction_cols:
            mask = merged["predicted_direction"] == direction
            if mask.sum() > 0:
                acc = merged.loc[mask, "correct"].mean()
                count = mask.sum()
                print(f"  {direction} accuracy: {acc:.1%} ({count} predictions)")

        # Accuracy for passing predictions
        if len(passing_predictions) > 0:
            passing_merged = merged[passing_mask].copy()
            passing_accuracy = passing_merged["correct"].mean()
            print(f"  Passing predictions accuracy: {passing_accuracy:.1%}")

            # Accuracy by direction for passing predictions
            for direction in prediction_cols:
                mask = passing_merged["predicted_direction"] == direction
                if mask.sum() > 0:
                    acc = passing_merged.loc[mask, "correct"].mean()
                    count = mask.sum()
                    print(f"    {direction} (passing): {acc:.1%} ({count} predictions)")

    # Recommendations
    print(f"\nRECOMMENDATIONS:")

    if current_passing < 10:
        print("  ⚠️  Too few predictions passing current thresholds")
        print("  → Consider reducing confidence threshold to 0.3-0.4")
        print("  → Consider reducing probability threshold to 0.5")
    elif current_passing > 100:
        print("  ⚠️  Too many predictions passing current thresholds")
        print("  → Consider increasing confidence threshold to 0.6-0.7")
        print("  → Consider increasing probability threshold to 0.7")
    else:
        print("  ✅ Current thresholds seem reasonable")

    # Suggest optimal thresholds based on data
    optimal_conf = predictions["prob_diff"].quantile(0.8)  # Top 20% confidence
    optimal_prob = predictions["max_prob"].quantile(0.7)  # Top 30% probability

    print(f"  📊 Suggested thresholds based on data distribution:")
    print(f"     Confidence threshold: {optimal_conf:.3f}")
    print(f"     Probability threshold: {optimal_prob:.3f}")

    # Test suggested thresholds
    suggested_passing = (
        (predictions["prob_diff"] > optimal_conf)
        & (predictions["max_prob"] > optimal_prob)
        & (predictions["do_predict"] == 1)
    ).sum()

    print(
        f"     Would result in {suggested_passing} predictions ({suggested_passing / len(predictions) * 100:.1f}%)"
    )


def create_visualizations(predictions):
    """Create visualizations for the analysis"""

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("BTC Price Direction Strategy v1.10 - Strict Filtering Analysis", fontsize=16)

    # 1. Prediction distribution
    prediction_cols = [col for col in predictions.columns if col in ["up", "down", "sideways"]]
    if prediction_cols:
        pred_counts = [
            (predictions[col] == predictions[prediction_cols].max(axis=1)).sum()
            for col in prediction_cols
        ]
        axes[0, 0].pie(pred_counts, labels=prediction_cols, autopct="%1.1f%%", startangle=90)
        axes[0, 0].set_title("Prediction Distribution")

    # 2. Confidence distribution
    predictions["max_prob"] = predictions[prediction_cols].max(axis=1)
    predictions["second_max_prob"] = predictions[prediction_cols].apply(
        lambda x: x.nlargest(2).iloc[1], axis=1
    )
    predictions["prob_diff"] = predictions["max_prob"] - predictions["second_max_prob"]

    axes[0, 1].hist(predictions["prob_diff"], bins=30, alpha=0.7, edgecolor="black")
    axes[0, 1].axvline(x=0.5, color="red", linestyle="--", label="Current threshold (0.5)")
    axes[0, 1].set_xlabel("Probability Difference (Confidence)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Confidence Distribution")
    axes[0, 1].legend()

    # 3. Probability distribution
    axes[1, 0].hist(predictions["max_prob"], bins=30, alpha=0.7, edgecolor="black")
    axes[1, 0].axvline(x=0.6, color="red", linestyle="--", label="Current threshold (0.6)")
    axes[1, 0].set_xlabel("Maximum Probability")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Probability Distribution")
    axes[1, 0].legend()

    # 4. Threshold analysis heatmap
    confidence_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    prob_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]

    heatmap_data = []
    for conf_thresh in confidence_thresholds:
        row = []
        for prob_thresh in prob_thresholds:
            passing = (
                (predictions["prob_diff"] > conf_thresh)
                & (predictions["max_prob"] > prob_thresh)
                & (predictions["do_predict"] == 1)
            ).sum()
            row.append(passing)
        heatmap_data.append(row)

    sns.heatmap(
        heatmap_data,
        xticklabels=prob_thresholds,
        yticklabels=confidence_thresholds,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        ax=axes[1, 1],
    )
    axes[1, 1].set_xlabel("Probability Threshold")
    axes[1, 1].set_ylabel("Confidence Threshold")
    axes[1, 1].set_title("Number of Predictions Passing Thresholds")

    plt.tight_layout()

    # Save the plot
    output_path = Path("user_data/report/v1.10_strict_filtering_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 Visualization saved to: {output_path}")

    plt.show()


def main():
    """Main analysis function"""

    # Load data
    predictions = load_predictions()
    if predictions is None:
        return

    price_data = load_price_data()

    # Run analysis
    analyze_predictions(predictions, price_data)

    # Create visualizations
    create_visualizations(predictions)

    print(f"\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
