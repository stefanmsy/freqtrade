#!/usr/bin/env python3
"""
Comprehensive analysis script for BTC Price Direction Strategy v1.13 with asymmetric thresholds and regime-based filtering
Analyzes prediction distribution and performance with asymmetric confidence/probability thresholds
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime


def load_all_predictions():
    """Load predictions from all model files for v1.13"""
    model_path = Path("user_data/models/btc_price_direction_v1.13_asymmetric_regime_v2")

    # Find all prediction files
    prediction_files = list(model_path.glob("backtesting_predictions/*_prediction.feather"))
    if not prediction_files:
        print("No prediction files found!")
        return None

    print(f"Found {len(prediction_files)} prediction files")

    # Load and combine all predictions
    all_predictions = []
    for file_path in prediction_files:
        try:
            pred = pd.read_feather(file_path)
            all_predictions.append(pred)
            print(f"Loaded {len(pred)} predictions from {file_path.name}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    if not all_predictions:
        return None

    # Combine all predictions
    combined = pd.concat(all_predictions, ignore_index=True)
    combined = combined.drop_duplicates(subset=["date"]).sort_values("date")

    print(f"Combined {len(combined)} unique predictions")
    return combined


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
    """Analyze prediction distribution and quality for v1.13"""

    print("=" * 80)
    print("BTC PRICE DIRECTION STRATEGY v1.13 - ASYMMETRIC REGIME FILTERING ANALYSIS")
    print("=" * 80)

    # Basic statistics
    print(f"\nTotal predictions: {len(predictions)}")
    print(f"Date range: {predictions['date'].min()} to {predictions['date'].max()}")

    # Check for prediction columns (2-class only)
    prediction_cols = [col for col in predictions.columns if col in ["up", "down"]]
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

    # Filter to only predictions that are actually made
    predictions_with_signals = predictions[predictions["do_predict"] == 1].copy()
    print(f"\nPredictions with signals (do_predict=1): {len(predictions_with_signals)}")

    if len(predictions_with_signals) == 0:
        print("No predictions with signals found!")
        return

    # Prediction distribution for signals
    print(f"\nPREDICTION DISTRIBUTION (with signals):")
    for col in prediction_cols:
        max_pred = (predictions_with_signals[col] == predictions_with_signals["max_prob"]).sum()
        print(
            f"  {col}: {max_pred} predictions ({max_pred / len(predictions_with_signals) * 100:.1f}%)"
        )

    # Confidence analysis
    print(f"\nCONFIDENCE ANALYSIS:")
    print(f"  Average max probability: {predictions_with_signals['max_prob'].mean():.3f}")
    print(f"  Average probability difference: {predictions_with_signals['prob_diff'].mean():.3f}")
    print(f"  Median probability difference: {predictions_with_signals['prob_diff'].median():.3f}")

    # Asymmetric threshold analysis for v1.13
    # Long thresholds: 45% confidence, 55% probability
    # Short thresholds: 60% confidence, 65% probability

    print(f"\nASYMMETRIC THRESHOLD ANALYSIS (v1.13):")

    # Long predictions (up)
    long_predictions = predictions_with_signals[
        predictions_with_signals["up"] > predictions_with_signals["down"]
    ].copy()
    if len(long_predictions) > 0:
        long_passing = (
            (long_predictions["prob_diff"] > 0.45) & (long_predictions["max_prob"] > 0.55)
        ).sum()
        print(f"  Long predictions (up): {len(long_predictions)} total")
        print(
            f"    Passing long thresholds (45% conf, 55% prob): {long_passing} ({long_passing / len(long_predictions) * 100:.1f}%)"
        )

    # Short predictions (down)
    short_predictions = predictions_with_signals[
        predictions_with_signals["down"] > predictions_with_signals["up"]
    ].copy()
    if len(short_predictions) > 0:
        short_passing = (
            (short_predictions["prob_diff"] > 0.60) & (short_predictions["max_prob"] > 0.65)
        ).sum()
        print(f"  Short predictions (down): {len(short_predictions)} total")
        print(
            f"    Passing short thresholds (60% conf, 65% prob): {short_passing} ({short_passing / len(short_predictions) * 100:.1f}%)"
        )

    # Total passing predictions
    total_passing = long_passing + short_passing
    print(
        f"\n  Total predictions passing asymmetric thresholds: {total_passing} ({total_passing / len(predictions_with_signals) * 100:.1f}%)"
    )

    # Compare with symmetric thresholds
    print(f"\nCOMPARISON WITH SYMMETRIC THRESHOLDS:")

    # Test different symmetric thresholds
    symmetric_thresholds = [
        (0.45, 0.55),  # Same as long thresholds
        (0.50, 0.60),  # Middle ground
        (0.60, 0.65),  # Same as short thresholds
    ]

    for conf_thresh, prob_thresh in symmetric_thresholds:
        symmetric_passing = (
            (predictions_with_signals["prob_diff"] > conf_thresh)
            & (predictions_with_signals["max_prob"] > prob_thresh)
        ).sum()
        print(
            f"  Symmetric ({conf_thresh:.0%} conf, {prob_thresh:.0%} prob): {symmetric_passing} predictions ({symmetric_passing / len(predictions_with_signals) * 100:.1f}%)"
        )

    # Calculate real accuracy if price data is available
    if price_data is not None:
        print(f"\nACCURACY ANALYSIS:")

        # Merge predictions with price data
        merged = predictions_with_signals.merge(
            price_data[["date", "close"]], on="date", how="left"
        )
        merged = merged.sort_values("date")

        # Calculate 3-day forward returns
        merged["future_close"] = merged["close"].shift(-3)
        merged["actual_return"] = merged["future_close"] / merged["close"] - 1

        # Determine actual direction (2-class with 1.5% threshold)
        threshold = 0.015  # 1.5% threshold (matching strategy)
        merged["actual_direction"] = np.where(
            merged["actual_return"] > threshold,
            "up",
            "down",
        )

        # Calculate accuracy for all predictions with signals
        merged["predicted_direction"] = merged[prediction_cols].idxmax(axis=1)
        merged["correct"] = merged["predicted_direction"] == merged["actual_direction"]

        overall_accuracy = merged["correct"].mean()
        print(f"  Overall accuracy (with signals): {overall_accuracy:.1%}")

        # Accuracy by direction
        for direction in prediction_cols:
            mask = merged["predicted_direction"] == direction
            if mask.sum() > 0:
                acc = merged.loc[mask, "correct"].mean()
                count = mask.sum()
                print(f"  {direction} accuracy: {acc:.1%} ({count} predictions)")

        # Accuracy for asymmetric passing predictions
        if len(long_predictions) > 0 or len(short_predictions) > 0:
            # Get the dates of passing predictions
            passing_dates = []
            if len(long_predictions) > 0:
                long_passing_dates = long_predictions[
                    (long_predictions["prob_diff"] > 0.45) & (long_predictions["max_prob"] > 0.55)
                ]["date"].tolist()
                passing_dates.extend(long_passing_dates)

            if len(short_predictions) > 0:
                short_passing_dates = short_predictions[
                    (short_predictions["prob_diff"] > 0.60) & (short_predictions["max_prob"] > 0.65)
                ]["date"].tolist()
                passing_dates.extend(short_passing_dates)

            if passing_dates:
                passing_merged = merged[merged["date"].isin(passing_dates)].copy()

                if len(passing_merged) > 0:
                    passing_accuracy = passing_merged["correct"].mean()
                    print(f"  Asymmetric passing predictions accuracy: {passing_accuracy:.1%}")

                    # Accuracy by direction for passing predictions
                    for direction in prediction_cols:
                        mask = passing_merged["predicted_direction"] == direction
                        if mask.sum() > 0:
                            acc = passing_merged.loc[mask, "correct"].mean()
                            count = mask.sum()
                            print(f"    {direction} (passing): {acc:.1%} ({count} predictions)")
                else:
                    print("  No passing predictions found in merged data")

    # Recommendations
    print(f"\nRECOMMENDATIONS:")

    if total_passing < 10:
        print("  ⚠️  Too few predictions passing asymmetric thresholds")
        print("  → Consider reducing long thresholds to (40% conf, 50% prob)")
        print("  → Consider reducing short thresholds to (55% conf, 60% prob)")
    elif total_passing > 200:
        print("  ⚠️  Too many predictions passing asymmetric thresholds")
        print("  → Consider increasing long thresholds to (50% conf, 60% prob)")
        print("  → Consider increasing short thresholds to (65% conf, 70% prob)")
    else:
        print("  ✅ Current asymmetric thresholds seem reasonable")

    # Analyze regime distribution
    if price_data is not None:
        print(f"\nREGIME ANALYSIS:")

        # Calculate SMA for regime detection
        price_data = price_data.sort_values("date")
        price_data["sma_20"] = price_data["close"].rolling(20).mean()
        price_data["sma_50"] = price_data["close"].rolling(50).mean()
        price_data["bull_regime"] = price_data["sma_20"] > price_data["sma_50"]

        # Merge with predictions
        regime_merged = predictions_with_signals.merge(
            price_data[["date", "bull_regime"]], on="date", how="left"
        )

        if len(regime_merged) > 0:
            bull_periods = regime_merged["bull_regime"].sum()
            bear_periods = (~regime_merged["bull_regime"]).sum()

            print(
                f"  Bullish periods: {bull_periods} ({bull_periods / len(regime_merged) * 100:.1f}%)"
            )
            print(
                f"  Bearish periods: {bear_periods} ({bear_periods / len(regime_merged) * 100:.1f}%)"
            )

            # Predictions by regime
            bull_predictions = regime_merged[regime_merged["bull_regime"] == True]
            bear_predictions = regime_merged[regime_merged["bull_regime"] == False]

            if len(bull_predictions) > 0:
                bull_up = (bull_predictions["up"] > bull_predictions["down"]).sum()
                print(f"  Bullish regime predictions: {len(bull_predictions)} total")
                print(
                    f"    Up predictions: {bull_up} ({bull_up / len(bull_predictions) * 100:.1f}%)"
                )
                print(
                    f"    Down predictions: {len(bull_predictions) - bull_up} ({(len(bull_predictions) - bull_up) / len(bull_predictions) * 100:.1f}%)"
                )

            if len(bear_predictions) > 0:
                bear_down = (bear_predictions["down"] > bear_predictions["up"]).sum()
                print(f"  Bearish regime predictions: {len(bear_predictions)} total")
                print(
                    f"    Up predictions: {len(bear_predictions) - bear_down} ({(len(bear_predictions) - bear_down) / len(bear_predictions) * 100:.1f}%)"
                )
                print(
                    f"    Down predictions: {bear_down} ({bear_down / len(bear_predictions) * 100:.1f}%)"
                )


def create_visualizations(predictions):
    """Create visualizations for the v1.13 analysis"""

    # Filter to predictions with signals
    predictions_with_signals = predictions[predictions["do_predict"] == 1].copy()

    if len(predictions_with_signals) == 0:
        print("No predictions with signals to visualize")
        return

    # Set up the plotting style
    plt.style.use("default")
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        "BTC Price Direction Strategy v1.13 - Asymmetric Regime Filtering Analysis", fontsize=16
    )

    # 1. Prediction distribution
    prediction_cols = [col for col in predictions_with_signals.columns if col in ["up", "down"]]
    if prediction_cols:
        pred_counts = [
            (
                predictions_with_signals[col]
                == predictions_with_signals[prediction_cols].max(axis=1)
            ).sum()
            for col in prediction_cols
        ]
        axes[0, 0].pie(pred_counts, labels=prediction_cols, autopct="%1.1f%%", startangle=90)
        axes[0, 0].set_title("Prediction Distribution (2-class)")

    # 2. Confidence distribution with asymmetric thresholds
    predictions_with_signals["max_prob"] = predictions_with_signals[prediction_cols].max(axis=1)
    predictions_with_signals["second_max_prob"] = predictions_with_signals[prediction_cols].apply(
        lambda x: x.nlargest(2).iloc[1], axis=1
    )
    predictions_with_signals["prob_diff"] = (
        predictions_with_signals["max_prob"] - predictions_with_signals["second_max_prob"]
    )

    axes[0, 1].hist(predictions_with_signals["prob_diff"], bins=30, alpha=0.7, edgecolor="black")
    axes[0, 1].axvline(x=0.45, color="green", linestyle="--", label="Long threshold (0.45)")
    axes[0, 1].axvline(x=0.60, color="red", linestyle="--", label="Short threshold (0.60)")
    axes[0, 1].set_xlabel("Probability Difference (Confidence)")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Confidence Distribution (Asymmetric)")
    axes[0, 1].legend()

    # 3. Probability distribution with asymmetric thresholds
    axes[1, 0].hist(predictions_with_signals["max_prob"], bins=30, alpha=0.7, edgecolor="black")
    axes[1, 0].axvline(x=0.55, color="green", linestyle="--", label="Long threshold (0.55)")
    axes[1, 0].axvline(x=0.65, color="red", linestyle="--", label="Short threshold (0.65)")
    axes[1, 0].set_xlabel("Maximum Probability")
    axes[1, 0].set_ylabel("Frequency")
    axes[1, 0].set_title("Probability Distribution (Asymmetric)")
    axes[1, 0].legend()

    # 4. Time series of predictions
    predictions_with_signals["year"] = predictions_with_signals["date"].dt.year
    yearly_counts = predictions_with_signals["year"].value_counts().sort_index()

    axes[1, 1].bar(yearly_counts.index, yearly_counts.values, alpha=0.7)
    axes[1, 1].set_xlabel("Year")
    axes[1, 1].set_ylabel("Number of Predictions")
    axes[1, 1].set_title("Predictions by Year (2-class)")

    plt.tight_layout()

    # Save the plot
    output_path = Path("user_data/report/v1.13_asymmetric_regime_analysis.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n📊 Visualization saved to: {output_path}")

    plt.show()


def main():
    """Main analysis function"""

    # Load data
    predictions = load_all_predictions()
    if predictions is None:
        return

    price_data = load_price_data()

    # Run analysis
    analyze_predictions(predictions, price_data)

    # Create visualizations
    create_visualizations(predictions)

    print(f"\n✅ v1.13 Asymmetric Regime Filtering Analysis complete!")


if __name__ == "__main__":
    main()
