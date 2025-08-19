#!/usr/bin/env python3
"""
Calculate real prediction accuracy for BTC Price Direction Strategy v1.9
Compares model predictions with actual future price movements
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime


def load_prediction_files(model_dir):
    """Load all prediction files from the model directory"""
    predictions = []

    pred_dir = Path(model_dir) / "backtesting_predictions"
    if not pred_dir.exists():
        print(f"Prediction directory not found: {pred_dir}")
        return pd.DataFrame()

    for pred_file in pred_dir.glob("*.feather"):
        try:
            df = pd.read_feather(pred_file)
            predictions.append(df)
        except Exception as e:
            print(f"Error loading {pred_file}: {e}")

    if not predictions:
        print("No prediction files found")
        return pd.DataFrame()

    # Combine all predictions
    combined_df = pd.concat(predictions, ignore_index=True)
    print(f"Combined {len(combined_df)} total predictions")

    return combined_df


def load_price_data():
    """Load BTC price data for calculating actual returns"""
    try:
        # Try to load from user_data/data directory
        price_file = "user_data/data/binance/futures/BTC_USDT_USDT-1d-futures.feather"
        if os.path.exists(price_file):
            df = pd.read_feather(price_file)
            print(f"Loaded price data: {len(df)} rows")
            return df
        else:
            print(f"Price file not found: {price_file}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading price data: {e}")
        return pd.DataFrame()


def calculate_real_accuracy(predictions_df, price_df):
    """Calculate real prediction accuracy by comparing with actual price movements"""
    if predictions_df.empty or price_df.empty:
        print("Missing data for accuracy calculation")
        return

    print("\n" + "=" * 60)
    print("REAL PREDICTION ACCURACY ANALYSIS")
    print("=" * 60)

    # Merge predictions with price data
    merged_df = predictions_df.merge(price_df[["date", "close"]], on="date", how="left")

    # Calculate actual 3-day forward returns
    merged_df["future_close"] = merged_df["close"].shift(-3)
    merged_df["actual_return"] = (merged_df["future_close"] / merged_df["close"]) - 1

    # Remove rows with missing future data
    merged_df = merged_df.dropna(subset=["actual_return"])

    print(f"Predictions with actual return data: {len(merged_df)}")

    # Define direction thresholds (same as in strategy)
    threshold = 0.005  # 0.5%

    # Calculate actual directions
    merged_df["actual_direction"] = np.where(
        merged_df["actual_return"] > threshold,
        "up",
        np.where(merged_df["actual_return"] < -threshold, "down", "sideways"),
    )

    # Get predicted directions
    merged_df["predicted_direction"] = merged_df[["up", "down", "sideways"]].idxmax(axis=1)

    # Calculate overall accuracy
    overall_accuracy = (merged_df["predicted_direction"] == merged_df["actual_direction"]).mean()
    print(f"\nOverall accuracy: {overall_accuracy:.3f} ({overall_accuracy * 100:.1f}%)")

    # Calculate accuracy by predicted class
    print(f"\nAccuracy by predicted class:")
    for pred_class in ["up", "down", "sideways"]:
        class_mask = merged_df["predicted_direction"] == pred_class
        if class_mask.sum() > 0:
            class_accuracy = (
                merged_df.loc[class_mask, "predicted_direction"]
                == merged_df.loc[class_mask, "actual_direction"]
            ).mean()
            print(
                f"  {pred_class}: {class_accuracy:.3f} ({class_accuracy * 100:.1f}%) - {class_mask.sum()} predictions"
            )

    # Calculate accuracy by actual class
    print(f"\nAccuracy by actual class:")
    for actual_class in ["up", "down", "sideways"]:
        class_mask = merged_df["actual_direction"] == actual_class
        if class_mask.sum() > 0:
            class_accuracy = (
                merged_df.loc[class_mask, "predicted_direction"]
                == merged_df.loc[class_mask, "actual_direction"]
            ).mean()
            print(
                f"  {actual_class}: {class_accuracy:.3f} ({class_accuracy * 100:.1f}%) - {class_mask.sum()} actual occurrences"
            )

    # Confusion matrix
    print(f"\nConfusion Matrix (Predicted vs Actual):")
    confusion_matrix = pd.crosstab(
        merged_df["predicted_direction"], merged_df["actual_direction"], margins=True
    )
    print(confusion_matrix)

    # Analyze confident predictions
    print(f"\nAccuracy for confident predictions (DI > 0.15):")
    confident_mask = merged_df["DI_values"] > 0.15
    if confident_mask.sum() > 0:
        confident_accuracy = (
            merged_df.loc[confident_mask, "predicted_direction"]
            == merged_df.loc[confident_mask, "actual_direction"]
        ).mean()
        print(
            f"  Confident predictions: {confident_accuracy:.3f} ({confident_accuracy * 100:.1f}%) - {confident_mask.sum()} predictions"
        )

        # Accuracy by confidence level
        confidence_levels = [0.2, 0.3, 0.5, 1.0, 2.0, 5.0]
        for level in confidence_levels:
            high_conf_mask = merged_df["DI_values"] > level
            if high_conf_mask.sum() > 0:
                high_conf_accuracy = (
                    merged_df.loc[high_conf_mask, "predicted_direction"]
                    == merged_df.loc[high_conf_mask, "actual_direction"]
                ).mean()
                print(
                    f"  DI > {level}: {high_conf_accuracy:.3f} ({high_conf_accuracy * 100:.1f}%) - {high_conf_mask.sum()} predictions"
                )

    # Analyze trading signals
    if "do_predict" in merged_df.columns:
        print(f"\nAccuracy for trading signals (do_predict == 1):")
        signal_mask = merged_df["do_predict"] == 1
        if signal_mask.sum() > 0:
            signal_accuracy = (
                merged_df.loc[signal_mask, "predicted_direction"]
                == merged_df.loc[signal_mask, "actual_direction"]
            ).mean()
            print(
                f"  Trading signals: {signal_accuracy:.3f} ({signal_accuracy * 100:.1f}%) - {signal_mask.sum()} signals"
            )

    # Analyze by time period
    print(f"\nAccuracy by time period:")
    merged_df["year"] = merged_df["date"].dt.year
    for year in sorted(merged_df["year"].unique()):
        year_mask = merged_df["year"] == year
        if year_mask.sum() > 0:
            year_accuracy = (
                merged_df.loc[year_mask, "predicted_direction"]
                == merged_df.loc[year_mask, "actual_direction"]
            ).mean()
            print(
                f"  {year}: {year_accuracy:.3f} ({year_accuracy * 100:.1f}%) - {year_mask.sum()} predictions"
            )

    # Analyze prediction strength vs accuracy
    print(f"\nAccuracy by prediction strength:")
    merged_df["max_probability"] = merged_df[["up", "down", "sideways"]].max(axis=1)
    prob_bins = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0]
    for i in range(len(prob_bins) - 1):
        prob_mask = (merged_df["max_probability"] >= prob_bins[i]) & (
            merged_df["max_probability"] < prob_bins[i + 1]
        )
        if prob_mask.sum() > 0:
            prob_accuracy = (
                merged_df.loc[prob_mask, "predicted_direction"]
                == merged_df.loc[prob_mask, "actual_direction"]
            ).mean()
            print(
                f"  Probability {prob_bins[i]:.2f}-{prob_bins[i + 1]:.2f}: {prob_accuracy:.3f} ({prob_accuracy * 100:.1f}%) - {prob_mask.sum()} predictions"
            )

    return merged_df


def create_accuracy_visualizations(df):
    """Create visualizations for accuracy analysis"""
    if df.empty:
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("BTC Price Direction Strategy v1.9 - Real Accuracy Analysis", fontsize=16)

    # 1. Accuracy by predicted class
    pred_classes = ["up", "down", "sideways"]
    accuracies = []
    counts = []
    for pred_class in pred_classes:
        class_mask = df["predicted_direction"] == pred_class
        if class_mask.sum() > 0:
            accuracy = (
                df.loc[class_mask, "predicted_direction"] == df.loc[class_mask, "actual_direction"]
            ).mean()
            accuracies.append(accuracy)
            counts.append(class_mask.sum())
        else:
            accuracies.append(0)
            counts.append(0)

    axes[0, 0].bar(pred_classes, accuracies, color=["green", "red", "orange"])
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("Accuracy by Predicted Class")
    axes[0, 0].set_ylim(0, 1)
    for i, (acc, count) in enumerate(zip(accuracies, counts)):
        axes[0, 0].text(i, acc + 0.02, f"{acc:.3f}\n({count})", ha="center", va="bottom")

    # 2. Accuracy by actual class
    actual_classes = ["up", "down", "sideways"]
    actual_accuracies = []
    actual_counts = []
    for actual_class in actual_classes:
        class_mask = df["actual_direction"] == actual_class
        if class_mask.sum() > 0:
            accuracy = (
                df.loc[class_mask, "predicted_direction"] == df.loc[class_mask, "actual_direction"]
            ).mean()
            actual_accuracies.append(accuracy)
            actual_counts.append(class_mask.sum())
        else:
            actual_accuracies.append(0)
            actual_counts.append(0)

    axes[0, 1].bar(actual_classes, actual_accuracies, color=["green", "red", "orange"])
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Accuracy by Actual Class")
    axes[0, 1].set_ylim(0, 1)
    for i, (acc, count) in enumerate(zip(actual_accuracies, actual_counts)):
        axes[0, 1].text(i, acc + 0.02, f"{acc:.3f}\n({count})", ha="center", va="bottom")

    # 3. Accuracy by confidence level
    confidence_levels = [0.2, 0.5, 1.0, 2.0, 5.0]
    conf_accuracies = []
    conf_counts = []
    for level in confidence_levels:
        conf_mask = df["DI_values"] > level
        if conf_mask.sum() > 0:
            accuracy = (
                df.loc[conf_mask, "predicted_direction"] == df.loc[conf_mask, "actual_direction"]
            ).mean()
            conf_accuracies.append(accuracy)
            conf_counts.append(conf_mask.sum())
        else:
            conf_accuracies.append(0)
            conf_counts.append(0)

    axes[0, 2].plot(confidence_levels, conf_accuracies, marker="o", linewidth=2, markersize=8)
    axes[0, 2].set_xlabel("DI Threshold")
    axes[0, 2].set_ylabel("Accuracy")
    axes[0, 2].set_title("Accuracy by Confidence Level")
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Accuracy by prediction strength
    prob_bins = [0.35, 0.4, 0.45, 0.5, 0.55, 0.6]
    prob_accuracies = []
    prob_counts = []
    for i in range(len(prob_bins) - 1):
        prob_mask = (df["max_probability"] >= prob_bins[i]) & (
            df["max_probability"] < prob_bins[i + 1]
        )
        if prob_mask.sum() > 0:
            accuracy = (
                df.loc[prob_mask, "predicted_direction"] == df.loc[prob_mask, "actual_direction"]
            ).mean()
            prob_accuracies.append(accuracy)
            prob_counts.append(prob_mask.sum())
        else:
            prob_accuracies.append(0)
            prob_counts.append(0)

    prob_labels = [f"{prob_bins[i]:.2f}-{prob_bins[i + 1]:.2f}" for i in range(len(prob_bins) - 1)]
    axes[1, 0].bar(prob_labels, prob_accuracies)
    axes[1, 0].set_xlabel("Max Probability Range")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_title("Accuracy by Prediction Strength")
    axes[1, 0].tick_params(axis="x", rotation=45)

    # 5. Accuracy over time
    df["year"] = df["date"].dt.year
    years = sorted(df["year"].unique())
    year_accuracies = []
    year_counts = []
    for year in years:
        year_mask = df["year"] == year
        if year_mask.sum() > 0:
            accuracy = (
                df.loc[year_mask, "predicted_direction"] == df.loc[year_mask, "actual_direction"]
            ).mean()
            year_accuracies.append(accuracy)
            year_counts.append(year_mask.sum())
        else:
            year_accuracies.append(0)
            year_counts.append(0)

    axes[1, 1].plot(years, year_accuracies, marker="o", linewidth=2, markersize=8)
    axes[1, 1].set_xlabel("Year")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].set_title("Accuracy Over Time")
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Actual vs Predicted distribution
    actual_counts = df["actual_direction"].value_counts()
    predicted_counts = df["predicted_direction"].value_counts()

    x = np.arange(len(actual_counts))
    width = 0.35

    axes[1, 2].bar(x - width / 2, actual_counts.values, width, label="Actual", alpha=0.7)
    axes[1, 2].bar(x + width / 2, predicted_counts.values, width, label="Predicted", alpha=0.7)
    axes[1, 2].set_xlabel("Direction")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].set_title("Actual vs Predicted Distribution")
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(actual_counts.index)
    axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig("user_data/report/v1.9_real_accuracy_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def main():
    """Main function"""
    model_dir = "user_data/models/btc_price_direction_v1.9_confidence_threshold"

    print("Loading prediction files...")
    predictions_df = load_prediction_files(model_dir)

    print("Loading price data...")
    price_df = load_price_data()

    if not predictions_df.empty and not price_df.empty:
        # Calculate real accuracy
        merged_df = calculate_real_accuracy(predictions_df, price_df)

        # Create visualizations
        print("\nCreating accuracy visualizations...")
        create_accuracy_visualizations(merged_df)

        print(
            f"\nAccuracy analysis complete! Visualizations saved to user_data/report/v1.9_real_accuracy_analysis.png"
        )
    else:
        print("Missing data for accuracy analysis")


if __name__ == "__main__":
    main()
