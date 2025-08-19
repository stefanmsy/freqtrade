#!/usr/bin/env python3
"""
Analysis script for BTC Price Direction Strategy v1.9 Confidence Threshold
Analyzes prediction accuracy and consistency for the confidence threshold approach
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
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
            print(f"Loaded {pred_file.name} with {len(df)} rows")
        except Exception as e:
            print(f"Error loading {pred_file}: {e}")

    if not predictions:
        print("No prediction files found")
        return pd.DataFrame()

    # Combine all predictions
    combined_df = pd.concat(predictions, ignore_index=True)
    print(f"Combined {len(combined_df)} total predictions")

    return combined_df


def analyze_predictions(df):
    """Analyze prediction accuracy and consistency"""
    if df.empty:
        print("No data to analyze")
        return

    print("\n" + "=" * 60)
    print("PREDICTION ANALYSIS FOR V1.9 CONFIDENCE THRESHOLD")
    print("=" * 60)

    # Basic statistics
    print(f"\nTotal predictions: {len(df)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Check for required columns
    required_cols = ["do_predict", "up", "down", "sideways", "DI_values"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Missing columns: {missing_cols}")
        return

    # Analyze prediction confidence
    print(f"\nPrediction confidence analysis:")
    print(f"Average DI value: {df['DI_values'].mean():.3f}")
    print(f"DI value range: {df['DI_values'].min():.3f} to {df['DI_values'].max():.3f}")

    # Analyze class probabilities
    print(f"\nClass probability analysis:")
    print(f"Average 'up' probability: {df['up'].mean():.3f}")
    print(f"Average 'down' probability: {df['down'].mean():.3f}")
    print(f"Average 'sideways' probability: {df['sideways'].mean():.3f}")

    # Find confident predictions (DI > 0.15)
    confident_df = df[df["DI_values"] > 0.15].copy()
    print(
        f"\nConfident predictions (DI > 0.15): {len(confident_df)} out of {len(df)} ({len(confident_df) / len(df) * 100:.1f}%)"
    )

    if len(confident_df) > 0:
        print(f"Confident predictions average DI: {confident_df['DI_values'].mean():.3f}")
        print(f"Confident 'up' probability: {confident_df['up'].mean():.3f}")
        print(f"Confident 'down' probability: {confident_df['down'].mean():.3f}")
        print(f"Confident 'sideways' probability: {confident_df['sideways'].mean():.3f}")

        # Analyze prediction distribution for confident predictions
        confident_df["predicted_class"] = confident_df[["up", "down", "sideways"]].idxmax(axis=1)
        class_counts = confident_df["predicted_class"].value_counts()
        print(f"\nPredicted class distribution (confident predictions):")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} ({count / len(confident_df) * 100:.1f}%)")

    # Analyze all predictions
    df["predicted_class"] = df[["up", "down", "sideways"]].idxmax(axis=1)
    all_class_counts = df["predicted_class"].value_counts()
    print(f"\nPredicted class distribution (all predictions):")
    for class_name, count in all_class_counts.items():
        print(f"  {class_name}: {count} ({count / len(df) * 100:.1f}%)")

    # Analyze prediction strength
    print(f"\nPrediction strength analysis:")
    df["max_probability"] = df[["up", "down", "sideways"]].max(axis=1)
    print(f"Average max probability: {df['max_probability'].mean():.3f}")
    print(
        f"Max probability range: {df['max_probability'].min():.3f} to {df['max_probability'].max():.3f}"
    )

    # Analyze probability differences
    df["prob_diff"] = df[["up", "down", "sideways"]].max(axis=1) - df[
        ["up", "down", "sideways"]
    ].apply(lambda x: x.nlargest(2).iloc[1], axis=1)
    print(f"Average probability difference: {df['prob_diff'].mean():.3f}")
    print(
        f"Probability difference range: {df['prob_diff'].min():.3f} to {df['prob_diff'].max():.3f}"
    )

    # Analyze trading signals
    print(f"\nTrading signal analysis:")
    if "do_predict" in df.columns:
        trading_signals = df[df["do_predict"] == 1]
        print(
            f"Trading signals generated: {len(trading_signals)} out of {len(df)} ({len(trading_signals) / len(df) * 100:.1f}%)"
        )

        if len(trading_signals) > 0:
            signal_class_counts = trading_signals["predicted_class"].value_counts()
            print(f"Trading signal class distribution:")
            for class_name, count in signal_class_counts.items():
                print(f"  {class_name}: {count} ({count / len(trading_signals) * 100:.1f}%)")

    # Analyze confidence threshold effectiveness
    print(f"\nConfidence threshold analysis:")
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    for threshold in thresholds:
        high_conf = df[df["DI_values"] > threshold]
        if len(high_conf) > 0:
            print(
                f"DI > {threshold}: {len(high_conf)} predictions ({len(high_conf) / len(df) * 100:.1f}%)"
            )
            high_conf["predicted_class"] = high_conf[["up", "down", "sideways"]].idxmax(axis=1)
            class_dist = high_conf["predicted_class"].value_counts()
            for class_name, count in class_dist.items():
                print(f"    {class_name}: {count} ({count / len(high_conf) * 100:.1f}%)")

    return df


def create_visualizations(df):
    """Create visualizations for the analysis"""
    if df.empty:
        return

    # Set up the plotting style
    plt.style.use("default")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("BTC Price Direction Strategy v1.9 - Confidence Threshold Analysis", fontsize=16)

    # 1. DI Values Distribution
    axes[0, 0].hist(df["DI_values"], bins=30, alpha=0.7, color="skyblue", edgecolor="black")
    axes[0, 0].axvline(x=0.15, color="red", linestyle="--", label="Confidence Threshold (0.15)")
    axes[0, 0].set_xlabel("DI Values")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Distribution of DI Values")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Class Probabilities Distribution
    axes[0, 1].hist(df["up"], bins=30, alpha=0.7, label="Up", color="green")
    axes[0, 1].hist(df["down"], bins=30, alpha=0.7, label="Down", color="red")
    axes[0, 1].hist(df["sideways"], bins=30, alpha=0.7, label="Sideways", color="orange")
    axes[0, 1].set_xlabel("Probability")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Distribution of Class Probabilities")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Predicted Class Distribution
    df["predicted_class"] = df[["up", "down", "sideways"]].idxmax(axis=1)
    class_counts = df["predicted_class"].value_counts()
    axes[0, 2].pie(class_counts.values, labels=class_counts.index, autopct="%1.1f%%", startangle=90)
    axes[0, 2].set_title("Predicted Class Distribution")

    # 4. DI Values vs Max Probability
    df["max_probability"] = df[["up", "down", "sideways"]].max(axis=1)
    axes[1, 0].scatter(df["DI_values"], df["max_probability"], alpha=0.6)
    axes[1, 0].set_xlabel("DI Values")
    axes[1, 0].set_ylabel("Max Probability")
    axes[1, 0].set_title("DI Values vs Max Probability")
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Confidence vs Prediction Count
    thresholds = np.arange(0.05, 0.35, 0.05)
    counts = []
    for threshold in thresholds:
        count = len(df[df["DI_values"] > threshold])
        counts.append(count)

    axes[1, 1].plot(thresholds, counts, marker="o", linewidth=2, markersize=8)
    axes[1, 1].set_xlabel("DI Threshold")
    axes[1, 1].set_ylabel("Number of Predictions")
    axes[1, 1].set_title("Predictions Above Confidence Threshold")
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Time series of DI values
    df_sorted = df.sort_values("date")
    axes[1, 2].plot(df_sorted["date"], df_sorted["DI_values"], alpha=0.7)
    axes[1, 2].axhline(y=0.15, color="red", linestyle="--", label="Confidence Threshold")
    axes[1, 2].set_xlabel("Date")
    axes[1, 2].set_ylabel("DI Values")
    axes[1, 2].set_title("DI Values Over Time")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(
        "user_data/report/v1.9_confidence_threshold_analysis.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


def main():
    """Main analysis function"""
    model_dir = "user_data/models/btc_price_direction_v1.9_confidence_threshold"

    print("Loading prediction files...")
    df = load_prediction_files(model_dir)

    if not df.empty:
        # Analyze predictions
        df = analyze_predictions(df)

        # Create visualizations
        print("\nCreating visualizations...")
        create_visualizations(df)

        print(
            f"\nAnalysis complete! Visualizations saved to user_data/report/v1.9_confidence_threshold_analysis.png"
        )
    else:
        print("No prediction data found for analysis")


if __name__ == "__main__":
    main()
