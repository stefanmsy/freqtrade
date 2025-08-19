#!/usr/bin/env python3
"""
BTC Prediction Accuracy Analysis - Corrected Version
Properly evaluates prediction accuracy by comparing model predictions with actual future price movements
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import warnings

warnings.filterwarnings("ignore")


class BTCPredictionAccuracyAnalyzer:
    """Analyzer for BTC prediction accuracy without data leakage"""

    def __init__(self, model_identifier: str = "btc_price_direction_v1.1"):
        self.model_identifier = model_identifier
        self.model_path = Path(f"user_data/models/{model_identifier}")
        self.report_path = Path("user_data/report")
        self.report_path.mkdir(exist_ok=True)

        # Find all prediction files
        self.prediction_files = list(self.model_path.glob("backtesting_predictions/*.feather"))
        self.prediction_files.sort()

        print(f"Found {len(self.prediction_files)} prediction files for analysis")

    def load_all_predictions(self) -> pd.DataFrame:
        """Load and combine all prediction files"""
        dfs = []

        for file_path in self.prediction_files:
            try:
                df = pd.read_feather(file_path)
                df["prediction_file"] = file_path.stem
                dfs.append(df)
                print(f"Loaded {len(df)} predictions from {file_path.name}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        if not dfs:
            raise ValueError("No prediction files could be loaded")

        combined_df = pd.concat(dfs, ignore_index=True)
        combined_df["date"] = pd.to_datetime(combined_df["date"])
        combined_df = combined_df.sort_values("date").reset_index(drop=True)

        print(f"Combined dataset: {len(combined_df)} total predictions")
        return combined_df

    def calculate_actual_future_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate actual future returns to compare with predictions"""
        # Calculate 3-day forward returns (matching our target period)
        df["future_return_3d"] = df["close"].shift(-3) / df["close"] - 1

        # Create actual direction labels based on future returns
        # Using same threshold as in strategy (0.5%)
        df["actual_direction"] = np.where(df["future_return_3d"] > 0.005, "up", "down")

        # Add confidence threshold for analysis
        df["prediction_confidence"] = np.maximum(df["up"], df["down"])
        df["predicted_direction"] = np.where(df["up"] > df["down"], "up", "down")

        return df

    def calculate_real_prediction_metrics(self, df: pd.DataFrame) -> Dict:
        """Calculate real prediction accuracy metrics"""

        # Filter for predictions where model was confident (do_predict == 1)
        pred_df = df[df["do_predict"] == 1].copy()

        if len(pred_df) == 0:
            return {"error": "No predictions with do_predict == 1"}

        # Remove rows where we don't have future data (last 3 days)
        pred_df = pred_df.dropna(subset=["future_return_3d", "actual_direction"])

        if len(pred_df) == 0:
            return {"error": "No predictions with future return data"}

        # Get actual and predicted labels
        y_true = pred_df["actual_direction"].values
        y_pred = pred_df["predicted_direction"].values
        y_prob_up = pred_df["up"].values

        # Convert to binary for ROC AUC (1 for 'up', 0 for 'down')
        y_true_binary = (y_true == "up").astype(int)

        metrics = {
            "total_predictions": len(pred_df),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_up": precision_score(y_true, y_pred, pos_label="up", zero_division=0),
            "recall_up": recall_score(y_true, y_pred, pos_label="up", zero_division=0),
            "f1_up": f1_score(y_true, y_pred, pos_label="up", zero_division=0),
            "precision_down": precision_score(y_true, y_pred, pos_label="down", zero_division=0),
            "recall_down": recall_score(y_true, y_pred, pos_label="down", zero_division=0),
            "f1_down": f1_score(y_true, y_pred, pos_label="down", zero_division=0),
            "roc_auc": roc_auc_score(y_true_binary, y_prob_up)
            if len(np.unique(y_true_binary)) > 1
            else 0.5,
        }

        # Add class distribution
        class_counts = pd.Series(y_true).value_counts()
        metrics["actual_up_count"] = class_counts.get("up", 0)
        metrics["actual_down_count"] = class_counts.get("down", 0)

        pred_counts = pd.Series(y_pred).value_counts()
        metrics["predicted_up_count"] = pred_counts.get("up", 0)
        metrics["predicted_down_count"] = pred_counts.get("down", 0)

        # Add future return statistics
        metrics["mean_future_return"] = pred_df["future_return_3d"].mean()
        metrics["std_future_return"] = pred_df["future_return_3d"].std()
        metrics["min_future_return"] = pred_df["future_return_3d"].min()
        metrics["max_future_return"] = pred_df["future_return_3d"].max()

        return metrics, pred_df

    def analyze_prediction_consistency(self, df: pd.DataFrame) -> Dict:
        """Analyze prediction consistency over time and confidence levels"""
        pred_df = df[df["do_predict"] == 1].copy()
        pred_df = pred_df.dropna(subset=["future_return_3d", "actual_direction"])

        if len(pred_df) == 0:
            return {"error": "No predictions available"}

        # Calculate prediction correctness
        pred_df["is_correct"] = pred_df["actual_direction"] == pred_df["predicted_direction"]

        # Monthly consistency
        pred_df["month"] = pred_df["date"].dt.to_period("M")
        monthly_accuracy = pred_df.groupby("month")["is_correct"].mean()
        monthly_counts = pred_df.groupby("month").size()

        # Confidence-based analysis
        pred_df["confidence_bin"] = pd.cut(
            pred_df["prediction_confidence"],
            bins=[0.5, 0.55, 0.6, 0.65, 0.7, 1.0],
            labels=["0.50-0.55", "0.55-0.60", "0.60-0.65", "0.65-0.70", "0.70+"],
        )

        confidence_accuracy = pred_df.groupby("confidence_bin", observed=True)["is_correct"].mean()
        confidence_counts = pred_df.groupby("confidence_bin", observed=True).size()

        # Model consistency across different training periods
        model_consistency = pred_df.groupby("prediction_file")["is_correct"].agg(["mean", "count"])

        consistency_stats = {
            "monthly_accuracy": monthly_accuracy.to_dict(),
            "monthly_counts": monthly_counts.to_dict(),
            "confidence_accuracy": confidence_accuracy.to_dict(),
            "confidence_counts": confidence_counts.to_dict(),
            "model_consistency": model_consistency.to_dict("index"),
            "overall_consistency": pred_df["is_correct"].mean(),
            "consistency_std": pred_df["is_correct"].std(),
        }

        return consistency_stats, pred_df

    def plot_real_prediction_analysis(
        self, df: pd.DataFrame, save_path: Optional[Path] = None
    ) -> None:
        """Create comprehensive real prediction analysis plots"""

        pred_df = df[df["do_predict"] == 1].copy()
        pred_df = pred_df.dropna(subset=["future_return_3d", "actual_direction"])

        if len(pred_df) == 0:
            print("No predictions to plot")
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(
            f"Real BTC Prediction Analysis - {self.model_identifier}",
            fontsize=16,
            fontweight="bold",
        )

        # 1. Real Confusion Matrix
        y_true = pred_df["actual_direction"].values
        y_pred = pred_df["predicted_direction"].values

        cm = confusion_matrix(y_true, y_pred, labels=["down", "up"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted Down", "Predicted Up"],
            yticklabels=["Actual Down", "Actual Up"],
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Real Prediction Confusion Matrix")

        # 2. Future Returns Distribution by Prediction
        up_predictions = pred_df[pred_df["predicted_direction"] == "up"]["future_return_3d"]
        down_predictions = pred_df[pred_df["predicted_direction"] == "down"]["future_return_3d"]

        axes[0, 1].hist(
            up_predictions, bins=30, alpha=0.7, color="green", label="Predicted Up", density=True
        )
        axes[0, 1].hist(
            down_predictions, bins=30, alpha=0.7, color="red", label="Predicted Down", density=True
        )
        axes[0, 1].set_xlabel("3-Day Future Return")
        axes[0, 1].set_ylabel("Density")
        axes[0, 1].set_title("Future Returns by Prediction")
        axes[0, 1].legend()
        axes[0, 1].axvline(x=0, color="black", linestyle="--", alpha=0.7)

        # 3. Accuracy by Confidence Bins
        pred_df["confidence_bin"] = pd.cut(
            pred_df["prediction_confidence"], bins=[0.5, 0.55, 0.6, 0.65, 0.7, 1.0]
        )

        bin_accuracy = pred_df.groupby("confidence_bin", observed=True)["is_correct"].mean()
        bin_counts = pred_df.groupby("confidence_bin", observed=True).size()

        bin_accuracy.plot(kind="bar", ax=axes[0, 2], color="lightgreen")
        axes[0, 2].set_title("Real Accuracy by Confidence Bins")
        axes[0, 2].set_ylabel("Accuracy")
        axes[0, 2].tick_params(axis="x", rotation=45)
        axes[0, 2].axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Random")
        axes[0, 2].legend()

        # Add count labels on bars
        for i, (acc, count) in enumerate(zip(bin_accuracy.values, bin_counts.values)):
            axes[0, 2].text(i, acc + 0.01, f"n={count}", ha="center", va="bottom", fontsize=9)

        # 4. Monthly Accuracy Trend
        pred_df["month"] = pred_df["date"].dt.to_period("M")
        monthly_accuracy = pred_df.groupby("month")["is_correct"].mean()

        axes[1, 0].plot(
            monthly_accuracy.index.astype(str),
            monthly_accuracy.values,
            marker="o",
            linewidth=2,
            markersize=4,
        )
        axes[1, 0].set_title("Monthly Prediction Accuracy Trend")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Random (50%)")
        axes[1, 0].legend()

        # 5. Prediction vs Actual Class Balance
        monthly_actual_up = pred_df.groupby("month")["actual_direction"].apply(
            lambda x: (x == "up").mean()
        )
        monthly_pred_up = pred_df.groupby("month")["predicted_direction"].apply(
            lambda x: (x == "up").mean()
        )

        axes[1, 1].plot(
            monthly_actual_up.index.astype(str),
            monthly_actual_up.values,
            marker="o",
            linewidth=2,
            markersize=4,
            color="green",
            label="Actual Up %",
        )
        axes[1, 1].plot(
            monthly_pred_up.index.astype(str),
            monthly_pred_up.values,
            marker="s",
            linewidth=2,
            markersize=4,
            color="orange",
            label="Predicted Up %",
        )

        axes[1, 1].set_title("Class Balance Over Time")
        axes[1, 1].set_ylabel("Percentage")
        axes[1, 1].tick_params(axis="x", rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0.5, color="red", linestyle="--", alpha=0.7)

        # 6. Model Performance by Training Period
        model_performance = pred_df.groupby("prediction_file")["is_correct"].agg(["mean", "count"])

        axes[1, 2].bar(
            range(len(model_performance)),
            model_performance["mean"],
            color="skyblue",
            edgecolor="navy",
        )
        axes[1, 2].set_title("Model Performance by Training Period")
        axes[1, 2].set_ylabel("Accuracy")
        axes[1, 2].set_xlabel("Training Period")
        axes[1, 2].axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Random")
        axes[1, 2].legend()

        # Add count labels
        for i, (acc, count) in enumerate(
            zip(model_performance["mean"], model_performance["count"])
        ):
            axes[1, 2].text(i, acc + 0.01, f"n={count}", ha="center", va="bottom", fontsize=9)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to {save_path}")

        plt.show()

    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive real prediction analysis report"""
        print("Loading prediction data...")
        df = self.load_all_predictions()

        print("Calculating actual future returns...")
        df = self.calculate_actual_future_returns(df)

        print("Calculating real prediction metrics...")
        metrics, pred_df = self.calculate_real_prediction_metrics(df)

        print("Analyzing prediction consistency...")
        consistency_stats, _ = self.analyze_prediction_consistency(df)

        print("Generating plots...")
        plot_path = self.report_path / f"{self.model_identifier}_real_prediction_analysis.png"
        self.plot_real_prediction_analysis(df, plot_path)

        # Create summary report
        report = {
            "model_identifier": self.model_identifier,
            "analysis_date": pd.Timestamp.now().isoformat(),
            "data_summary": {
                "total_rows": len(df),
                "date_range": f"{df['date'].min()} to {df['date'].max()}",
                "prediction_files": len(self.prediction_files),
            },
            "real_prediction_metrics": metrics,
            "consistency_analysis": consistency_stats,
        }

        # Save report to JSON
        report_file = self.report_path / f"{self.model_identifier}_real_analysis_report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved to {report_file}")

        return report

    def print_summary(self, report: Dict) -> None:
        """Print real analysis summary to console"""
        print("\n" + "=" * 80)
        print(f"REAL BTC PREDICTION ANALYSIS SUMMARY - {report['model_identifier']}")
        print("=" * 80)

        data_summary = report["data_summary"]
        print(f"📊 Data Summary:")
        print(f"   • Total predictions: {data_summary['total_rows']:,}")
        print(f"   • Date range: {data_summary['date_range']}")
        print(f"   • Prediction files: {data_summary['prediction_files']}")

        metrics = report["real_prediction_metrics"]
        if "error" not in metrics:
            print(f"\n🎯 REAL Prediction Performance:")
            print(f"   • Total predictions: {metrics['total_predictions']:,}")
            print(
                f"   • Overall accuracy: {metrics['accuracy']:.3f} ({metrics['accuracy'] * 100:.1f}%)"
            )
            print(f"   • ROC AUC: {metrics['roc_auc']:.3f}")

            print(f"\n   📈 'Up' Class Performance:")
            print(f"   • Precision: {metrics['precision_up']:.3f}")
            print(f"   • Recall: {metrics['recall_up']:.3f}")
            print(f"   • F1-Score: {metrics['f1_up']:.3f}")

            print(f"\n   📉 'Down' Class Performance:")
            print(f"   • Precision: {metrics['precision_down']:.3f}")
            print(f"   • Recall: {metrics['recall_down']:.3f}")
            print(f"   • F1-Score: {metrics['f1_down']:.3f}")

            print(f"\n   📊 Class Distribution:")
            print(
                f"   • Actual Up/Down: {metrics['actual_up_count']}/{metrics['actual_down_count']}"
            )
            print(
                f"   • Predicted Up/Down: {metrics['predicted_up_count']}/{metrics['predicted_down_count']}"
            )

            print(f"\n   💰 Future Return Statistics:")
            print(
                f"   • Mean future return: {metrics['mean_future_return']:.4f} ({metrics['mean_future_return'] * 100:.2f}%)"
            )
            print(
                f"   • Return range: {metrics['min_future_return']:.4f} to {metrics['max_future_return']:.4f}"
            )
            print(f"   • Return std: {metrics['std_future_return']:.4f}")

        consistency_stats = report["consistency_analysis"]
        if "error" not in consistency_stats:
            print(f"\n🔍 Consistency Analysis:")
            print(f"   • Overall consistency: {consistency_stats['overall_consistency']:.3f}")
            print(f"   • Consistency std: {consistency_stats['consistency_std']:.3f}")

            if "confidence_accuracy" in consistency_stats:
                print(f"\n   📊 Accuracy by Confidence:")
                for conf_bin, acc in consistency_stats["confidence_accuracy"].items():
                    count = consistency_stats["confidence_counts"].get(conf_bin, 0)
                    print(f"   • {conf_bin}: {acc:.3f} (n={count})")

        print(
            "\n✅ Real analysis complete! Check the user_data/report directory for detailed results."
        )
        print("=" * 80)


def main():
    """Main function to run the analysis"""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Real BTC Prediction Accuracy")
    parser.add_argument(
        "--model-id",
        type=str,
        default="btc_price_direction_v1.6",
        help="Model identifier (folder name under user_data/models)",
    )

    args = parser.parse_args()

    try:
        analyzer = BTCPredictionAccuracyAnalyzer(args.model_id)
        report = analyzer.generate_comprehensive_report()
        analyzer.print_summary(report)

    except Exception as e:
        print(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    main()
