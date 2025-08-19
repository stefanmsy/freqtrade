import pandas as pd
from pathlib import Path
import re
import sys

MODEL_ID = "glassnode-swing-lgbm-daily-v1.4-reduced-features"
BASE_DIR = Path("user_data/models") / MODEL_ID
OUT_CSV = Path("user_data/report/feature_importances_aggregated.csv")


def extract_from_html(html_path: Path) -> pd.DataFrame | None:
    """Try to extract a feature-importance table from an HTML file using pandas.read_html.
    Returns a DataFrame with columns ['feature', 'importance'] or None if not found.
    """
    try:
        tables = pd.read_html(html_path, flavor="lxml")
    except Exception:
        try:
            tables = pd.read_html(html_path)
        except Exception:
            return None

    # Heuristics: find a table that contains feature names and importance
    for tbl in tables:
        cols = [str(c).strip().lower() for c in tbl.columns]
        if any("feature" in c for c in cols) and any(
            "importance" in c or "gain" in c for c in cols
        ):
            df = tbl.copy()
            # Standardize column names
            rename_map = {}
            for c in df.columns:
                lc = str(c).strip().lower()
                if "feature" in lc:
                    rename_map[c] = "feature"
                if "importance" in lc or "gain" in lc:
                    rename_map[c] = "importance"
            df = df.rename(columns=rename_map)
            if "feature" in df.columns and "importance" in df.columns:
                # Coerce importance to numeric
                df["importance"] = pd.to_numeric(df["importance"], errors="coerce")
                df = df.dropna(subset=["feature", "importance"]).reset_index(drop=True)
                return df[["feature", "importance"]]
    return None


def collect_feature_importances() -> pd.DataFrame:
    if not BASE_DIR.exists():
        print(f"❌ Models directory not found: {BASE_DIR}")
        sys.exit(1)

    html_files = list(BASE_DIR.rglob("*--target_3d.html"))
    if not html_files:
        print("❌ No feature-importance HTML files found.")
        sys.exit(1)

    all_rows: list[pd.DataFrame] = []
    processed = 0
    for html in sorted(html_files):
        df = extract_from_html(html)
        if df is None or df.empty:
            continue
        # Tag with window (timestamp from filename if present)
        window = html.stem
        df = df.copy()
        df["window"] = window
        all_rows.append(df)
        processed += 1

    if not all_rows:
        print("❌ Could not parse any feature-importance tables from HTML files.")
        sys.exit(1)

    combined = pd.concat(all_rows, ignore_index=True)

    # Aggregate: average importance across windows; also report coverage (how many windows each feature appears in)
    agg = (
        combined.groupby("feature")
        .agg(
            avg_importance=("importance", "mean"),
            med_importance=("importance", "median"),
            appearances=("importance", "count"),
        )
        .sort_values(["avg_importance", "med_importance"], ascending=False)
        .reset_index()
    )

    # Save CSV
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(OUT_CSV, index=False)

    # Print top 30
    topn = agg.head(30)
    print("\n=== Top 30 Features by Average Importance ===")
    for i, row in topn.iterrows():
        print(
            f"{i + 1:2d}. {row['feature']}: avg={row['avg_importance']:.6f} med={row['med_importance']:.6f} n={row['appearances']}"
        )

    print(f"\n💾 Saved aggregated importances to: {OUT_CSV}")
    return agg


if __name__ == "__main__":
    collect_feature_importances()
