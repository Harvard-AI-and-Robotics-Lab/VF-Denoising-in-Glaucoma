import pandas as pd
import glob
from pathlib import Path
import numpy as np

# --- Configuration ---
LOG_DIR = Path(__file__).resolve().parent / "logs"
OUTPUT_FILE = "final_sf_results_summary.csv"

# Priority list: (column_name, "max" or "min")
BEST_BY_PRIORITY = [
    ("Val_R2", "max"),
    ("Val_Pearson", "max"),
    ("Val_MAE", "min"),
    ("Train_Loss", "min"),
]

# Metrics you want to record if present (missing -> NaN)
METRICS_TO_SAVE = ["Val_R2", "Val_Pearson", "Val_MAE", "Train_Loss"]

def pick_best_row(df: pd.DataFrame):
    """
    Return (best_row, best_by_col, direction) using the first available metric in BEST_BY_PRIORITY.
    If no usable metric exists, return (None, None, None).
    """
    for col, direction in BEST_BY_PRIORITY:
        if col not in df.columns:
            continue
        df_valid = df.dropna(subset=[col])
        if df_valid.empty:
            continue

        if direction == "max":
            idx = df_valid[col].idxmax()
        else:
            idx = df_valid[col].idxmin()

        return df.loc[idx], col, direction

    return None, None, None

def summarize_all_logs():
    log_files = glob.glob(str(LOG_DIR / "*.txt"))
    if not log_files:
        print(f"No .txt files found in {LOG_DIR}")
        return

    summary_data = []

    for log_path in log_files:
        model_name = Path(log_path).stem.replace("sf_", "")

        try:
            df = pd.read_csv(log_path)
            if df.empty:
                print(f"Skipping {model_name}: empty file")
                continue

            best_row, best_by, direction = pick_best_row(df)
            if best_row is None:
                print(f"Skipping {model_name}: no usable metric columns (or all NaN)")
                continue

            out = {
                "Model": model_name,
                "Best_Epoch": int(best_row["Epoch"]) if "Epoch" in df.columns and pd.notna(best_row.get("Epoch")) else -1,
                "Best_By": best_by,
                "Best_By_Direction": direction,
            }

            # Save whatever metrics exist; missing -> NaN
            for m in METRICS_TO_SAVE:
                out[m] = best_row.get(m, np.nan)

            summary_data.append(out)

        except Exception as e:
            print(f"Error processing {log_path}: {e}")
            continue

    if not summary_data:
        print("No valid logs found.")
        return

    summary_df = pd.DataFrame(summary_data)

    # Sort: primarily by Best_By metric (respecting direction), then by Model
    # Since Best_By differs across rows, a simple, consistent sort is:
    # - prefer rows where Best_By is earlier in priority list
    priority_rank = {col: i for i, (col, _) in enumerate(BEST_BY_PRIORITY)}
    summary_df["Best_By_Priority"] = summary_df["Best_By"].map(priority_rank).fillna(999).astype(int)

    # For numeric sort within each Best_By, make a unified "SortScore"
    def row_sort_score(r):
        col = r["Best_By"]
        val = r.get(col, np.nan)
        if pd.isna(val):
            return np.nan
        # higher is better for max metrics, lower is better for min metrics
        if r["Best_By_Direction"] == "max":
            return float(val)
        else:
            return -float(val)  # invert so "higher is better" for sorting

    summary_df["SortScore"] = summary_df.apply(row_sort_score, axis=1)

    summary_df = summary_df.sort_values(
        by=["Best_By_Priority", "SortScore"],
        ascending=[True, False],
        na_position="last"
    ).reset_index(drop=True)

    summary_df.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "="*90)
    print("FINAL STRUCTURE-FUNCTION PERFORMANCE SUMMARY (robust to missing metrics)")
    print("="*90)
    print(summary_df.drop(columns=["Best_By_Priority", "SortScore"]).to_string(index=False))
    print("="*90)
    print(f"Summary saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    summarize_all_logs()
