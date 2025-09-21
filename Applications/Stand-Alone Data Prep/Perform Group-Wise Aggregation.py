import pandas as pd
from tkinter import filedialog, Tk

def custom_mean(series):
    series = series.dropna()
    return series.mean() if len(series) > 0 else pd.NA

def main():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Excel or CSV file",
        filetypes=[
            ("Excel files", "*.xlsx *.xls"),
            ("CSV files", "*.csv"),
            ("All files", "*.*")
        ]
    )
    root.destroy()

    if not file_path:
        print("❌ No file selected.")
        return

    # --- Chunked reading for CSV ---
    if file_path.lower().endswith(".csv"):
        chunks = []
        for chunk in pd.read_csv(file_path, chunksize=100000):
            chunk = chunk[~chunk.iloc[:, 0].astype(str).str.lower().str.strip().eq("date")]
            chunk.iloc[:, 0] = pd.to_datetime(chunk.iloc[:, 0], errors="coerce")
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    elif file_path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(file_path)
        df = df[~df.iloc[:, 0].astype(str).str.lower().str.strip().eq("date")]
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    else:
        raise ValueError("Unsupported file type selected.")

    date_col = df.columns[0]
    value_cols = df.columns[1:]

    # --- Group and aggregate ---
    grouped = df.groupby(date_col).agg({col: custom_mean for col in value_cols}).reset_index()
    grouped[value_cols] = grouped[value_cols].round(2)

    # --- Save output ---
    save_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Save averaged output as..."
    )

    if save_path:
        grouped.to_csv(save_path, index=False)
        print(f"✅ Exported to {save_path}")
    else:
        print("❌ Save cancelled.")

if __name__ == "__main__":
    main()
