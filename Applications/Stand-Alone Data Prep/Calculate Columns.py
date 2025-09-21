import pandas as pd
from tkinter import filedialog, Tk

# --- File Picker ---
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select the consolidated CSV file",
    filetypes=[("CSV files", "*.csv")]
)
root.destroy()

# --- Load data ---
df = pd.read_csv(file_path)

# --- Parse date column ---
date_col = df.columns[0]
df[date_col] = df[date_col].astype(str).str.strip()
df[date_col] = pd.to_datetime(df[date_col], format="%Y-%m-%d", errors="coerce")
df = df.dropna(subset=[date_col])

# --- Identify value columns ---
value_cols = df.columns[1:]

# --- Add helper columns ---
df['MM-DD'] = df[date_col].dt.strftime('%m-%d')
df['Year'] = df[date_col].dt.year

# --- 1. Delta from day-of-year (MM-DD) average ---
for col in value_cols:
    climatology = df.groupby('MM-DD')[col].mean()
    df[f'{col}_DeltaFromAvg'] = df.apply(
        lambda row: row[col] - climatology[row['MM-DD']] if pd.notna(row[col]) else pd.NA,
        axis=1
    )

# --- 2. Optimized rolling averages ---
for col in value_cols:
    sub = df[[date_col, 'MM-DD', 'Year', col]].dropna().copy()
    for window in [5, 10, 20, 30]:
        # Create a lookup of historical means
        history = (
            sub.assign(JoinKey=sub['Year'])
            .merge(sub, on='MM-DD', suffixes=('', '_hist'))
            .query('Year_hist < JoinKey and Year_hist >= JoinKey - @window')
            .groupby([date_col])[f'{col}_hist']
            .mean()
            .rename(f'{col}_RollingAvg_{window}y')
        )
        df = df.merge(history, on=date_col, how='left')

# --- 3. Percentile rank (0–100) ---
for col in value_cols:
    df[f'{col}_Percentile'] = df[col].rank(pct=True) * 100
    df[f'{col}_Percentile'] = df[f'{col}_Percentile'].round(2)

# --- 4. Z-score ---
for col in value_cols:
    mean = df[col].mean(skipna=True)
    std = df[col].std(skipna=True)
    df[f'{col}_Zscore'] = pd.to_numeric(
        df[col].apply(
            lambda x: (x - mean) / std if pd.notna(x) and std != 0 else pd.NA
        ),
        errors='coerce'
    ).round(2)

# --- Drop helper columns ---
df.drop(columns=['MM-DD', 'Year'], inplace=True)

# --- Save File Dialog ---
root = Tk()
root.withdraw()
save_path = filedialog.asksaveasfilename(
    defaultextension=".csv",
    filetypes=[("CSV files", "*.csv")],
    title="Save enriched data as..."
)
root.destroy()

# --- Export result ---
if save_path:
    df.to_csv(save_path, index=False)
    print(f"✅ Enriched file saved to {save_path}")
else:
    print("❌ Save cancelled.")
