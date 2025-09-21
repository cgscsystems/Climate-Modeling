#Last known good csv link, though noaa links are REALLy unstable.
#https://psl.noaa.gov/data/timeseries/month/data/nino12.long.anom.csv

import pandas as pd
from tkinter import Tk, filedialog, messagebox
import re
from datetime import date

def coerce_oni(x):
    s = (str(x).strip()
         .replace("\u2212", "-")  # unicode minus → hyphen
         .replace(",", ""))       # strip thousands separators if any
    s = re.sub(r"[^0-9.\-]+", "", s)
    try:
        return float(s)
    except:
        return float("nan")

def intensity_from_oni(v):
    if pd.isna(v):
        return "Neutral"
    a = abs(v)
    if a >= 1.5:
        return "Strong"
    elif a >= 1.0:
        return "Medium"
    elif a >= 0.5:
        return "Weak"
    else:
        return "Neutral"

def main():
    root = Tk(); root.withdraw()

    src = filedialog.askopenfilename(
        title="Select monthly ONI CSV (date, value)",
        filetypes=[("CSV files","*.csv"), ("All files","*.*")]
    )
    if not src:
        print("No file selected."); return

    # Flexible read: auto-detect delimiter, headerless friendly
    try:
        raw = pd.read_csv(src, sep=None, engine="python", header=None)
    except Exception:
        raw = pd.read_csv(src, header=None)

    if raw.shape[1] < 2:
        messagebox.showerror("Error", "Expected at least two columns: date and ONI.")
        return

    df = raw.iloc[:, :2].copy()
    df.columns = ["date", "oni"]

    # Parse date and normalize to first-of-month
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["date"] = df["date"].dt.to_period("M").dt.to_timestamp(how="start")  # first day of month

    # Coerce ONI to numeric robustly
    df["oni"] = df["oni"].map(coerce_oni)

    # Keep valid rows, sort, format date as YYYY-MM-01
    df = df.dropna(subset=["date", "oni"]).sort_values("date").reset_index(drop=True)
    df["intensity"] = df["oni"].apply(intensity_from_oni)
    df["date"] = df["date"].dt.strftime("%Y-%m-01")

    save = filedialog.asksaveasfilename(
        title="Save ENSO intensity CSV",
        defaultextension=".csv",
        initialfile=f"enso monthly intensity {date.today().isoformat()}.csv",
        filetypes=[("CSV files","*.csv")]
    )
    if not save:
        print("Save cancelled."); return

    df[["date", "oni", "intensity"]].to_csv(save, index=False, encoding="utf-8-sig")
    messagebox.showinfo("Done", f"Saved {len(df)} rows to:\n{save}")

if __name__ == "__main__":
    main()
