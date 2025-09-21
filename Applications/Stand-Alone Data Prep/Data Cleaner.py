import pandas as pd
import re
import os
import unicodedata
from tkinter import filedialog, Tk

def remove_diacritics(s):
    return ''.join(
        c for c in unicodedata.normalize('NFKD', str(s))
        if not unicodedata.combining(c)
    )

def clean_csv_file():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select the CSV file to clean",
        filetypes=[("CSV files", "*.csv")]
    )
    root.destroy()

    if not file_path:
        print("❌ No file selected.")
        return

    # --- Load CSV ---
    df = pd.read_csv(file_path, dtype=str)

    # --- Clean headers: remove diacritics and special characters ---
    df.columns = [remove_diacritics(col) for col in df.columns]
    df.columns = [re.sub(r'[^\w\s\-\.]', '', col) for col in df.columns]

    # --- Clean data rows ---
    for col in df.columns:
        # Remove diacritics and special characters from all cells
        df[col] = df[col].apply(remove_diacritics)
        # Remove hidden/control characters (carriage returns, newlines, tabs, etc.)
        df[col] = df[col].astype(str).apply(lambda x: re.sub(r'[\x00-\x1F]+', '', x))
        # Now remove unwanted characters (keep digits, dash, period)
        if col == df.columns[0]:  # First column: keep dash and period for dates
            df[col] = df[col].apply(lambda x: re.sub(r'[^\d\-.]', '', x))
        else:
            df[col] = df[col].apply(lambda x: re.sub(r'[^\d\-.]', '', x))

    # --- Convert numeric columns to float (except first column) ---
    for col in df.columns[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # --- Drop columns that became fully empty ---
    df.dropna(axis=1, how='all', inplace=True)

    # --- Save file next to source with _cleaned suffix ---
    folder, filename = os.path.split(file_path)
    name, ext = os.path.splitext(filename)
    save_path = os.path.join(folder, f"{name}_cleaned.csv")

    df.to_csv(save_path, index=False)
    print(f"✅ Cleaned file saved to: {save_path}")

# --- Main execution ---
if __name__ == "__main__":
    clean_csv_file()