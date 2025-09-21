import pandas as pd
import re
import os
import tempfile
from tkinter import Tk, filedialog
import requests
from bs4 import BeautifulSoup
from io import StringIO
import unicodedata
import tkinter as tk

def scrape_data_set():
    BASE_URL = "https://dd.weather.gc.ca/climate/observations/daily/csv/ON/"
    STATION_RANGES = {
        '6105978': (2000, 2025),
        '7032685': (2018, 2025)
    }

    try:
        resp = requests.get(BASE_URL)
        resp.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to fetch index page: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    csv_links = [a['href'] for a in soup.find_all('a') if a.get('href', '').endswith('.csv')]

    target_links = []
    for link in csv_links:
        for station_id, (start_yr, end_yr) in STATION_RANGES.items():
            if f"_{station_id}_" in link and link.endswith("_P1D.csv"):
                try:
                    year = int(link.split("_")[4])
                    if start_yr <= year <= end_yr:
                        target_links.append(link)
                except (IndexError, ValueError):
                    print(f"⚠️ Unexpected filename format: {link}")

    all_dfs = []
    for link in sorted(target_links):
        file_url = BASE_URL + link
        print(f"Fetching {link}...")
        try:
            r = requests.get(file_url)
            r.raise_for_status()
            df = pd.read_csv(StringIO(r.text))
            all_dfs.append(df)
        except Exception as e:
            print(f"⚠️ Skipped {link}: {e}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True, sort=False)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        combined_df.to_csv(temp_file.name, index=False)
        return temp_file.name
    else:
        print("❌ No valid data retrieved.")
        return None

def clean_data(input_csv):
    df = pd.read_csv(input_csv, dtype=str, encoding="utf-8")

    def clean_header(col):
        col = unicodedata.normalize('NFKD', col).encode('ascii', 'ignore').decode('ascii')
        col = col.strip()
        return re.sub(r"[^\w\-. ]", "", col)

    df.columns = [clean_header(col) for col in df.columns]

    for col in df.columns:
        df[col] = df[col].astype(str).apply(
            lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').decode('ascii').strip()
        )
        df[col] = df[col].str.replace(r'[\r\n\t\x00-\x1f\x7f-\x9f]', '', regex=True)
        df[col] = df[col].apply(lambda x: re.sub(r"[^\d\-.]", "", x))

    df.replace('', pd.NA, inplace=True)
    df.dropna(axis=1, how='all', inplace=True)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    df.to_csv(temp_file.name, index=False)
    return temp_file.name

def select_date_column(input_csv):
    df = pd.read_csv(input_csv, dtype=str)
    columns = list(df.columns)

    def on_ok():
        idx = listbox.curselection()
        root.selected_col = columns[idx[0]] if idx else None
        root.destroy()

    root = tk.Tk()
    root.title("Select Date Column")
    tk.Label(root, text="Select your date column:").pack()
    listbox = tk.Listbox(root, selectmode=tk.SINGLE, width=80)
    for col in columns:
        listbox.insert(tk.END, col)
    listbox.pack()
    tk.Button(root, text="OK", command=on_ok).pack()
    root.mainloop()

    selected = getattr(root, 'selected_col', None)
    if selected not in columns:
        print("❌ Invalid column selected.")
        return None

    cols = [selected] + [c for c in columns if c != selected]
    df = df[cols]
    df.rename(columns={selected: "date"}, inplace=True)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    df.to_csv(temp_file.name, index=False)
    return temp_file.name

def select_columns_to_drop(input_csv):
    df = pd.read_csv(input_csv, dtype=str)
    columns = list(df.columns)

    def on_ok():
        selected = [columns[i] for i in listbox.curselection()]
        root.selected_cols = selected
        root.destroy()

    root = tk.Tk()
    root.title("Select Columns to Drop")
    tk.Label(root, text="Select columns to drop:").pack()
    listbox = tk.Listbox(root, selectmode=tk.MULTIPLE, width=80)
    for col in columns:
        listbox.insert(tk.END, col)
    listbox.pack()
    tk.Button(root, text="OK", command=on_ok).pack()
    root.mainloop()

    drop_cols = getattr(root, 'selected_cols', [])
    df.drop(columns=drop_cols, inplace=True)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    df.to_csv(temp_file.name, index=False)
    return temp_file.name

def drop_problem_columns(input_csv):
    df = pd.read_csv(input_csv)

    for col in df.columns:
        if col == "date":
            dt = pd.to_datetime(df[col], errors='coerce')
            if dt.isna().all():
                df.drop(columns=[col], inplace=True)
            else:
                df[col] = dt
        else:
            num = pd.to_numeric(df[col], errors='coerce')
            if num.isna().all():
                df.drop(columns=[col], inplace=True)
            else:
                df[col] = num

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    df.to_csv(temp_file.name, index=False)
    return temp_file.name

def groupwise_aggregation(input_csv):
    df = pd.read_csv(input_csv)
    df.columns = [col.strip().lower() for col in df.columns]
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(axis=1, how='all')
    df['year'] = df['date'].apply(lambda d: d.year + 1 if d.month == 12 else d.year).astype('Int64')
    df = df[df['year'].notna()]
    df['md'] = df['date'].dt.strftime('%m-%d')
    df = df[df['md'] != '02-29']
    anchor = pd.to_datetime("2001-12-01")
    df['day_of_year'] = (pd.to_datetime('2001-' + df['md'], format='%Y-%m-%d', errors='coerce') - anchor).dt.days % 365 + 1
    df.drop(columns='md', inplace=True)

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    df.to_csv(temp_file.name, index=False)
    return temp_file.name

def calculate_columns(input_csv):
    df = pd.read_csv(input_csv)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    value_cols = [col for col in df.columns if col not in ['date', 'year', 'day_of_year']]
    df['MM-DD'] = df['date'].dt.strftime('%m-%d')
    df['Year'] = df['date'].dt.year

    for col in value_cols:
        climatology = df.groupby('MM-DD')[col].mean()
        df[f'{col}_DeltaFromAvg'] = df.apply(
            lambda row: row[col] - climatology.get(row['MM-DD'], pd.NA) if pd.notna(row[col]) else pd.NA,
            axis=1
        )

    for col in value_cols:
        df[f'{col}_Percentile'] = pd.to_numeric(df[col], errors='coerce').rank(pct=True) * 100
        df[f'{col}_Percentile'] = df[f'{col}_Percentile'].round(2)

    for col in value_cols:
        mean = df[col].mean(skipna=True)
        std = df[col].std(skipna=True)
        df[f'{col}_Zscore'] = df[col].apply(
            lambda x: round((x - mean) / std, 2) if pd.notna(x) and std != 0 else pd.NA
        )

    df.drop(columns=['MM-DD', 'Year'], inplace=True)
    return df

def main():
    temp_files = []

    merged_csv = scrape_data_set()
    if not merged_csv:
        return
    temp_files.append(merged_csv)

    cleaned_csv = clean_data(merged_csv)
    temp_files.append(cleaned_csv)

    dated_csv = select_date_column(cleaned_csv)
    if not dated_csv:
        return
    temp_files.append(dated_csv)

    filtered_csv = select_columns_to_drop(dated_csv)
    temp_files.append(filtered_csv)

    cleaned_filtered_csv = drop_problem_columns(filtered_csv)
    temp_files.append(cleaned_filtered_csv)

    aggregated_csv = groupwise_aggregation(cleaned_filtered_csv)
    temp_files.append(aggregated_csv)

    final_df = calculate_columns(aggregated_csv)

    root = Tk()
    root.withdraw()
    save_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Save completed CSV as..."
    )
    root.destroy()

    if save_path:
        final_df.to_csv(save_path, index=False)
        print(f"✅ Completed CSV saved to: {save_path}")
    else:
        print("❌ Save cancelled.")

    for f in temp_files:
        try:
            os.remove(f)
        except Exception as e:
            print(f"⚠️ Failed to delete temp file {f}: {e}")

if __name__ == "__main__":
    main()
