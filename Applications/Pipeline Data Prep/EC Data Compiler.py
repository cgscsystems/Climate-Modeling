# Full GUI + ETL pipeline script, optimized for large datasets
# Dependencies: pandas, requests, bs4, tkinter

from logging import root
import pandas as pd
import re
import os
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox
import requests
from bs4 import BeautifulSoup
from io import StringIO
import unicodedata
import math
import sys

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def scrape_data_set(station_ranges: dict, province_code: str):
    BASE_URL = f"https://dd.weather.gc.ca/climate/observations/daily/csv/{province_code}/"
    try:
        resp = requests.get(BASE_URL)
        resp.raise_for_status()
    except Exception as e:
        print(f"❌ Failed to access {BASE_URL}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    csv_links = []
    from bs4 import Tag
    for a in soup.find_all('a'):
        if isinstance(a, Tag):
            href = a.get('href')
            if isinstance(href, str) and href.endswith('.csv'):
                csv_links.append(href)

    target_links = []
    for link in csv_links:
        for station_id, (start_yr, end_yr) in station_ranges.items():
            if isinstance(link, str) and f"_{station_id}_" in link and link.endswith("_P1D.csv"):
                try:
                    year = int(link.split("_")[4])
                    if start_yr <= year <= end_yr:
                        target_links.append(link)
                except Exception:
                    continue

    # --- Prompt user for download directory ---
    download_dir = filedialog.askdirectory(title="Select download directory for original CSVs")
    if not download_dir:
        print("❌ No download directory selected.")
        return None

    # Use center station name for folder
    center_station_id = list(station_ranges.keys())[0]
    station_folder = os.path.join(download_dir, f"station_{center_station_id}")
    os.makedirs(station_folder, exist_ok=True)

    # Download each CSV to the station folder
    local_csv_paths = []
    for link in sorted(target_links):
        file_url = BASE_URL + link
        local_path = os.path.join(station_folder, link)
        print(f"📥 Downloading {link} to {local_path}...")
        try:
            r = requests.get(file_url, stream=True)
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            local_csv_paths.append(local_path)
        except Exception as e:
            print(f"⚠️ Skipped {link}: {e}")

    # --- Merge all downloaded CSVs into one temp file for processing ---
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    header_written = False
    for csv_path in local_csv_paths:
        with open(csv_path, 'r', encoding='latin1') as f:
            for i, line in enumerate(f):
                if i == 0:
                    if not header_written:
                        temp_file.write(line.encode('latin1'))
                        header_written = True
                else:
                    temp_file.write(line.encode('latin1'))
    temp_file.close()
    return temp_file.name

def auto_detect_date_column(df):
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors='coerce')
        if parsed.notna().mean() > 0.9:
            return col
    return None

def auto_drop_redundant_columns(df, date_col):
    # Drop columns with all nulls
    df = df.dropna(axis=1, how='all')
    # Drop columns with only one unique value
    for col in df.columns:
        if df[col].nunique(dropna=True) == 1:
            df = df.drop(columns=[col])
    # Drop columns that duplicate date info
    if date_col:
        year_col = df[date_col].apply(lambda x: pd.to_datetime(x, errors='coerce').year if pd.notna(x) else None)
        day_col = df[date_col].apply(lambda x: pd.to_datetime(x, errors='coerce').day if pd.notna(x) else None)
        for col in df.columns:
            if df[col].equals(year_col) or df[col].equals(day_col):
                df = df.drop(columns=[col])
    return df

# --- Cleaned column names (after diacritics removal) ---
DATE_COL_CLEAN = "datetime"
KEEP_COLS_CLEAN = [
    "datetime",
    "max temp c",
    "min temp c",
    "mean temp c",
    "heat deg days c",
    "cool deg days c",
    "total rain mm",
    "total snow cm",
    "snow on grnd cm",
    "spd of max gust kmh"
]

def clean_data(input_csv):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    reader = pd.read_csv(input_csv, dtype=str, chunksize=100000, encoding="latin1")
    with open(temp_file.name, 'w', encoding='latin1') as out_f:
        for i, chunk in enumerate(reader):
            # Clean headers only once
            if i == 0:
                def clean_header(col):
                    col = unicodedata.normalize('NFKD', col).encode('ascii', 'ignore').decode('ascii')
                    col = col.strip()
                    col = re.sub(r"[^\w\-. ]", "", col)
                    return col.lower()
                chunk.columns = [clean_header(col) for col in chunk.columns]
                out_f.write(','.join(chunk.columns) + '\n')
            for col in chunk.columns:
                chunk[col] = chunk[col].apply(lambda x: unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode('ascii').strip())
                chunk[col] = chunk[col].astype(str).apply(lambda x: re.sub(r'[\x00-\x1F]+', '', x))
            chunk.replace('', pd.NA, inplace=True)
            chunk.dropna(axis=1, how='all', inplace=True)
            chunk.to_csv(out_f, header=False, index=False)
    return temp_file.name

def preprocess_columns(input_csv, date_col_name, keep_cols):
    # Only keep columns that exist in the file and are in keep_cols
    df = pd.read_csv(input_csv, dtype=str, nrows=1)
    actual_cols = [col for col in keep_cols if col in df.columns]
    if date_col_name not in df.columns:
        raise Exception(f"Date column '{date_col_name}' not found in file.")
    # Move and rename date column to 'date'
    cols = ["date"] + [c for c in actual_cols if c != date_col_name]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    reader = pd.read_csv(input_csv, dtype=str, chunksize=100000)
    with open(temp_file.name, 'w', encoding='latin1') as out_f:
        out_f.write(','.join(cols) + '\n')
        for chunk in reader:
            chunk = chunk[[col for col in [date_col_name] + actual_cols if col in chunk.columns]]
            chunk.rename(columns={date_col_name: "date"}, inplace=True)
            chunk = chunk[cols]
            chunk.to_csv(out_f, header=False, index=False)
    return temp_file.name

def drop_problem_columns(input_csv):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    reader = pd.read_csv(input_csv, chunksize=100000)
    with open(temp_file.name, 'w', encoding='latin1') as out_f:
        for i, chunk in enumerate(reader):
            if i == 0:
                out_f.write(','.join(chunk.columns) + '\n')
            for col in chunk.columns:
                if col == "date":
                    dt = pd.to_datetime(chunk[col], errors='coerce')
                    if dt.isna().all():
                        chunk.drop(columns=[col], inplace=True)
                    else:
                        chunk[col] = dt
                else:
                    num = pd.to_numeric(chunk[col], errors='coerce')
                    if num.isna().all():
                        chunk.drop(columns=[col], inplace=True)
                    else:
                        chunk[col] = num
            chunk.to_csv(out_f, header=False, index=False)
    return temp_file.name

def groupwise_aggregation(input_csv):
    agg_chunks = []
    for chunk in pd.read_csv(input_csv, chunksize=100000):
        chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
        chunk = chunk.dropna(axis=1, how='all')
        chunk['year'] = chunk['date'].apply(lambda d: d.year + 1 if d.month == 12 else d.year).astype('Int64')
        chunk = chunk[chunk['year'].notna()]
        chunk['md'] = chunk['date'].dt.strftime('%m-%d')
        chunk = chunk[chunk['md'] != '02-29']
        anchor = pd.to_datetime("2001-12-01")
        chunk['day_of_year'] = (pd.to_datetime('2001-' + chunk['md'], format='%Y-%m-%d', errors='coerce') - anchor).dt.days % 365 + 1
        chunk.drop(columns='md', inplace=True)
        # Only aggregate numeric columns, exclude 'date', 'year', 'day_of_year'
        numeric_cols = chunk.select_dtypes(include='number').columns.tolist()
        value_cols = [col for col in numeric_cols if col not in ['year', 'day_of_year']]
        grouped = chunk.groupby(['date', 'year', 'day_of_year'])[value_cols].mean().reset_index()
        grouped[value_cols] = grouped[value_cols].round(2)
        agg_chunks.append(grouped)
    df = pd.concat(agg_chunks, ignore_index=True)
    # Final aggregation to combine duplicate dates across chunks
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    value_cols = [col for col in numeric_cols if col not in ['year', 'day_of_year']]
    grouped = df.groupby(['date', 'year', 'day_of_year'])[value_cols].mean().reset_index()
    grouped[value_cols] = grouped[value_cols].round(2)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    grouped.to_csv(temp_file.name, index=False)
    return temp_file.name

def calculate_columns(input_csv):
    df = pd.read_csv(input_csv)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    value_cols = [col for col in numeric_cols if col not in ['year', 'day_of_year']]
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

def main_gui():
    global province_var
    global province_menu
    global station_listbox
    global radius_var

    root = tk.Tk()
    root.withdraw()
    root.title("Climate Station Data Extractor")

    province_var = tk.StringVar()  # <-- Now safe to initialize
    province_menu = None
    station_listbox = None
    radius_var = tk.StringVar(value="1 km")

    data: dict[str, pd.DataFrame | None] = {'df': None}

    def run_from_directory():
        dir_path = filedialog.askdirectory(title="Select directory containing CSVs")
        if not dir_path:
            messagebox.showerror("Error", "No directory selected.")
            return
        # Merge all CSVs in the directory into one temp file
        csv_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.lower().endswith('.csv')]
        if not csv_files:
            messagebox.showerror("Error", "No CSV files found in the selected directory.")
            return
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        header_written = False
        for csv_path in csv_files:
            with open(csv_path, 'r', encoding='latin1') as f:
                for i, line in enumerate(f):
                    if i == 0:
                        if not header_written:
                            temp_file.write(line.encode('latin1'))
                            header_written = True
                    else:
                        temp_file.write(line.encode('latin1'))
        temp_file.close()
        process_pipeline(temp_file.name)

    def process_pipeline(merged_file):
        temp_files = []
        try:
            cleaned_file = clean_data(merged_file)
            temp_files.append(cleaned_file)
            preprocessed_file = preprocess_columns(cleaned_file, DATE_COL_CLEAN, KEEP_COLS_CLEAN)
            temp_files.append(preprocessed_file)
            scrubbed_file = drop_problem_columns(preprocessed_file)
            temp_files.append(scrubbed_file)
            aggregated_file = groupwise_aggregation(scrubbed_file)
            temp_files.append(aggregated_file)
            final_df = calculate_columns(aggregated_file)
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
            if not save_path:
                print("❌ Save cancelled.")
                root.destroy()  # <-- Close window if cancelled
                return
            final_df.to_csv(save_path, index=False)
            messagebox.showinfo("Saved", f"✅ Final CSV saved to:\n{save_path}")
        except Exception as e:
            messagebox.showerror("Processing Error", str(e))
        finally:
            for f in temp_files + [merged_file]:
                try:
                    os.remove(f)
                except Exception:
                    pass
            try:
                root.destroy()  # <-- Always close window at the end
            except Exception:
                pass

    def process_data():
        global province_var
        global radius_var
        global station_listbox
        global province_menu
        try:
            if station_listbox is None:
                messagebox.showerror("Error", "Station listbox is not initialized.")
                root.destroy()
                sys.exit()
            idx = station_listbox.curselection()
            if not idx:
                messagebox.showwarning("No Station Selected", "Please select a center station.")
                root.destroy()
                sys.exit()

            root.destroy()

            radius_km = int(radius_var.get().replace(" km", ""))
            prov = province_var.get().strip()
            prov_code = PROVINCE_CODE_MAP.get(prov.upper())
            if not prov_code:
                messagebox.showerror("Error", f"Unknown province: {prov}")
                sys.exit()

            if data['df'] is None:
                messagebox.showerror("Error", "No station data loaded.")
                sys.exit()

            selected_row = data['df'][data['df']['Province'] == prov].iloc[idx[0]]
            lat1, lon1 = selected_row['Latitude'], selected_row['Longitude']

            results = []
            for _, row in data['df'].iterrows():
                lat2, lon2 = row['Latitude'], row['Longitude']
                if pd.isna(lat2) or pd.isna(lon2):
                    continue
                dist = haversine(lat1, lon1, lat2, lon2)
                if dist <= radius_km:
                    results.append((row['Climate ID'], int(row['DLY First Year']), int(row['DLY Last Year'])))

            if not results:
                messagebox.showinfo("No Matches", f"No stations found within {radius_km} km.")
                sys.exit()

            station_ranges = {sid: (start, end) for sid, start, end in results}
            merged_file = scrape_data_set(station_ranges, prov_code)
            if not merged_file:
                sys.exit()

            process_pipeline(merged_file)

        except Exception as e:
            messagebox.showerror("Processing Error", str(e))
            try:
                root.destroy()
            except Exception:
                pass
            sys.exit()

    def load_file():
        path = filedialog.askopenfilename(title="Select station list CSV", filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        try:
            df = pd.read_csv(path, dtype=str)
            df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
            df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
            df['DLY First Year'] = pd.to_numeric(df['DLY First Year'], errors='coerce')
            df['DLY Last Year'] = pd.to_numeric(df['DLY Last Year'], errors='coerce')
            data['df'] = df
            populate_provinces()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")

    def populate_provinces():
        global province_var
        global province_menu
        if data['df'] is None:
            messagebox.showerror("Error", "No station data loaded.")
            return
        provinces = sorted(data['df']['Province'].dropna().unique())
        if province_menu is None:
            province_menu = tk.OptionMenu(root, province_var, *provinces)
            province_menu.pack()
        else:
            province_menu['menu'].delete(0, 'end')
            for p in provinces:
                province_menu['menu'].add_command(label=p, command=tk._setit(province_var, p, update_station_list))
        if provinces:
            province_var.set(provinces[0])
            update_station_list()

    def update_station_list(*args):
        prov = province_var.get()
        if data['df'] is None:
            return
        filtered = data['df'][data['df']['Province'] == prov]
        filtered = filtered.sort_values(by="Station Name")
        if station_listbox is not None:
            station_listbox.delete(0, tk.END)
            for _, row in filtered.iterrows():
                station_listbox.insert(tk.END, f"{row['Station Name']} [{row['Climate ID']}]")

    # --- UI Choice ---
    def choose_mode():
        mode_win = tk.Toplevel(root)
        mode_win.title("Choose Data Source")
        tk.Label(mode_win, text="How would you like to start?").pack(pady=10)
        tk.Button(mode_win, text="Use Directory of CSVs", command=lambda: [mode_win.destroy(), run_from_directory()]).pack(pady=5)
        tk.Button(mode_win, text="Use Station Picker", command=lambda: [mode_win.destroy(), show_station_picker()]).pack(pady=5)

    def show_station_picker():
        root.deiconify()
        tk.Button(root, text="Load Station File", command=load_file).pack(pady=5)
        province_var = tk.StringVar()
        tk.Label(root, text="Select Province:").pack()
        global province_menu
        province_menu = tk.OptionMenu(root, province_var, "")
        province_menu.pack()
        tk.Label(root, text="Select Center Station:").pack()
        global station_listbox
        station_listbox = tk.Listbox(root, width=60, height=10)
        station_listbox.pack()
        tk.Label(root, text="Select Radius:").pack()
        global radius_var
        radius_var = tk.StringVar(value="1 km")
        tk.OptionMenu(root, radius_var, "1 km", "5 km", "10 km", "20 km", "50 km", "100 km").pack()
        tk.Button(root, text="Run Full Extraction", command=process_data).pack(pady=10)

    choose_mode()
    root.mainloop()

PROVINCE_CODE_MAP = {
    "ALBERTA": "AB",
    "BRITISH COLUMBIA": "BC",
    "MANITOBA": "MB",
    "NEW BRUNSWICK": "NB",
    "NEWFOUNDLAND": "NL",
    "NOVA SCOTIA": "NS",
    "ONTARIO": "ON",
    "PRINCE EDWARD ISLAND": "PE",
    "QUEBEC": "QC",
    "SASKATCHEWAN": "SK",
    "NORTHWEST TERRITORIES": "NT",
    "NUNAVUT": "NU",
    "YUKON": "YT"
}

if __name__ == "__main__":
    main_gui()
