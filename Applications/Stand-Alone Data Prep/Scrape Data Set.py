import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
from tkinter import Tk, filedialog

# --- Setup save dialog ---
def ask_save_path(default_name="_______Complete_Weather.csv"):
    root = Tk()
    root.withdraw()
    save_path = filedialog.asksaveasfilename(
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        title="Save combined weather data as...",
        initialfile=default_name
    )
    root.destroy()
    return save_path

# Source config
BASE_URL = "https://dd.weather.gc.ca/climate/observations/daily/csv/ON/" #AB/BC/MB/NB/NL/NS/NU/ON/PE/QC/SK/YT
STATION_RANGES = { #Ottawa
    '6105887': (1872, 1935),
    '6105910': (1954, 1954),
    '6105913': (1961, 1963),
    '6105938': (1955, 1961),
    '6105950': (1953, 1954),
    '6105960': (1972, 1984),
    '6105976': (1889, 2025),
    '6105978': (2000, 2025),
    '6105980': (1966, 1975),
    '7032685': (2018, 2025),
    '6105993': (1969, 1969),
    '6105995': (1953, 1954),
    '6106001': (2011, 2025),
    '6106003': (1969, 1969),
    '6106014': (1954, 1967),
    '6106052': (1953, 1979),
    '6106000': (1938, 2011),
    '6106080': (1960, 1962),
    '6106090': (1951, 1984),
    '6106098': (1972, 1975),
    '6106100': (1942, 1964),
    '6106102': (1969, 1969),
    '6106105': (1954, 1955)
}

# Get all CSV file links
resp = requests.get(BASE_URL)
resp.raise_for_status()
soup = BeautifulSoup(resp.text, "html.parser")
csv_links = [a['href'] for a in soup.find_all('a') if a['href'].endswith('.csv')]

# Filter only valid Halifax station files within year range
target_links = []
for link in csv_links:
    for station_id, (start_yr, end_yr) in STATION_RANGES.items():
        if f"_{station_id}_" in link and link.endswith("_P1D.csv"):
            try:
                year = int(link.split("_")[4])
                if start_yr <= year <= end_yr:
                    target_links.append(link)
            except Exception:
                continue

# Download and combine
all_dfs = []
for link in sorted(target_links):
    file_url = BASE_URL + link
    print(f"Fetching {link}...")
    try:
        r = requests.get(file_url)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        df['source_file'] = link
        all_dfs.append(df)
        all_dfs.append(pd.DataFrame())  # spacer row
    except Exception as e:
        print(f"⚠️ Skipped {link}: {e}")

# Prompt for save location
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True, sort=False)
    save_path = ask_save_path()
    if save_path:
        combined_df.to_csv(save_path, index=False)
        print(f"✅ Saved as {save_path}")
    else:
        print("❌ Save cancelled.")
else:
    print("❌ No valid data retrieved.")
