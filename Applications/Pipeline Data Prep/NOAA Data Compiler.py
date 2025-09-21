# NOAA Data Compiler - Station Picker Format
# Simple station list approach matching EC Data Compiler design
# Dependencies: pandas, requests, tkinter

import pandas as pd
import numpy as np
import os
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox
import requests
import math
import sys
import re
import unicodedata
from datetime import datetime
from io import StringIO

# --- NOAA-specific column names (after transformation) ---
DATE_COL_CLEAN = "date"
KEEP_COLS_CLEAN = [
    "date",
    "max_temp_c",
    "min_temp_c", 
    "mean_temp_c",
    "total_rain_mm",
    "total_snow_cm",
    "snow_depth_cm",
    "avg_wind_speed_mps",
    "max_wind_2min_mps",
    "max_wind_5sec_mps",
    "max_wind_gust_mps",
    "sunshine_pct",
    "sunshine_minutes"
]

def haversine(lat1, lon1, lat2, lon2):
    """Calculate distance between two points on Earth using Haversine formula."""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def scrape_noaa_data_set(station_id, raw_files_dir=None):
    """Download single NOAA station CSV file and optionally save raw copy."""
    BASE_URL = "https://www.ncei.noaa.gov/data/daily-summaries/access/"
    
    try:
        file_url = f"{BASE_URL}{station_id}.csv"
        print(f"  Downloading {station_id}...")
        
        response = requests.get(file_url, timeout=30)
        response.raise_for_status()
        
        # Save to temporary file for processing
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
        temp_file.write(response.content)
        temp_file.close()
        
        # Also save raw copy if directory provided
        if raw_files_dir:
            raw_file_path = os.path.join(raw_files_dir, f"{station_id}.csv")
            with open(raw_file_path, 'wb') as raw_file:
                raw_file.write(response.content)
            print(f"    Raw file saved: {raw_file_path}")
        
        return temp_file.name
        
    except Exception as e:
        print(f"  Failed to download {station_id}: {e}")
        return None

def clean_noaa_columns(df):
    """Clean and filter NOAA columns, removing compound/attribute columns."""
    # First identify and remove compound columns (those ending with _ATTRIBUTES)
    attribute_cols = [col for col in df.columns if col.endswith('_ATTRIBUTES')]
    clean_df = df.drop(columns=attribute_cols)
    
    # Also remove obvious metadata columns that don't contain measurements
    metadata_cols = ['STATION', 'LATITUDE', 'LONGITUDE', 'ELEVATION', 'NAME']
    metadata_to_drop = [col for col in metadata_cols if col in clean_df.columns and col != 'DATE']
    clean_df = clean_df.drop(columns=metadata_to_drop)
    
    print(f"    Removed {len(attribute_cols)} attribute columns and {len(metadata_to_drop)} metadata columns")
    print(f"    Remaining measurement columns: {[col for col in clean_df.columns if col != 'DATE']}")
    
    return clean_df

def transform_noaa_to_ec_format(noaa_df, station_id):
    """Transform NOAA format to EC-compatible format with improved column handling."""
    try:
        # Clean the DataFrame first - remove compound/attribute columns
        clean_df = clean_noaa_columns(noaa_df.copy())
        
        # Start with date column
        result_df = pd.DataFrame()
        if 'DATE' in clean_df.columns:
            result_df['date'] = pd.to_datetime(clean_df['DATE'], errors='coerce')
        else:
            print(f"No DATE column found for {station_id}")
            return pd.DataFrame()
        
        # Enhanced column mapping with more NOAA variables
        column_mapping = {
            # Temperature (tenths of degrees C)
            'TMAX': 'max_temp_c',           # Maximum temperature
            'TMIN': 'min_temp_c',           # Minimum temperature  
            'TAVG': 'avg_temp_c',           # Average temperature
            
            # Precipitation (tenths of mm)
            'PRCP': 'total_rain_mm',        # Precipitation (rain + melted snow)
            'SNOW': 'total_snow_cm',        # Snowfall
            'SNWD': 'snow_depth_cm',        # Snow depth
            
            # Wind (whole units)
            'AWND': 'avg_wind_speed_mps',   # Average daily wind speed (m/s)
            'WSF2': 'max_wind_2min_mps',    # Maximum 2-minute wind speed
            'WSF5': 'max_wind_5sec_mps',    # Maximum 5-second wind speed
            'WSFG': 'max_wind_gust_mps',    # Maximum wind gust speed
            
            # Other measurements
            'PSUN': 'sunshine_pct',         # Daily percent possible sunshine
            'TSUN': 'sunshine_minutes',     # Daily total sunshine (minutes)
        }
        
        # Process each available column
        for noaa_col, ec_col in column_mapping.items():
            if noaa_col in clean_df.columns:
                # Convert to numeric, handling NOAA's various formats (spaces, etc.)
                values = pd.to_numeric(clean_df[noaa_col], errors='coerce')
                
                # Apply unit conversions based on NOAA documentation
                if noaa_col in ['TMAX', 'TMIN', 'TAVG']:
                    # Temperature: convert from tenths of degrees C to degrees C
                    result_df[ec_col] = values / 10.0
                elif noaa_col in ['PRCP']:
                    # Precipitation: convert from tenths of mm to mm
                    result_df[ec_col] = values / 10.0
                elif noaa_col in ['SNOW', 'SNWD']:
                    # Snow: already in mm, convert to cm
                    result_df[ec_col] = values / 10.0
                elif noaa_col in ['AWND', 'WSF2', 'WSF5', 'WSFG']:
                    # Wind speeds: convert from tenths of m/s to m/s
                    result_df[ec_col] = values / 10.0
                else:
                    # Other measurements: use as-is
                    result_df[ec_col] = values
        
        # Calculate derived measurements
        if 'max_temp_c' in result_df.columns and 'min_temp_c' in result_df.columns:
            # Calculate mean temperature if not already provided
            if 'avg_temp_c' not in result_df.columns:
                result_df['mean_temp_c'] = (result_df['max_temp_c'] + result_df['min_temp_c']) / 2.0
            else:
                result_df['mean_temp_c'] = result_df['avg_temp_c']
        
        # Add metadata
        result_df['source_station'] = station_id
        
        # Remove rows with no valid date
        result_df = result_df.dropna(subset=['date'])
        
        # Show what we successfully processed
        numeric_cols = result_df.select_dtypes(include='number').columns.tolist()
        valid_data_cols = [col for col in numeric_cols if result_df[col].notna().sum() > 0]
        print(f"    Successfully processed: {valid_data_cols}")
        
        return result_df
        
    except Exception as e:
        print(f"Transform error for {station_id}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def clean_data(input_csv):
    """Clean and standardize NOAA data with unicode normalization and encoding cleanup."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    reader = pd.read_csv(input_csv, dtype=str, chunksize=100000, encoding="utf-8")
    
    with open(temp_file.name, 'w', encoding='utf-8') as out_f:
        for i, chunk in enumerate(reader):
            # Clean headers only once
            if i == 0:
                def clean_header(col):
                    # Remove unicode diacritics and normalize to ASCII
                    cleaned = unicodedata.normalize('NFKD', str(col)).encode('ascii', 'ignore').decode('ascii')
                    # Clean up common problematic characters and standardize spacing
                    cleaned = re.sub(r'[\x00-\x1F]+', '', cleaned)  # Remove control characters
                    cleaned = re.sub(r'\s+', '_', cleaned.strip())  # Normalize whitespace to underscores
                    cleaned = cleaned.lower()  # Standardize to lowercase
                    return cleaned
                
                chunk.columns = [clean_header(col) for col in chunk.columns]
                out_f.write(','.join(chunk.columns) + '\n')
            
            # Clean data values in each column
            for col in chunk.columns:
                # Unicode normalization and ASCII conversion
                chunk[col] = chunk[col].apply(
                    lambda x: unicodedata.normalize('NFKD', str(x)).encode('ascii', 'ignore').decode('ascii').strip()
                    if pd.notna(x) else x
                )
                # Remove control characters
                chunk[col] = chunk[col].astype(str).apply(lambda x: re.sub(r'[\x00-\x1F]+', '', x))
            
            # Standardize null values
            chunk.replace('', pd.NA, inplace=True)
            chunk.replace(['NULL', 'null', 'Null', 'N/A', 'n/a'], pd.NA, inplace=True)
            
            # Drop columns that are completely empty
            chunk.dropna(axis=1, how='all', inplace=True)
            
            # Write cleaned chunk
            chunk.to_csv(out_f, header=False, index=False)
    
    return temp_file.name

def preprocess_columns(input_csv, date_col_name, keep_cols):
    """Preprocess and filter columns, standardizing date column and keeping only specified measurement columns."""
    # Check what columns actually exist in the file
    df_sample = pd.read_csv(input_csv, dtype=str, nrows=1)
    actual_cols = [col for col in keep_cols if col in df_sample.columns]
    
    if date_col_name not in df_sample.columns:
        raise Exception(f"Date column '{date_col_name}' not found in file. Available: {list(df_sample.columns)}")
    
    # Define output columns (date first, then available measurement columns)
    output_cols = ["date"] + [c for c in actual_cols if c != date_col_name]
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    reader = pd.read_csv(input_csv, dtype=str, chunksize=100000)
    
    with open(temp_file.name, 'w', encoding='utf-8') as out_f:
        # Write header
        out_f.write(','.join(output_cols) + '\n')
        
        for chunk in reader:
            # Select and reorder columns
            available_input_cols = [col for col in [date_col_name] + actual_cols if col in chunk.columns]
            chunk = chunk[available_input_cols]
            
            # Rename date column to standardized name
            chunk.rename(columns={date_col_name: "date"}, inplace=True)
            
            # Ensure columns are in the right order
            chunk = chunk[output_cols]
            
            # Write processed chunk
            chunk.to_csv(out_f, header=False, index=False)
    
    return temp_file.name

def drop_problem_columns(input_csv):
    """Identify and remove problematic columns that can't be processed."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    reader = pd.read_csv(input_csv, chunksize=100000)
    
    with open(temp_file.name, 'w', encoding='utf-8') as out_f:
        for i, chunk in enumerate(reader):
            if i == 0:
                # Write headers for first chunk
                out_f.write(','.join(chunk.columns) + '\n')
            
            # Validate and clean each column
            columns_to_drop = []
            
            for col in chunk.columns:
                if col == "date":
                    # Validate date column
                    dt = pd.to_datetime(chunk[col], errors='coerce')
                    if dt.isna().all():
                        print(f"    Warning: Date column '{col}' has no valid dates, dropping")
                        columns_to_drop.append(col)
                    else:
                        chunk[col] = dt
                else:
                    # Validate numeric columns
                    num = pd.to_numeric(chunk[col], errors='coerce')
                    if num.isna().all():
                        print(f"    Warning: Column '{col}' has no numeric data, dropping")
                        columns_to_drop.append(col)
                    else:
                        chunk[col] = num
            
            # Drop problematic columns
            if columns_to_drop:
                chunk.drop(columns=columns_to_drop, inplace=True)
            
            # Write cleaned chunk
            chunk.to_csv(out_f, header=False, index=False)
    
    return temp_file.name

def groupwise_aggregation(input_csv):
    """Perform groupwise aggregation with year adjustment and day-of-year calculation."""
    agg_chunks = []
    
    for chunk in pd.read_csv(input_csv, chunksize=100000):
        # Ensure date is datetime
        chunk['date'] = pd.to_datetime(chunk['date'], errors='coerce')
        chunk = chunk.dropna(subset=['date'])
        
        # Drop completely empty columns
        chunk = chunk.dropna(axis=1, how='all')
        
        # Year adjustment: December belongs to next year
        chunk['year'] = chunk['date'].apply(lambda d: d.year + 1 if d.month == 12 else d.year).astype('Int64')
        chunk = chunk[chunk['year'].notna()]
        
        # Month-day string for processing
        chunk['md'] = chunk['date'].dt.strftime('%m-%d')
        
        # Remove leap year dates (Feb 29) for consistent yearly comparisons
        chunk = chunk[chunk['md'] != '02-29']
        
        # Calculate day of year with December 1st as anchor (day 1)
        anchor = pd.to_datetime("2001-12-01")
        chunk['day_of_year'] = (
            pd.to_datetime('2001-' + chunk['md'], format='%Y-%m-%d', errors='coerce') - anchor
        ).dt.days % 365 + 1
        
        # Drop the temporary md column
        chunk.drop(columns='md', inplace=True)
        
        # Identify numeric columns for aggregation (exclude date, year, day_of_year)
        numeric_cols = chunk.select_dtypes(include='number').columns.tolist()
        value_cols = [col for col in numeric_cols if col not in ['year', 'day_of_year']]
        
        if not value_cols:
            print("    Warning: No numeric columns found for aggregation")
            continue
        
        # Group by date and aggregate numeric columns
        grouped = chunk.groupby(['date', 'year', 'day_of_year'])[value_cols].mean().reset_index()
        grouped[value_cols] = grouped[value_cols].round(2)
        agg_chunks.append(grouped)
    
    if not agg_chunks:
        print("    Error: No data chunks could be processed")
        return None
    
    # Combine all chunks
    df = pd.concat(agg_chunks, ignore_index=True)
    
    # Final aggregation to combine any duplicate dates across chunks
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    value_cols = [col for col in numeric_cols if col not in ['year', 'day_of_year']]
    
    final_grouped = df.groupby(['date', 'year', 'day_of_year'])[value_cols].mean().reset_index()
    final_grouped[value_cols] = final_grouped[value_cols].round(2)
    
    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    final_grouped.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    return temp_file.name

def calculate_columns(input_csv):
    """Calculate climatology, anomalies, percentiles, and z-scores for enhanced analysis."""
    df = pd.read_csv(input_csv)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Identify numeric value columns (exclude date, year, day_of_year)
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    value_cols = [col for col in numeric_cols if col not in ['year', 'day_of_year']]
    
    if not value_cols:
        print("    Warning: No value columns found for statistical analysis")
        return df
    
    # Create month-day string for climatology calculations
    df['MM-DD'] = df['date'].dt.strftime('%m-%d')
    df['Year'] = df['date'].dt.year
    
    print(f"    Calculating climatology and anomalies for {len(value_cols)} variables...")
    
    # Calculate climatological averages and anomalies
    for col in value_cols:
        # Calculate long-term average for each day of year
        climatology = df.groupby('MM-DD')[col].mean()
        
        # Calculate daily anomalies (departure from climatological average)
        df[f'{col}_DeltaFromAvg'] = df.apply(
            lambda row: row[col] - climatology.get(row['MM-DD'], pd.NA) if pd.notna(row[col]) else pd.NA,
            axis=1
        )
    
    print(f"    Calculating percentile rankings...")
    
    # Calculate percentile rankings
    for col in value_cols:
        df[f'{col}_Percentile'] = pd.to_numeric(df[col], errors='coerce').rank(pct=True) * 100
        df[f'{col}_Percentile'] = df[f'{col}_Percentile'].round(2)
    
    print(f"    Calculating z-scores...")
    
    # Calculate z-scores (standard deviations from mean)
    for col in value_cols:
        mean = df[col].mean(skipna=True)
        std = df[col].std(skipna=True)
        df[f'{col}_Zscore'] = df[col].apply(
            lambda x: round((x - mean) / std, 2) if pd.notna(x) and std != 0 else pd.NA
        )
    
    # Clean up temporary columns
    df.drop(columns=['MM-DD', 'Year'], inplace=True)
    
    print(f"    Statistical analysis complete: {len(df)} days with enhanced metrics")
    return df

def merge_and_transform_noaa_files(file_info_list):
    """Merge and transform multiple NOAA station files with proper column alignment."""
    print("Phase 1: Analyzing column structures across all stations...")
    
    # Phase 1: Analyze all files to understand column variations
    all_columns = set()
    station_column_info = {}
    
    for file_path, station_id, start_year, end_year in file_info_list:
        try:
            df = pd.read_csv(file_path)
            clean_df = clean_noaa_columns(df)
            columns = list(clean_df.columns)
            all_columns.update(columns)
            station_column_info[station_id] = {
                'file_path': file_path,
                'columns': columns,
                'start_year': start_year,
                'end_year': end_year,
                'data_shape': clean_df.shape
            }
            print(f"  {station_id}: {len(columns)} columns, {clean_df.shape[0]} rows")
        except Exception as e:
            print(f"  Error analyzing {station_id}: {e}")
            station_column_info[station_id] = None
    
    print(f"  Total unique columns found: {len(all_columns)}")
    print(f"  Column inventory: {sorted(all_columns)}")
    
    # Phase 2: Process and transform with consistent column alignment
    print("\nPhase 2: Processing and transforming station data...")
    all_station_data = []
    
    for station_id, info in station_column_info.items():
        if info is None:
            continue
            
        print(f"  Processing {station_id}...")
        
        try:
            df = pd.read_csv(info['file_path'])
            df['DATE'] = pd.to_datetime(df['DATE'], errors='coerce')
            df = df.dropna(subset=['DATE'])
            df = df[(df['DATE'].dt.year >= info['start_year']) & (df['DATE'].dt.year <= info['end_year'])]
            
            if df.empty:
                print(f"    No data in date range for {station_id}")
                continue
            
            # Clean and transform with consistent column handling
            transformed_df = transform_noaa_to_ec_format(df, station_id)
            
            if not transformed_df.empty:
                all_station_data.append(transformed_df)
                print(f"    Processed {len(transformed_df)} days for {station_id}")
                print(f"    Output columns: {list(transformed_df.columns)}")
                
        except Exception as e:
            print(f"    Error processing {station_id}: {e}")
    
    if not all_station_data:
        print("No usable data from any station files.")
        return None
    
    print(f"\nPhase 3: Merging {len(all_station_data)} station datasets...")
    
    # Ensure all dataframes have consistent column structures before concatenation
    if len(all_station_data) > 1:
        # Get union of all columns from transformed data
        all_output_columns = set()
        for df in all_station_data:
            all_output_columns.update(df.columns)
        
        print(f"  Standardizing {len(all_output_columns)} output columns across all stations")
        
        # Standardize all dataframes to have the same columns
        standardized_data = []
        for df in all_station_data:
            # Add missing columns with NaN values
            for col in all_output_columns:
                if col not in df.columns:
                    df[col] = np.nan
            # Reorder columns consistently
            df = df[sorted(all_output_columns)]
            standardized_data.append(df)
        
        combined_df = pd.concat(standardized_data, ignore_index=True)
    else:
        combined_df = all_station_data[0]
    
    # Aggregate multi-station data by date with name-based column alignment
    print(f"  Aggregating {len(combined_df)} total records by date...")
    
    # Identify numeric columns for aggregation (exclude date and station info)
    numeric_cols = []
    for col in combined_df.columns:
        if col not in ['date', 'source_station'] and pd.api.types.is_numeric_dtype(combined_df[col]):
            numeric_cols.append(col)
    
    print(f"  Aggregating {len(numeric_cols)} numeric columns: {numeric_cols}")
    
    # Build aggregation dictionary
    agg_dict = {col: 'mean' for col in numeric_cols}
    
    # Aggregate numeric columns
    final_df = combined_df.groupby('date').agg(agg_dict).reset_index()
    
    # Handle source_station separately to combine unique stations
    if 'source_station' in combined_df.columns:
        station_agg = combined_df.groupby('date')['source_station'].agg(lambda x: ', '.join(sorted(set(str(v) for v in x if pd.notna(v))))).reset_index()
        final_df = final_df.merge(station_agg, on='date', how='left')
    
    # Round numeric columns
    for col in numeric_cols:
        if col in final_df.columns:
            final_df[col] = final_df[col].round(2)
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
    final_df.to_csv(temp_file.name, index=False)
    temp_file.close()
    
    print(f"Final merged data: {len(final_df)} days, {len(final_df.columns)} columns")
    print(f"Output columns: {list(final_df.columns)}")
    return temp_file.name

def main_gui():
    """Launch the NOAA Data Compiler GUI with station picker."""
    global province_var, province_menu, station_listbox, radius_var
    
    root = tk.Tk()
    root.withdraw()
    root.title("NOAA Climate Data Compiler")
    
    # Load station data - try default path first, then file picker fallback
    station_file = os.path.join("..", "Support CSV", "noaa_stations_ec_format.csv")
    
    if not os.path.exists(station_file):
        print(f"Station file not found at default location: {station_file}")
        
        # Fallback to file picker
        response = messagebox.askyesno("Station File Not Found", 
            "The NOAA station file was not found at the default location.\n\n" +
            "Would you like to browse for the station file?\n" +
            "(Expected file: noaa_stations_ec_format.csv)")
        
        if not response:
            return
        
        station_file = filedialog.askopenfilename(
            title="Select NOAA Station File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="noaa_stations_ec_format.csv"
        )
        
        if not station_file:
            print("No station file selected. Exiting.")
            return
    
    try:
        print(f"Loading station data from: {station_file}")
        station_df = pd.read_csv(station_file, dtype=str)
        
        # Validate required columns
        required_cols = ['Climate ID', 'Station Name', 'Province', 'Latitude', 'Longitude', 'DLY First Year', 'DLY Last Year']
        missing_cols = [col for col in required_cols if col not in station_df.columns]
        
        if missing_cols:
            messagebox.showerror("Error", f"Station file is missing required columns: {missing_cols}")
            return
        
        # Convert numeric columns
        station_df['Latitude'] = pd.to_numeric(station_df['Latitude'], errors='coerce')
        station_df['Longitude'] = pd.to_numeric(station_df['Longitude'], errors='coerce')
        station_df['DLY First Year'] = pd.to_numeric(station_df['DLY First Year'], errors='coerce')
        station_df['DLY Last Year'] = pd.to_numeric(station_df['DLY Last Year'], errors='coerce')
        
        print(f"✅ Successfully loaded {len(station_df)} NOAA stations")
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load station data: {e}")
        print(f"Error details: {e}")
        return
    
    # Initialize variables
    province_var = tk.StringVar()
    province_menu = None
    station_listbox = None
    radius_var = tk.StringVar(value="50 km")
    
    data = {'df': station_df}
    
    def process_noaa_data():
        """Process selected NOAA station data."""
        if station_listbox is None:
            messagebox.showerror("Error", "No station list available.")
            return
        
        selected_indices = station_listbox.curselection()
        if not selected_indices:
            messagebox.showerror("Error", "Please select a station.")
            return
        
        try:
            radius_text = radius_var.get()
            radius_km = float(radius_text.replace(" km", ""))
        except ValueError:
            messagebox.showerror("Error", "Please enter valid radius.")
            return
        
        try:
            # Get selected province and station
            prov = province_var.get()
            idx = selected_indices
            
            if data['df'] is None:
                messagebox.showerror("Error", "No station data loaded.")
                return
            
            # Get the selected station as center point
            province_stations = data['df'][data['df']['Province'] == prov].reset_index(drop=True)
            selected_row = province_stations.iloc[idx[0]]
            lat1, lon1 = selected_row['Latitude'], selected_row['Longitude']
            
            print(f"Center station: {selected_row['Station Name']} ({selected_row['Climate ID']})")
            print(f"Searching within {radius_km}km radius...")
            
            # Find all stations within radius
            results = []
            for _, row in data['df'].iterrows():
                lat2, lon2 = row['Latitude'], row['Longitude']
                if pd.isna(lat2) or pd.isna(lon2):
                    continue
                dist = haversine(lat1, lon1, lat2, lon2)
                if dist <= radius_km:
                    # Get max available date range for each station
                    station_start = int(row['DLY First Year']) if pd.notna(row['DLY First Year']) else 1900
                    station_end = int(row['DLY Last Year']) if pd.notna(row['DLY Last Year']) else 2025
                    results.append((row['Climate ID'], station_start, station_end))
            
            if not results:
                messagebox.showinfo("No Matches", f"No stations found within {radius_km} km.")
                return
            
            # Calculate total date range for display
            all_start_years = [start for _, start, _ in results]
            all_end_years = [end for _, _, end in results]
            earliest_year = min(all_start_years) if all_start_years else 1900
            latest_year = max(all_end_years) if all_end_years else 2025
            
            # Inform user of scope
            response = messagebox.askyesno("Processing", 
                f"Found {len(results)} stations within {radius_km}km.\n" +
                f"This will download maximum available data ({earliest_year}-{latest_year}).\n\n" +
                "Continue with download and processing?")
            
            if not response:
                return
            
            # Prompt user to select directory for saving raw files
            raw_files_base_dir = filedialog.askdirectory(
                title="Select directory to save raw NOAA station files"
            )
            
            if not raw_files_base_dir:
                messagebox.showinfo("Cancelled", "No directory selected. Raw files will not be saved.")
                raw_files_dir = None
            else:
                # Create subdirectory named after the center station
                center_station_name = selected_row['Station Name'].replace('/', '_').replace('\\', '_')
                station_folder_name = f"NOAA_Raw_{center_station_name}_{selected_row['Climate ID']}"
                raw_files_dir = os.path.join(raw_files_base_dir, station_folder_name)
                
                try:
                    os.makedirs(raw_files_dir, exist_ok=True)
                    print(f"Created raw files directory: {raw_files_dir}")
                except Exception as e:
                    messagebox.showerror("Error", f"Could not create directory: {e}")
                    raw_files_dir = None
            
            # Download data from all stations in radius
            file_info_list = []
            for station_id, start_yr, end_yr in results:
                temp_file = scrape_noaa_data_set(station_id, raw_files_dir)
                if temp_file:
                    file_info_list.append((temp_file, station_id, start_yr, end_yr))
            
            if not file_info_list:
                messagebox.showerror("Error", "Failed to download data from any station.")
                return
            
            # Process data through complete pipeline
            print("="*60)
            print("Starting Complete NOAA Data Processing Pipeline...")
            print("="*60)
            
            temp_files = []
            try:
                # Phase 1: Download and merge station data
                print("Phase 1: Downloading and merging station data...")
                merged_file = merge_and_transform_noaa_files(file_info_list)
                if not merged_file:
                    messagebox.showerror("Error", "Failed to merge station data.")
                    return
                temp_files.append(merged_file)
                
                # Phase 2: Clean and standardize data
                print("Phase 2: Cleaning and standardizing data...")
                cleaned_file = clean_data(merged_file)
                temp_files.append(cleaned_file)
                
                # Phase 3: Preprocess columns and filter
                print("Phase 3: Preprocessing columns...")
                preprocessed_file = preprocess_columns(cleaned_file, DATE_COL_CLEAN, KEEP_COLS_CLEAN)
                temp_files.append(preprocessed_file)
                
                # Phase 4: Remove problematic columns
                print("Phase 4: Validating and cleaning columns...")
                scrubbed_file = drop_problem_columns(preprocessed_file)
                temp_files.append(scrubbed_file)
                
                # Phase 5: Perform groupwise aggregation
                print("Phase 5: Performing temporal aggregation...")
                aggregated_file = groupwise_aggregation(scrubbed_file)
                if not aggregated_file:
                    messagebox.showerror("Error", "Failed during temporal aggregation.")
                    return
                temp_files.append(aggregated_file)
                
                # Phase 6: Calculate enhanced statistical metrics
                print("Phase 6: Calculating climatology and statistical metrics...")
                final_df = calculate_columns(aggregated_file)
                
                # Save final enhanced dataset to temporary file
                final_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.csv')
                final_df.to_csv(final_temp_file.name, index=False)
                final_temp_file.close()
                temp_files.append(final_temp_file.name)
                
                print("="*60)
                print("Pipeline Complete! Data ready for analysis.")
                print(f"Final dataset: {len(final_df)} days with enhanced statistical metrics")
                print("="*60)
                
                final_file = final_temp_file.name  # Use the final enhanced dataset
                
            except Exception as e:
                messagebox.showerror("Pipeline Error", f"Error in processing pipeline: {str(e)}")
                # Clean up any temporary files on error
                for temp_file in temp_files:
                    try:
                        if os.path.exists(temp_file):
                            os.unlink(temp_file)
                    except:
                        pass
                return
            
            if final_file:
                # Save processed data
                save_path = filedialog.asksaveasfilename(
                    title="Save NOAA Climate Data",
                    defaultextension=".csv",
                    filetypes=[("CSV files", "*.csv")]
                )
                if save_path:
                    import shutil
                    shutil.move(final_file, save_path)
                    
                    # Create success message with raw files info
                    success_msg = f"✅ NOAA climate data saved to:\n{save_path}"
                    if raw_files_dir:
                        station_count = len([f for f in file_info_list if f[0]])
                        success_msg += f"\n\n📁 Raw station files ({station_count} stations) saved to:\n{raw_files_dir}"
                    
                    messagebox.showinfo("Success", success_msg)
                else:
                    messagebox.showinfo("Cancelled", "Save operation cancelled.")
            else:
                messagebox.showerror("Error", "Failed to process downloaded data.")
            
            # Clean up temporary files
            for file_path, _, _, _ in file_info_list:
                try:
                    os.unlink(file_path)
                except:
                    pass
            
            # Clean up pipeline temporary files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass
                    
        except Exception as e:
            messagebox.showerror("Processing Error", f"Error processing data: {str(e)}")
    
    def populate_provinces():
        """Populate province dropdown with available regions."""
        global province_var, province_menu
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
        """Update station list based on selected province."""
        prov = province_var.get()
        if data['df'] is None:
            return
        
        filtered = data['df'][data['df']['Province'] == prov]
        filtered = filtered.sort_values(by="Station Name")
        
        if station_listbox is not None:
            station_listbox.delete(0, tk.END)
            for _, row in filtered.iterrows():
                station_text = f"{row['Station Name']} [{row['Climate ID']}] - {int(row['DLY First Year'])}-{int(row['DLY Last Year'])}"
                station_listbox.insert(tk.END, station_text)
    
    def show_station_picker():
        """Display the main station picker interface."""
        root.deiconify()
        
        # Title
        tk.Label(root, text="NOAA Climate Data Compiler", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Province selection
        tk.Label(root, text="Select Region/State:").pack()
        global province_menu
        province_menu = tk.OptionMenu(root, province_var, "")
        province_menu.pack()
        
        # Station selection
        tk.Label(root, text="Select Center Station:").pack()
        global station_listbox
        station_listbox = tk.Listbox(root, width=80, height=12)
        station_listbox.pack(pady=5)
        
        # Info label about date range
        info_frame = tk.Frame(root)
        info_frame.pack(pady=5)
        
        tk.Label(info_frame, text="Note: Maximum available date range will be used automatically", 
                font=("Arial", 9), fg="gray").pack()        # Radius selection
        tk.Label(root, text="Select Search Radius:").pack()
        global radius_var
        tk.OptionMenu(root, radius_var, "10 km", "25 km", "50 km", "100 km", "200 km").pack()
        
        # Run button
        tk.Button(root, text="Download NOAA Climate Data", command=process_noaa_data, 
                 bg="lightgreen", font=("Arial", 12, "bold")).pack(pady=15)
        
        # Populate initial data
        populate_provinces()
    
    # Start with station picker
    show_station_picker()
    root.mainloop()

if __name__ == "__main__":
    print("NOAA Data Compiler - Station Picker Format")
    print("Loading...")
    main_gui()
