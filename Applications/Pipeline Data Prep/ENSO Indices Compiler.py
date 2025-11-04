# ENSO Indices Compiler - Download and standardize ENSO indices from NOAA CPC
# Dependencies: pandas, requests, tkinter

import pandas as pd
import requests
import os
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox
from datetime import datetime
import sys

# ENSO Indices Data Sources
ENSO_INDICES = {
    'ONI': {
        'url': 'https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt',
        'name': 'Oceanic Niño Index',
        'description': '3-month running average of Niño 3.4 SST anomalies'
    },
    'SOI': {
        'url': 'https://www.cpc.ncep.noaa.gov/data/indices/soi.3m.txt',
        'name': 'Southern Oscillation Index',
        'description': '3-month running mean of standardized SLP differences'
    },
    'NINO34': {
        'url': 'https://www.cpc.ncep.noaa.gov/data/indices/3mth.nino34.91-20.ascii.txt',
        'name': 'Niño 3.4 Index',
        'description': '3-month running average of Niño 3.4 SST anomalies (1991-2020 base)'
    },
    'RONI': {
        'url': 'https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt',
        'name': 'Relative Oceanic Niño Index',
        'description': 'Relative 3-month running average of Niño 3.4 SST anomalies'
    }
}

def download_enso_index(index_key, url, description):
    """Download and parse ENSO index data."""
    try:
        print(f"📥 Downloading {description}...")
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Parse the data - format is typically: SEAS YR TOTAL ANOM
        lines = response.text.strip().split('\n')
        data = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith(' SEAS') or line.startswith('SEAS'):
                continue  # Skip header and empty lines
                
            parts = line.split()
            if len(parts) >= 4:
                season = parts[0]
                year = int(parts[1])
                total = float(parts[2])
                anomaly = float(parts[3])
                
                # Convert season to approximate date
                season_map = {
                    'DJF': f"{year}-01-15",  # Dec-Jan-Feb
                    'JFM': f"{year}-02-15",  # Jan-Feb-Mar
                    'FMA': f"{year}-03-15",  # Feb-Mar-Apr
                    'MAM': f"{year}-04-15",  # Mar-Apr-May
                    'AMJ': f"{year}-05-15",  # Apr-May-Jun
                    'MJJ': f"{year}-06-15",  # May-Jun-Jul
                    'JJA': f"{year}-07-15",  # Jun-Jul-Aug
                    'JAS': f"{year}-08-15",  # Jul-Aug-Sep
                    'ASO': f"{year}-09-15",  # Aug-Sep-Oct
                    'SON': f"{year}-10-15",  # Sep-Oct-Nov
                    'OND': f"{year}-11-15",  # Oct-Nov-Dec
                    'NDJ': f"{year+1}-12-15" if season == 'NDJ' else f"{year}-12-15"  # Nov-Dec-Jan
                }
                
                date = season_map.get(season, f"{year}-06-15")  # Default to mid-year
                
                data.append({
                    'date': date,
                    'season': season,
                    'year': year,
                    f'{index_key.lower()}_total': total,
                    f'{index_key.lower()}_anomaly': anomaly
                })
        
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        print(f"✅ Downloaded {len(df)} records for {description}")
        return df
        
    except Exception as e:
        print(f"❌ Failed to download {description}: {e}")
        return None

def merge_enso_indices(indices_data):
    """Merge all ENSO indices into a single DataFrame."""
    if not indices_data:
        return None
    
    # Start with the first index
    merged_df = list(indices_data.values())[0].copy()
    
    # Merge other indices
    for index_name, df in list(indices_data.items())[1:]:
        # Merge on date and season for precise alignment
        merge_cols = ['date', 'season', 'year']
        value_cols = [col for col in df.columns if col not in merge_cols]
        
        merged_df = merged_df.merge(
            df[merge_cols + value_cols], 
            on=merge_cols, 
            how='outer'
        )
    
    # Sort by date and clean up
    merged_df = merged_df.sort_values('date').reset_index(drop=True)
    
    # Add metadata columns
    merged_df['data_source'] = 'NOAA Climate Prediction Center'
    merged_df['download_date'] = datetime.now().strftime('%Y-%m-%d')
    
    return merged_df

def categorize_enso_phases(df):
    """Add ENSO phase categorization based on ONI values."""
    if 'oni_anomaly' not in df.columns:
        print("⚠️ ONI data not available for phase categorization")
        return df
    
    # Simple approach using numpy.where for phase assignment
    import numpy as np
    
    df['enso_phase'] = np.where(
        df['oni_anomaly'] >= 0.5, 'El Niño',
        np.where(df['oni_anomaly'] <= -0.5, 'La Niña', 'Neutral')
    )
    
    return df

def main_gui():
    """Main GUI for ENSO Indices Compiler."""
    root = tk.Tk()
    root.title("ENSO Indices Compiler")
    root.geometry("600x500")
    
    # Header
    tk.Label(root, text="ENSO Indices Compiler", 
             font=("Arial", 16, "bold")).pack(pady=10)
    tk.Label(root, text="Download live ENSO indices from NOAA Climate Prediction Center", 
             font=("Arial", 10)).pack(pady=5)
    
    # Available indices
    tk.Label(root, text="Available ENSO Indices:", 
             font=("Arial", 12, "bold")).pack(pady=(20, 10))
    
    # Checkboxes for each index
    index_vars = {}
    for key, info in ENSO_INDICES.items():
        frame = tk.Frame(root)
        frame.pack(anchor='w', padx=50, pady=2)
        
        var = tk.BooleanVar(value=True)  # Default all selected
        index_vars[key] = var
        
        cb = tk.Checkbutton(frame, variable=var, 
                           text=f"{info['name']} ({key})",
                           font=("Arial", 10, "bold"))
        cb.pack(side='left')
        
        tk.Label(frame, text=f"- {info['description']}", 
                font=("Arial", 9), fg="gray").pack(side='left', padx=(10, 0))
    
    # Options
    tk.Label(root, text="Options:", 
             font=("Arial", 12, "bold")).pack(pady=(20, 10))
    
    phase_var = tk.BooleanVar(value=True)
    tk.Checkbutton(root, variable=phase_var, 
                   text="Add ENSO phase categorization (El Niño/La Niña/Neutral)",
                   font=("Arial", 10)).pack(anchor='w', padx=50)
    
    def download_indices():
        """Download selected indices and save to file."""
        try:
            # Get selected indices
            selected = {key: info for key, info in ENSO_INDICES.items() 
                       if index_vars[key].get()}
            
            if not selected:
                messagebox.showerror("Error", "Please select at least one ENSO index.")
                return
            
            # Download data
            indices_data = {}
            for key, info in selected.items():
                df = download_enso_index(key, info['url'], info['name'])
                if df is not None:
                    indices_data[key] = df
            
            if not indices_data:
                messagebox.showerror("Error", "Failed to download any ENSO indices.")
                return
            
            # Merge indices
            print("🔄 Merging ENSO indices...")
            merged_df = merge_enso_indices(indices_data)
            
            if merged_df is None:
                messagebox.showerror("Error", "Failed to merge ENSO indices.")
                return
            
            # Add phase categorization if requested
            if phase_var.get():
                print("🔄 Adding ENSO phase categorization...")
                merged_df = categorize_enso_phases(merged_df)
            
            # Prompt for save location
            save_path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Save ENSO Indices Data"
            )
            
            if not save_path:
                # Suggest default filename if user didn't specify
                default_name = f"enso_indices_{datetime.now().strftime('%Y%m%d')}.csv"
                print(f"❌ Save cancelled. Suggested filename: {default_name}")
                return
            
            # Save to CSV
            merged_df.to_csv(save_path, index=False)
            
            # Summary
            date_range = f"{merged_df['date'].min().strftime('%Y-%m')} to {merged_df['date'].max().strftime('%Y-%m')}"
            summary = f"""✅ ENSO indices successfully downloaded and saved!

📄 File: {save_path}
📊 Records: {len(merged_df)}
📅 Date Range: {date_range}
📋 Indices: {', '.join(selected.keys())}
🌊 Phase Data: {'Yes' if phase_var.get() else 'No'}

Data Source: NOAA Climate Prediction Center
Download Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"""
            
            messagebox.showinfo("Download Complete", summary)
            print(summary)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to download ENSO indices: {e}")
            print(f"❌ Error: {e}")
    
    # Download button
    tk.Button(root, text="Download ENSO Indices", 
              command=download_indices,
              bg="lightblue", font=("Arial", 12, "bold"),
              pady=10).pack(pady=30)
    
    # Info footer
    info_text = """Data automatically updated by NOAA CPC
Perfect for climate correlation analysis and ENSO monitoring"""
    
    tk.Label(root, text=info_text, 
             font=("Arial", 9), fg="gray", justify='center').pack(side='bottom', pady=10)
    
    root.mainloop()

if __name__ == "__main__":
    print("ENSO Indices Compiler - NOAA Climate Prediction Center Data")
    print("=" * 60)
    main_gui()