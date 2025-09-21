"""
Convert NOAA Station Inventory to Environment Canada Format
This script transforms the NOAA station inventory CSV to match the exact column 
structure expected by the EC Data Compiler, enabling seamless integration.
"""

import pandas as pd
import os
from datetime import datetime

def extract_year_from_date(date_string):
    """Extract year from ISO date string, handling various formats."""
    if pd.isna(date_string) or date_string == "":
        return None
    
    try:
        # Handle ISO format dates
        if "T" in str(date_string):
            return datetime.fromisoformat(date_string.replace("Z", "+00:00")).year
        elif "-" in str(date_string):
            return int(str(date_string).split("-")[0])
        else:
            return int(str(date_string)[:4])
    except:
        return None

def map_country_to_province(country, state):
    """Map country/state combinations to province-like regions for grouping."""
    if pd.isna(country) or pd.isna(state):
        return "UNKNOWN"
    
    country = str(country).upper()
    state = str(state).upper()
    
    # Canadian provinces (keep as is)
    if country == "CA":
        province_map = {
            "AB": "ALBERTA",
            "BC": "BRITISH COLUMBIA", 
            "MB": "MANITOBA",
            "NB": "NEW BRUNSWICK",
            "NL": "NEWFOUNDLAND",
            "NS": "NOVA SCOTIA",
            "ON": "ONTARIO",
            "PE": "PRINCE EDWARD ISLAND",
            "QC": "QUEBEC",
            "SK": "SASKATCHEWAN",
            "NT": "NORTHWEST TERRITORIES",
            "NU": "NUNAVUT",
            "YT": "YUKON"
        }
        return province_map.get(state, f"CANADA-{state}")
    
    # US states (group by region for better organization)
    elif country == "US":
        # Use state name directly for US states
        return f"US-{state}"
    
    # Other countries
    else:
        return f"{country}-{state}"

def convert_noaa_to_ec_format(input_file, output_file=None):
    """
    Convert NOAA station inventory to EC Data Compiler format.
    
    Args:
        input_file: Path to NOAA station inventory CSV
        output_file: Path for output CSV (optional, defaults to same directory)
    """
    
    if not os.path.exists(input_file):
        print(f" Input file not found: {input_file}")
        return False
    
    print(f" Loading NOAA station inventory from: {input_file}")
    
    try:
        # Load NOAA station data
        noaa_df = pd.read_csv(input_file)
        print(f"   Loaded {len(noaa_df)} stations")
        
        # Display original columns for verification
        print(f"   Original columns: {list(noaa_df.columns)}")
        
        # Create EC-compatible DataFrame
        ec_df = pd.DataFrame()
        
        # Map columns to EC format
        print(" Converting to EC Data Compiler format...")
        
        # Climate ID -> station_id (this is the key identifier)
        ec_df['Climate ID'] = noaa_df['station_id']
        
        # Station Name -> station_name or location (whichever has better data)
        if 'location' in noaa_df.columns and noaa_df['location'].notna().sum() > 0:
            ec_df['Station Name'] = noaa_df['location'].fillna(noaa_df['station_name'])
        else:
            ec_df['Station Name'] = noaa_df['station_name']
        
        # Province -> mapped from country + state (handle both lowercase and capitalized column names)
        country_col = 'Country' if 'Country' in noaa_df.columns else 'country'
        state_col = 'State' if 'State' in noaa_df.columns else 'state'
        
        ec_df['Province'] = noaa_df.apply(
            lambda row: map_country_to_province(row.get(country_col), row.get(state_col)), 
            axis=1
        )
        
        # Coordinates (direct mapping)
        ec_df['Latitude'] = pd.to_numeric(noaa_df['latitude'], errors='coerce')
        ec_df['Longitude'] = pd.to_numeric(noaa_df['longitude'], errors='coerce')
        
        # Extract years from date ranges
        ec_df['DLY First Year'] = noaa_df['earliest_data_start'].apply(extract_year_from_date)
        ec_df['DLY Last Year'] = noaa_df['latest_data_end'].apply(extract_year_from_date)
        
        # Clean up missing values
        ec_df['DLY First Year'] = ec_df['DLY First Year'].fillna(1900).astype('Int64')
        ec_df['DLY Last Year'] = ec_df['DLY Last Year'].fillna(2025).astype('Int64')
        
        # Remove stations without coordinates (required for distance calculations)
        original_count = len(ec_df)
        ec_df = ec_df.dropna(subset=['Latitude', 'Longitude'])
        coordinate_count = len(ec_df)
        
        if coordinate_count < original_count:
            print(f"    Removed {original_count - coordinate_count} stations without coordinates")
        
        # Sort by Province then Station Name for better organization
        ec_df = ec_df.sort_values(['Province', 'Station Name'])
        ec_df = ec_df.reset_index(drop=True)
        
        # Determine output file path
        if output_file is None:
            input_dir = os.path.dirname(input_file)
            output_file = os.path.join(input_dir, "noaa_stations_ec_format.csv")
        
        # Save converted file
        ec_df.to_csv(output_file, index=False)
        
        print(f" Conversion completed successfully!")
        print(f"   Output file: {output_file}")
        print(f"   Final station count: {len(ec_df)}")
        
        # Display statistics
        print(f"\n Conversion Statistics:")
        print(f"   Provinces/Regions: {ec_df['Province'].nunique()}")
        print(f"   Date range: {ec_df['DLY First Year'].min()} - {ec_df['DLY Last Year'].max()}")
        print(f"   Stations with valid coordinates: {coordinate_count}")
        
        # Show top provinces by station count
        print(f"\n Top 10 Provinces/Regions by Station Count:")
        province_counts = ec_df['Province'].value_counts().head(10)
        for province, count in province_counts.items():
            print(f"     {province}: {count} stations")
        
        # Display sample of converted data
        print(f"\n Sample of converted data:")
        sample_cols = ['Climate ID', 'Station Name', 'Province', 'Latitude', 'Longitude', 'DLY First Year', 'DLY Last Year']
        print(ec_df[sample_cols].head(10).to_string(index=False))
        
        return True
        
    except Exception as e:
        print(f" Conversion failed: {e}")
        return False

def main():
    """Main function to convert NOAA station inventory."""
    
    # Default paths
    support_dir = "Support CSV"
    input_file = os.path.join(support_dir, "noaa_station_inventory.csv")
    output_file = os.path.join(support_dir, "noaa_stations_ec_format.csv")
    
    print(" NOAA to EC Format Conversion Tool")
    print("="*50)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f" NOAA station inventory not found at: {input_file}")
        print("   Please run 'noaa Station List.py' first to generate the inventory.")
        return
    
    # Perform conversion
    success = convert_noaa_to_ec_format(input_file, output_file)
    
    if success:
        print(f"\n Conversion completed!")
        print(f"   Your NOAA stations are now in EC Data Compiler format.")
        print(f"   You can load '{output_file}' in the EC Data Compiler station picker.")
        print(f"\n Next steps:")
        print(f"   1. Open EC Data Compiler")
        print(f"   2. Choose 'Use Station Picker'")
        print(f"   3. Load the converted file: {output_file}")
        print(f"   4. Select a NOAA station and radius")
        print(f"   5. The existing GUI will work seamlessly!")
    else:
        print(f"\n Conversion failed. Please check the error messages above.")

if __name__ == "__main__":
    main()
