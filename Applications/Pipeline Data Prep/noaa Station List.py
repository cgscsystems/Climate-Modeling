import requests
import pandas as pd
import time
import re
from datetime import datetime

def get_available_datasets():
    """Get list of availab        # Parse location information from station name
        location, state, country = parse_location_from_name(station_name)
        
        stations.append({
            "station_id": station_id,
            "station_name": station_name,
            "location": location,
            "state": state,
            "country": country,
            "latitude": lat,
            "longitude": lon,
            "file_start_date": start_date,
            "file_end_date": end_date,
            "file_size_bytes": result.get("fileSize", 0),
            "data_types_count": len(data_types),
            "data_types": data_types  # Detailed data type info
        })asets."""
    try:
        response = requests.get("https://www.ncei.noaa.gov/access/services/search/v1/datasets", timeout=30)
        response.raise_for_status()
        datasets = response.json().get("results", [])
        return {d["id"]: d["name"] for d in datasets}
    except:
        return {}

def parse_location_from_name(station_name):
    """
    Parse location name, state, and country from station name.
    
    Format: "STATION NAME, STATE COUNTRY"
    Examples:
    "CARIBOU WEATHER FORECAST OFFICE, ME US" -> ("CARIBOU WEATHER FORECAST OFFICE", "ME", "US")
    "MACON MIDDLE GA REGIONAL AIRPORT, GA US" -> ("MACON MIDDLE GA REGIONAL AIRPORT", "GA", "US") 
    "TORONTO PEARSON INT'L A, ON CA" -> ("TORONTO PEARSON INT'L A", "ON", "CA")
    """
    if not station_name or pd.isna(station_name):
        return None, None, None
    
    # Clean up the name
    name = str(station_name).strip()
    
    # Pattern to match: ", STATE COUNTRY" at the end (STATE and COUNTRY are 2-letter codes)
    location_pattern = r',\s*([A-Z]{2})\s+([A-Z]{2})$'
    match = re.search(location_pattern, name)
    
    if match:
        state = match.group(1)
        country = match.group(2)
        
        # Extract station location name (everything before the comma)
        location_name = name[:match.start()].strip()
        
        return location_name, state, country
    
    # If no match found, return the original name and None for state/country
    return station_name, None, None

def extract_station_metadata(results):
    """Extract detailed station metadata from API results."""
    stations = []
    
    for result in results:
        # Get basic file information
        station_id = result.get("name", "").replace(".csv", "")
        file_path = result.get("filePath", "")
        
        # Extract location info
        bounding_points = result.get("boundingPoints", [])
        lat, lon = None, None
        if bounding_points:
            point = bounding_points[0].get("point", [])
            if len(point) >= 2:
                lon, lat = point[0], point[1]
        
        # Get date range for the file
        start_date = result.get("startDate", "")
        end_date = result.get("endDate", "")
        
        # Extract station details from the stations array
        station_details = result.get("stations", [])
        station_name = ""
        data_types = []
        
        if station_details:
            station_info = station_details[0]
            station_name = station_info.get("name", "")
            station_id = station_info.get("id", station_id)
            
            # Extract available data types with their date ranges
            data_types_info = station_info.get("dataTypes", [])
            for dt in data_types_info:
                data_types.append({
                    "id": dt.get("id", ""),
                    "name": dt.get("name", ""),
                    "start_date": dt.get("startDate", ""),
                    "end_date": dt.get("endDate", ""),
                    "coverage": dt.get("coverage", 0)
                })
        
        stations.append({
            "station_id": station_id,
            "station_name": station_name,
            "latitude": lat,
            "longitude": lon,
            "file_start_date": start_date,
            "file_end_date": end_date,
            "file_size": result.get("fileSize", 0),
            "data_types_count": len(data_types),
            "data_types": data_types  # Detailed data type info
        })
    
    return stations

def get_noaa_station_inventory(dataset="daily-summaries", sample_size=None):
    """
    Retrieve NOAA weather station inventory with detailed metadata.
    
    Args:
        dataset: Dataset to query (daily-summaries, global-hourly, etc.)
        sample_size: If specified, limit results to this number for testing
    """
    base_url = "https://www.ncei.noaa.gov/access/services/search/v1/data"
    
    print(f"Fetching NOAA station inventory for dataset: {dataset}")
    
    # Parameters for getting detailed results
    params = {
        "dataset": dataset,
        "limit": sample_size or 1000,  # Use 1000 per request for full inventory
        "offset": 0
    }
    
    all_stations = []
    max_requests = 200 if sample_size is None else 1  # Allow up to 200 requests for complete inventory
    
    for request_num in range(max_requests):
        try:
            print(f"Request {request_num + 1}: offset {params['offset']}, limit {params['limit']}")
            
            response = requests.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                print("No more results found.")
                break
            
            # Extract station metadata from results
            batch_stations = extract_station_metadata(results)
            all_stations.extend(batch_stations)
            
            print(f"  Extracted {len(batch_stations)} stations. Total so far: {len(all_stations)}")
            
            # Check if we should continue
            if len(results) < params["limit"] or sample_size:
                print("Reached end of available data.")
                break
                
            # Update offset for next request
            params["offset"] += params["limit"]
            
            # Rate limiting - be respectful to the API
            time.sleep(0.5)
            
            # Progress indicator for large requests
            if request_num > 0 and request_num % 10 == 0:
                print(f"  Progress: {request_num + 1} requests completed, {len(all_stations)} stations collected so far...")
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching data: {e}")
            break
        except Exception as e:
            print(f"Unexpected error: {e}")
            break
    
    return all_stations

def create_summary_dataframe(stations):
    """Create a summary DataFrame with key station information."""
    summary_data = []
    
    for station in stations:
        # Calculate date ranges from data types
        earliest_start = None
        latest_end = None
        data_type_list = []
        
        for dt in station.get("data_types", []):
            data_type_list.append(dt["id"])
            
            # Parse dates for range calculation
            try:
                start_date = datetime.fromisoformat(dt["start_date"].replace("Z", "+00:00"))
                if earliest_start is None or start_date < earliest_start:
                    earliest_start = start_date
            except:
                pass
                
            try:
                end_date = datetime.fromisoformat(dt["end_date"].replace("Z", "+00:00"))
                if latest_end is None or end_date > latest_end:
                    latest_end = end_date
            except:
                pass
        
        summary_data.append({
            "station_id": station["station_id"],
            "station_name": station["station_name"],
            "location": station.get("location"),
            "state": station.get("state"),
            "country": station.get("country"),
            "latitude": station["latitude"],
            "longitude": station["longitude"],
            "earliest_data_start": earliest_start.isoformat() if earliest_start else station.get("file_start_date", ""),
            "latest_data_end": latest_end.isoformat() if latest_end else station.get("file_end_date", ""),
            "file_size_bytes": station["file_size"],
            "data_types_count": station["data_types_count"],
            "primary_data_types": ", ".join(data_type_list[:10])  # First 10 data types
        })
    
    return pd.DataFrame(summary_data)



def main():
    """Main function to fetch and save NOAA station inventory with comprehensive metadata."""
    try:
        # Show available datasets
        print("Available NOAA datasets:")
        datasets = get_available_datasets()
        for dataset_id, name in list(datasets.items())[:8]:
            print(f"  - {dataset_id}: {name}")
        
        print("\n" + "="*60)
        
        # Get station inventory for daily summaries (most comprehensive)
        print("Fetching COMPLETE station inventory for daily-summaries dataset...")
        print("WARNING: This will fetch ALL available stations - may take 10-15 minutes!")
        print("Progress will be reported every 10 requests...")
        stations = get_noaa_station_inventory("daily-summaries", sample_size=None)  # Get ALL stations
        
        if not stations:
            print("No stations retrieved. Please check your internet connection and try again.")
            return
        
        print(f"\nSuccessfully retrieved {len(stations)} weather stations with detailed metadata.")
        
        # Create station inventory DataFrame (one row per station)
        station_df = create_summary_dataframe(stations)
        
        # Add retrieval metadata
        station_df["retrieved_date"] = datetime.now().isoformat()
        station_df["api_source"] = "NCEI Search Service API v1"
        station_df["dataset"] = "daily-summaries"
        
        # Sort by station ID
        station_df = station_df.sort_values("station_id")
        
        # Save to CSV file in the Support CSV directory
        import os
        support_dir = "Support CSV"
        os.makedirs(support_dir, exist_ok=True)
        
        inventory_file = os.path.join(support_dir, "noaa_station_inventory.csv")
        station_df.to_csv(inventory_file, index=False)
        
        print(f"\nStation inventory saved to: {inventory_file}")
        
        print(f"\nInventory Statistics:")
        print(f"  Total stations: {len(station_df)}")
        print(f"  Stations with coordinates: {station_df[['latitude', 'longitude']].notna().all(axis=1).sum()}")
        print(f"  US stations: {station_df[station_df['country'] == 'US'].shape[0] if 'US' in station_df['country'].values else 0}")
        print(f"  Canadian stations: {station_df[station_df['country'] == 'CA'].shape[0] if 'CA' in station_df['country'].values else 0}")
        print(f"  Date range: {station_df['earliest_data_start'].min()} to {station_df['latest_data_end'].max()}")
        
        print(f"\nSample stations:")
        print(station_df[['station_id', 'location', 'state', 'country', 'latitude', 'longitude', 'data_types_count']].head(10))
        
        # Show geographic distribution
        if 'state' in station_df.columns:
            print(f"\nTop 10 states/provinces by station count:")
            state_counts = station_df['state'].value_counts().head(10)
            for state, count in state_counts.items():
                print(f"  {state}: {count} stations")
        
        # Optionally check for hourly data availability (sample only)
        print(f"\n" + "="*60)
        print("Checking for hourly data availability (sample only)...")
        
        try:
            hourly_stations = get_noaa_station_inventory("global-hourly", sample_size=100)
            if hourly_stations:
                hourly_summary = create_summary_dataframe(hourly_stations)
                hourly_file = os.path.join(support_dir, "noaa_hourly_stations_sample.csv")
                hourly_summary.to_csv(hourly_file, index=False)
                print(f"  Found {len(hourly_stations)} hourly data stations (sample)")
                print(f"  Saved to: {hourly_file}")
            else:
                print("  No hourly stations found or accessible")
        except Exception as e:
            print(f"  Error checking hourly data: {e}")
        
    except Exception as e:
        print(f"Error in main execution: {e}")

if __name__ == "__main__":
    main()
