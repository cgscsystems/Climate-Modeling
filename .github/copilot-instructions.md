# Climate Modeling AI Assistant Instructions

## Project Overview
This is a **climate data modeling workspace** focused on weather station data ETL, 3D visualization, and ENSO correlation analysis. The project processes Environment Canada (EC) and NOAA weather station data through standardized pipelines and displays temporal patterns using Plotly/Dash.

## Core Architecture

### Data Flow Pipeline
1. **Raw Data Acquisition**: Station-based weather data from EC/NOAA APIs
2. **ETL Processing**: Standardized cleaning, groupwise aggregation, and anomaly calculation
3. **Visualization**: Interactive 3D surfaces showing temporal patterns with ENSO overlays

### Key Components
- **`EC Data Compiler.py`**: Primary GUI-driven ETL pipeline for Environment Canada data
- **`NOAA Data Compiler.py`**: Parallel pipeline for NOAA station data with unified output format
- **`Weather Model Application.py`**: Main Dash app for 3D visualization and analysis
- **`Applications/Stand-Alone Data Prep/`**: Modular data processing utilities
- **`climate_station_list.csv`**: Master station registry with coordinates and operational periods

## Development Patterns

### Data Processing Conventions
- **Date handling**: Always convert to `YYYY-MM-DD` format, handle leap years with exclusion of `02-29`
- **Column naming**: Use lowercase, snake_case after diacritics removal via `unicodedata.normalize('NFKD')`
- **Chunked processing**: Default 100K row chunks for memory efficiency in `groupwise_aggregation()`
- **Temp file workflow**: Use `tempfile.NamedTemporaryFile(delete=False)` for intermediate CSV storage

### Standard ETL Pipeline Order
```python
# Always follow this sequence:
clean_data() → preprocess_columns() → drop_problem_columns() → 
groupwise_aggregation() → calculate_columns()
```

### GUI Framework Pattern
- **Tkinter root management**: Always `root.withdraw()` initially, then `root.deiconify()` when needed
- **File dialogs**: Use `filedialog.askopenfilename()` with specific filetypes
- **Error handling**: Wrap processing in try/except with `messagebox.showerror()`

### Visualization Standards
- **3D surface plots**: Use Plotly `go.Surface` with customizable color palettes (default: 'Turbo')
- **ENSO integration**: Overlay temporal climate phases as semi-transparent 3D bands
- **Anomaly calculation**: Always provide "raw" vs "anomaly from average" display modes
- **Aspect ratios**: Default to `x=2, y=4, z=1` for optimal temporal visualization

## Critical File Locations
- **Station data**: `climate_station_list.csv` (master registry)
- **ENSO phases**: `Support CSV/enso monthly phases 2025-08-12.csv`
- **Province mapping**: `PROVINCE_CODE_MAP` in EC Data Compiler
- **Output directory**: Always prompt user for save location with `filedialog.asksaveasfilename()`

## Data Source URLs
- **Environment Canada**: `https://dd.weather.gc.ca/climate/observations/daily/csv/{province_code}/`
- **NOAA**: `https://www.ncei.noaa.gov/data/daily-summaries/access/`

## Common Workflows

### Adding New Data Sources
1. Implement `scrape_[source]_data_set()` function following existing patterns
2. Add source-specific column mapping to `KEEP_COLS_CLEAN`
3. Update GUI with source selection option
4. Ensure output matches standardized format for visualization compatibility

### Debugging Data Pipeline Issues
- Check encoding with fallback: `utf-8` → `latin1` → `iso-8859-1`
- Verify station ID format matches URL patterns (`_{station_id}_` for EC, station ID directly for NOAA)
- Validate date parsing with `pd.to_datetime(errors='coerce')` then `dropna()`

### Performance Optimization
- Use `chunksize=100000` for large datasets
- Implement `stream=True` for file downloads
- Clean up temp files in `finally` blocks
- Cache station registry data in GUI applications

## Testing Approaches
- **Manual validation**: Use `climate_station_list.csv` to verify station coordinates and operational periods
- **Data integrity**: Check for `NaN` inflation after ETL pipeline completion
- **Visualization testing**: Load sample datasets through complete pipeline to verify 3D surface rendering