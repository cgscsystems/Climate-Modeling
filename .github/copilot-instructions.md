# Climate Modeling AI Assistant Instructions

<!-- Last Updated: 2025-11-04 19:23:19

## Project Overview
This is a **climate data modeling workspace** focused on weather station data ETL, 3D visualization, and ENSO correlation analysis. The project processes Environment Canada (EC) and NOAA weather station data through standardized pipelines and displays temporal patterns using Plotly/Dash.

## Core Architecture

### Data Flow Pipeline
1. **Raw Data Acquisition**: Station-based weather data from EC/NOAA APIs
2. **ETL Processing**: Standardized cleaning, groupwise aggregation, and anomaly calculation
3. **Visualization**: Interactive 3D surfaces showing temporal patterns with ENSO overlays

### Key Components
- **`Applications/Pipeline Data Prep/EC Data Compiler.py`**: Primary GUI-driven ETL pipeline for Environment Canada data
- **`Applications/Pipeline Data Prep/NOAA Data Compiler.py`**: Parallel pipeline for NOAA station data with unified output format
- **`Applications/Climate Data Visualizer.py`**: Main Dash app for 3D visualization and analysis (full-featured)
- **`Applications/Stand-Alone Data Prep/`**: Modular data processing utilities
- **`climate_station_list.csv`**: Master station registry with coordinates and operational periods

## Development Patterns

### Data Processing Conventions
- **Date handling**: Always convert to `YYYY-MM-DD` format, handle leap years with exclusion of `02-29`
- **Column naming**: Use lowercase, snake_case after diacritics removal via `unicodedata.normalize('NFKD')`
- **Chunked processing**: Default 100K row chunks for memory efficiency in `groupwise_aggregation()`
- **Temp file workflow**: Use `tempfile.NamedTemporaryFile(delete=False)` for intermediate CSV storage
- **Encoding fallback**: Always try `utf-8` → `latin1` → `iso-8859-1` for robust file reading

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
- **Station selection**: Haversine distance calculations for radius-based station filtering

### Dash Application Structure
- **Layout pattern**: Sidebar controls + main graph area with `flex` display
- **Callback architecture**: Use `dcc.Store` for data persistence across callbacks
- **File upload**: Base64 decoding with encoding fallback for CSV parsing
- **Data transformation**: Melt dataframes for time-series plotting (`id_vars=['date', 'year', 'md']`)

### Visualization Standards
- **3D surface plots**: Use Plotly `go.Surface` with customizable color palettes (default: 'Turbo')
- **ENSO integration**: Overlay temporal climate phases as semi-transparent 3D bands
- **Anomaly calculation**: Always provide "raw" vs "anomaly from average" display modes
- **Aspect ratios**: Default to `x=2, y=4, z=1` for optimal temporal visualization
- **Dark mode**: Full theming support with dynamic style callbacks

## Critical File Locations
- **Station data**: `climate_station_list.csv` (master registry)
- **ENSO phases**: `Support CSV/enso monthly phases 2025-08-12.csv`
- **Province mapping**: `PROVINCE_CODE_MAP` in Applications/Pipeline Data Prep/EC Data Compiler (13 provinces/territories)
- **Output directory**: Always prompt user for save location with `filedialog.asksaveasfilename()`
- **NOAA station format**: `Applications/Support CSV/noaa_stations_ec_format.csv`

## Dependencies & Environment
- **Core libraries**: `pandas`, `plotly`, `dash`, `requests`, `beautifulsoup4`, `tkinter`
- **Scientific**: `numpy`, `scipy` (for gaussian filtering)
- **Development**: No formal requirements.txt - install as needed
- **Launch pattern**: Dash apps auto-open browser via `webbrowser.open_new("http://127.0.0.1:8050/")`

## Data Source URLs
- **Environment Canada**: `https://dd.weather.gc.ca/today/climate/observations/daily/csv/{province_code}/` (Updated Nov 2025)
- **NOAA**: `https://www.ncei.noaa.gov/data/daily-summaries/access/`

**Note**: Environment Canada changed their URL structure in 2025. Old URLs under `/climate/observations/` no longer work.

## API Health Monitoring
**CRITICAL**: External data APIs can change without notice, breaking pipelines. Proactively monitor for changes:

### Routine Health Checks
When users report pipeline failures or during regular maintenance, **always test core API endpoints**:

```bash
# Environment Canada - Test base URL and file structure
curl -I "https://dd.weather.gc.ca/today/climate/observations/daily/csv/AB/"
curl -s "https://dd.weather.gc.ca/today/climate/observations/daily/csv/AB/" | grep -o "climate_daily_AB_[^\"]*\.csv" | head -5

# NOAA - Test base URL and sample station
curl -I "https://www.ncei.noaa.gov/data/daily-summaries/access/"
curl -I "https://www.ncei.noaa.gov/data/daily-summaries/access/USC00010252.csv"

# Environment Canada Station List
curl -I "https://dd.weather.gc.ca/today/climate/observations/climate_station_list.csv"
```

### Breaking Change Indicators
Watch for these warning signs:
- **404 errors** on previously working URLs
- **Changed file naming patterns** (EC: `climate_daily_{province}_{station}_{year}_P1D.csv`)
- **New directory structures** in HTML directory listings
- **Missing or moved station registry files**
- **Changed response headers** or content types

### Proactive Monitoring Strategy
1. **Monthly API checks**: Test all base URLs and sample files
2. **Pipeline testing**: Run EC/NOAA compilers with test stations monthly
3. **Documentation review**: Check official MSC/NOAA documentation for announcements
4. **User feedback**: Pipeline failures often indicate upstream changes

### Emergency Response Protocol
When API breaks are detected:
1. **Immediate diagnosis**: Use `curl` and `fetch_webpage` tools to investigate
2. **Documentation research**: Check official NOAA/MSC websites for migration notices
3. **URL pattern analysis**: Compare old vs new structures to identify changes
4. **Code updates**: Update base URLs, file patterns, and parsing logic
5. **Testing validation**: Verify fixes with multiple provinces/stations
6. **Documentation updates**: Update this file with new URL patterns and breaking change notes

### Historical Breaking Changes
- **Nov 2025**: Environment Canada moved from `/climate/observations/` to `/today/climate/observations/`
- **Nov 2025**: EC filename format changed to include `_P1D.csv` suffix
- **[Future changes will be documented here]**

## Development Roadmap

### Active Development: Fork 1 - ENSO Indices Integration
**Goal**: Enhance ENSO analysis with live indices from NOAA Climate Prediction Center
- **Data Source**: `https://www.cpc.ncep.noaa.gov/data/indices/`
- **Target Indices**: ONI (Oceanic Niño Index), SOI (Southern Oscillation Index), MEI (Multivariate ENSO Index)
- **Integration**: Replace/supplement static ENSO phases with dynamic, continuously updated indices
- **Visualization**: Enhanced 3D climate surface overlays with real-time ENSO correlation analysis
- **Pipeline**: New `Applications/Pipeline Data Prep/ENSO Indices Compiler.py`

### Planned Development: Fork 2 - Global Sea Surface Temperature Mapping
**Goal**: Animated time-series "movies" of global ocean temperature changes
- **Data Source**: `https://www.ncei.noaa.gov/data/sea-surface-temperature-extended-reconstructed/`
- **Format**: NetCDF/CSV grids for global coverage
- **Visualization Strategy**: 
  - Animated world map overlays showing temperature changes over time
  - Large-scale regional analysis over maximum available periods
  - Static image generation and knitting for movie creation
  - Raw data animation on interactive world maps
- **Technical Approach**: Geospatial data handling, map projection, temporal animation
- **Applications**: Global ocean-atmosphere interaction analysis, climate baseline establishment

## Common Workflows

### Running the Application
```python
# Main visualization app (full-featured with ENSO, dark mode, outlier management)
python "Applications/Climate Data Visualizer.py"
```

### Data Processing Workflows
1. **EC Data**: Run `Applications/Pipeline Data Prep/EC Data Compiler.py` → Choose "Station Picker" → Select province/station/radius
2. **NOAA Data**: Run `Applications/Pipeline Data Prep/NOAA Data Compiler.py` 
3. **Format Conversion**: Use `Applications/Pipeline Data Prep/convert_noaa_to_ec_format.py`
4. **Stand-alone processing**: Individual utilities in `Applications/Stand-Alone Data Prep/`

### Adding New Data Sources
1. Implement `scrape_[source]_data_set()` function following existing patterns
2. Add source-specific column mapping to `KEEP_COLS_CLEAN`
3. Update GUI with source selection option
4. Ensure output matches standardized format for visualization compatibility

### Debugging Data Pipeline Issues
- Check encoding with fallback: `utf-8` → `latin1` → `iso-8859-1`
- Verify station ID format matches URL patterns (`_{station_id}_` for EC, station ID directly for NOAA)
- **NEW**: EC file format changed Nov 2025: `climate_daily_{province}_{station_id}_{year}_P1D.csv`
- Validate date parsing with `pd.to_datetime(errors='coerce')` then `dropna()`
- **API Health**: If pipelines fail unexpectedly, immediately test base URLs with `curl` commands (see API Health Monitoring section)

### Routine Maintenance Tasks
- **Monthly**: Test Environment Canada and NOAA API endpoints for availability and structure changes
- **Quarterly**: Update station list from Environment Canada using "Update from Web" button
- **As needed**: Monitor official MSC and NOAA websites for API deprecation announcements
- **After failures**: Always investigate potential upstream API changes before assuming code issues

### Performance Optimization
- Use `chunksize=100000` for large datasets
- Implement `stream=True` for file downloads
- Clean up temp files in `finally` blocks
- Cache station registry data in GUI applications

## Testing Approaches
- **Manual validation**: Use `climate_station_list.csv` to verify station coordinates and operational periods
- **Data integrity**: Check for `NaN` inflation after ETL pipeline completion
- **Visualization testing**: Load sample datasets through complete pipeline to verify 3D surface rendering

## Auto-Maintenance System
This project includes automated documentation maintenance via git pre-commit hooks:

### What Gets Auto-Updated
- **This file**: Timestamp and last modified date on every commit
- **`Weather-Modeling.code-workspace`**: Metadata about recent file changes
- **`.ai-conversations/conversation-summary.md`**: Project status and recent changes context

### Setup
```powershell
# Run once to enable (already configured)
.\setup-auto-maintenance.ps1
```

### Manual Override
To commit without auto-maintenance (if needed):
```bash
git commit --no-verify -m "your message"
```

## Visualizations Directory
- **Current samples**: 4 visualization files (MP4/BMP) are tracked as examples
- **Future files**: New files in `Visualizations/` are automatically ignored
- **Manual inclusion**: Use `git add -f Visualizations/filename` to explicitly track new samples