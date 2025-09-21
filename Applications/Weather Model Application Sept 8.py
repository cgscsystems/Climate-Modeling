import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx
from scipy.ndimage import gaussian_filter
import base64
import io
import json
import requests
import re
from datetime import datetime

# --- ENSO: static file + styling ---
# Point this to your prebuilt CSV (date, oni, intensity). Date must be YYYY-MM-01.
ENSO_CSV_PATH = r"Support CSV\enso monthly phases 2025-08-12.csv"

# Intensity -> base opacity (before global slider multiplier)
ENSO_OPACITY_LEVEL = {
    "Neutral": 0.00,
    "Weak":    0.20,
    "Medium":  0.45,
    "Strong":  0.75,
}

# Sign color: warm (El Niño, oni>=0) vs cold (La Niña, oni<0)
ENSO_SIGN_COLOR = {
    "warm": "rgba(255, 80, 80, 1.0)",   # red-ish
    "cold": "rgba( 80, 80,255, 1.0)",   # blue-ish
}

def load_static_enso(path=ENSO_CSV_PATH):
    df = pd.read_csv(path)
    # Normalize column names and types
    df.columns = [c.strip().lower() for c in df.columns]
    # expected columns: date, oni, intensity
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df["oni"] = pd.to_numeric(df["oni"], errors="coerce")
    df["intensity"] = df["intensity"].astype(str).str.strip().str.title()
    # derive year, month, sign
    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["sign"] = np.where(df["oni"] >= 0, "warm", "cold")
    # anything non-matching -> Neutral
    df.loc[~df["intensity"].isin(ENSO_OPACITY_LEVEL.keys()), "intensity"] = "Neutral"
    return df

ENSO_DF = None
try:
    ENSO_DF = load_static_enso()
except Exception as _e:
    # We'll proceed without overlay if file is missing; message printed in console
    print(f"[ENSO] Could not load static ENSO file: {_e}")


# ---- Dash App ----
app = Dash(__name__)

# Add external stylesheets for better dropdown styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Dark mode dropdown styling */
            .dark-mode .Select-control {
                background-color: #3c3c3c !important;
                border: 1px solid #555 !important;
                color: #ffffff !important;
            }
            .dark-mode .Select-input > input {
                color: #ffffff !important;
            }
            .dark-mode .Select-placeholder {
                color: #cccccc !important;
            }
            .dark-mode .Select-value-label {
                color: #ffffff !important;
            }
            .dark-mode .Select-menu-outer {
                background-color: #3c3c3c !important;
                border: 1px solid #555 !important;
            }
            .dark-mode .Select-option {
                background-color: #3c3c3c !important;
                color: #ffffff !important;
            }
            .dark-mode .Select-option:hover {
                background-color: #4a4a4a !important;
                color: #ffffff !important;
            }
            .dark-mode .Select-option.is-selected {
                background-color: #5a5a5a !important;
                color: #ffffff !important;
            }
            .dark-mode .Select-option.is-focused {
                background-color: #4a4a4a !important;
                color: #ffffff !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    html.Div(id='main-container', children=[
        html.H1("Climate Visualization Tool", style={'margin': '0 0 10px 0', 'fontSize': 24}),
        dcc.Store(id='plot-data-store'),
        html.Div([
            html.Div(id='sidebar-container', children=[
                dcc.Upload(
                    id='upload-data',
                    children=html.Button('Upload Weather Data (CSV/JSON)', id='upload-button', style={'width': '100%', 'fontSize': 12, 'padding': '4px'}),
                    multiple=False,
                    style={'marginBottom': '8px'}
                ),
                html.Div(id='upload-error', style={'color': 'red', 'fontSize': 12, 'marginBottom': '8px'}),
                html.Div(id='controls-container'),
            ], style={
                'width': '220px',
                'minWidth': '180px',
                'maxWidth': '260px',
                'padding': '8px',
                'background': '#f8f8f8',
                'borderRight': '1px solid #ddd',
                'height': '85vh',
                'overflowY': 'auto',
                'display': 'inline-block',
                'verticalAlign': 'top'
            }),
            html.Div([
                dcc.Graph(id='value-3d-graph', style={'height': '85vh', 'width': '100%'})
            ], style={'marginLeft': '10px', 'display': 'inline-block', 'width': 'calc(100% - 240px)', 'verticalAlign': 'top'})
        ], style={'display': 'flex', 'flexDirection': 'row'})
    ])
], style={'height': '100vh', 'margin': 0, 'padding': 0, 'fontFamily': 'Segoe UI, Arial, sans-serif'})

# ---- Helper: Parse JSON Time Series ----
def parse_json_data(json_data):
    """
    Parse JSON time series data in the format:
    [{"name": "year", "data": [daily_values...]}, ...]
    Convert to the same format as CSV data.
    """
    all_data = []
    
    for series in json_data:
        series_name = series['name']
        daily_values = series['data']
        
        # Skip statistical series (keep only year data)
        if not series_name.isdigit():
            continue
            
        year = int(series_name)
        
        # Create dates for the year (assuming Jan 1 start, 366 days to handle leap years)
        dates = pd.date_range(start=f'{year}-01-01', periods=len(daily_values), freq='D')
        
        for i, (date, value) in enumerate(zip(dates, daily_values)):
            if value is not None:  # Skip null values
                # Create the variable name (you can customize this)
                variable_name = 'sea_surface_temp'  # Default name, could be extracted from filename
                
                # Adjust year for December dates (matching CSV logic)
                adjusted_year = date.year + 1 if date.month == 12 else date.year
                
                all_data.append({
                    'variable': variable_name,
                    'year': adjusted_year,
                    'md': date.strftime('%m-%d'),
                    'date': date,
                    'value': float(value)
                })
    
    return pd.DataFrame(all_data)

# ---- Helper: Parse Uploaded CSV or JSON ----
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    try:
        # First, decode the content
        try:
            content_str = decoded.decode('utf-8')
        except UnicodeDecodeError:
            content_str = decoded.decode('latin1')
        
        # Try to parse as JSON first
        try:
            json_data = json.loads(content_str)
            # Check if it's our expected JSON format (array of objects with 'name' and 'data')
            if (isinstance(json_data, list) and 
                len(json_data) > 0 and 
                isinstance(json_data[0], dict) and 
                'name' in json_data[0] and 
                'data' in json_data[0]):
                
                plot_data = parse_json_data(json_data)
                if not plot_data.empty:
                    return plot_data, None
                else:
                    return None, "No valid numeric year data found in JSON file"
            else:
                # Not our expected JSON format, fall through to CSV parsing
                pass
        except json.JSONDecodeError:
            # Not valid JSON, try CSV parsing
            pass
        
        # Parse as CSV
        df = pd.read_csv(io.StringIO(content_str))
        df.columns = [col.strip().lower() for col in df.columns]
        df.rename(columns={df.columns[0]: "date"}, inplace=True)
        df['date'] = pd.to_datetime(df['date'].astype(str).str.strip(), format='%Y-%m-%d', errors='coerce')
        df = df.dropna(axis=1, how='all')
        df['year'] = df['date'].apply(lambda d: d.year + 1 if d.month == 12 else d.year).astype('Int64')
        df = df[df['year'].notna()]
        df['md'] = df['date'].dt.strftime('%m-%d')
        df_melted = df.melt(id_vars=['date', 'year', 'md'], var_name='variable', value_name='value')
        df_melted.sort_values(by=['variable', 'date'], inplace=True)
        plot_data = df_melted.dropna(subset=['value'])[['variable', 'year', 'md', 'date', 'value']]
        return plot_data, None
        
    except Exception as e:
        return None, f"Error parsing file (tried both JSON and CSV formats): {str(e)}"

@app.callback(
    Output('controls-container', 'children'),
    Output('upload-error', 'children'),
    Output('plot-data-store', 'data'),
    Input('upload-data', 'contents')
)
def render_controls(contents):
    if not contents:
        return None, "", None
    plot_data, error = parse_contents(contents)
    if error:
        return None, f"Error parsing file: {error}", None
    if plot_data is None or plot_data.empty:
        return None, "No valid data found in file.", None
    return html.Div([
        html.Label("Dark Mode:"),
        dcc.RadioItems(
            id='dark-mode-toggle',
            options=[
                {'label': 'Light', 'value': 'light'},
                {'label': 'Dark', 'value': 'dark'}
            ],
            value='light',
            labelStyle={'display': 'inline-block', 'marginRight': '10px'},
            style={'marginBottom': '8px'}
        ),

        html.Label("Select Variable:"),
        dcc.Dropdown(
            id='variable-dropdown',
            options=[{'label': var, 'value': var} for var in sorted(plot_data['variable'].unique())],
            value=plot_data['variable'].unique()[0],
            style={'marginBottom': '8px'}
        ),

        html.Label("Plot Style:"),
        dcc.Dropdown(
            id='plot-style-dropdown',
            options=[
                {'label': 'Surface', 'value': 'surface'},
                {'label': 'Wireframe', 'value': 'wireframe'},
                {'label': 'Heatmap', 'value': 'heatmap'},
                {'label': 'Scatter3D', 'value': 'scatter3d'}
            ],
            value='surface',
            clearable=False,
            style={'marginBottom': '8px'}
        ),

        html.Label("Color Palette:"),
        dcc.Dropdown(
            id='color-palette-dropdown',
            options=[
                {'label': 'Turbo', 'value': 'Turbo'},
                {'label': 'Viridis', 'value': 'Viridis'},
                {'label': 'Cividis', 'value': 'Cividis'},
                {'label': 'Plasma', 'value': 'Plasma'},
                {'label': 'Inferno', 'value': 'Inferno'},
                {'label': 'Magma', 'value': 'Magma'},
                {'label': 'Jet', 'value': 'Jet'},
                {'label': 'Rainbow', 'value': 'Rainbow'},
                {'label': 'Portland', 'value': 'Portland'},
                {'label': 'Earth', 'value': 'Earth'},
                {'label': 'Electric', 'value': 'Electric'},
                {'label': 'Hot', 'value': 'Hot'},
                {'label': 'Picnic', 'value': 'Picnic'},
                {'label': 'Blackbody', 'value': 'Blackbody'},
            ],
            value='Turbo',
            clearable=False,
            style={'marginBottom': '8px'}
        ),

        html.Label("Display Mode:"),
        dcc.RadioItems(
            id='display-mode',
            options=[
                {'label': 'Raw Values', 'value': 'raw'},
                {'label': 'Anomaly from Average', 'value': 'anomaly'}
            ],
            value='raw',
            labelStyle={'display': 'inline-block', 'marginRight': '10px'}
        ),

        html.Label("Surface Display:"),
        dcc.RadioItems(
            id='surface-mode',
            options=[
                {'label': 'Raw Surface', 'value': 'raw'},
                {'label': 'Smoothed Surface', 'value': 'smooth'}
            ],
            value='raw',
            labelStyle={'display': 'inline-block', 'marginRight': '10px'}
        ),

        html.Label("Show Threshold Plane:"),
        dcc.RadioItems(
            id='plane-toggle',
            options=[
                {'label': 'Yes', 'value': 'show'},
                {'label': 'No', 'value': 'hide'}
            ],
            value='hide',
            labelStyle={'display': 'inline-block', 'marginRight': '10px'}
        ),
        html.Div(id='slider-container'),  # Threshold slider will appear here

        html.Label("Threshold Plane Mode:"),
        dcc.Dropdown(
            id='threshold-mode-dropdown',
            options=[
                {'label': 'Black Plane', 'value': 'black'},
                {'label': 'Days Above Threshold', 'value': 'heatmap_above'},
                {'label': 'Days Below Threshold', 'value': 'heatmap_below'},
                {'label': 'Penetration Clustering', 'value': 'penetration_clustering'}
            ],
            value='black',
            clearable=False,
            style={'marginBottom': '8px'}
        ),

        html.Label("Select Start Month:"),
        dcc.Dropdown(
            id='start-month-dropdown',
            options=[
                {'label': 'January 1', 'value': '01-01'},
                {'label': 'January 15', 'value': '01-15'},
                {'label': 'February 1', 'value': '02-01'},
                {'label': 'February 15', 'value': '02-15'},
                {'label': 'March 1', 'value': '03-01'},
                {'label': 'March 15', 'value': '03-15'},
                {'label': 'April 1', 'value': '04-01'},
                {'label': 'April 15', 'value': '04-15'},
                {'label': 'May 1', 'value': '05-01'},
                {'label': 'May 15', 'value': '05-15'},
                {'label': 'June 1', 'value': '06-01'},
                {'label': 'June 15', 'value': '06-15'},
                {'label': 'July 1', 'value': '07-01'},
                {'label': 'July 15', 'value': '07-15'},
                {'label': 'August 1', 'value': '08-01'},
                {'label': 'August 15', 'value': '08-15'},
                {'label': 'September 1', 'value': '09-01'},
                {'label': 'September 15', 'value': '09-15'},
                {'label': 'October 1', 'value': '10-01'},
                {'label': 'October 15', 'value': '10-15'},
                {'label': 'November 1', 'value': '11-01'},
                {'label': 'November 15', 'value': '11-15'},
                {'label': 'December 1', 'value': '12-01'},
                {'label': 'December 15', 'value': '12-15'}
            ],
            value='12-01'
        ),

        # Axis controls as numeric input fields
        html.Div([
            dcc.Input(
                id='x-aspect-input',
                type='number',
                min=1,
                max=6,
                step=1,
                value=6,
                style={'width': '60px', 'marginRight': '8px'},
                placeholder='X Axis'
            ),
            dcc.Input(
                id='y-aspect-input',
                type='number',
                min=1,
                max=6,
                step=1,
                value=4,
                style={'width': '60px', 'marginRight': '8px'},
                placeholder='Y Axis'
            ),
            dcc.Input(
                id='z-aspect-input',
                type='number',
                min=1,
                max=6,
                step=1,
                value=2,
                style={'width': '60px'},
                placeholder='Z Axis'
            ),
        ], style={'marginTop': '10px', 'display': 'flex', 'flexDirection': 'row', 'gap': '4px'}),
        html.Label("Show ENSO Phases:"),
        dcc.RadioItems(
            id='enso-toggle',
            options=[
                {'label': 'Yes', 'value': 'show'},
                {'label': 'No', 'value': 'hide'}
            ],
            value='hide',
            labelStyle={'display': 'inline-block', 'marginRight': '10px'}
        ),
        html.Label("ENSO Bar Opacity:"),
        dcc.Slider(
            id='enso-opacity-slider',
            min=0.05,
            max=1.0,
            step=0.01,
            value=0.25,
            marks={0.05: '5%', 0.5: '50%', 1.0: '100%'},
            tooltip={'placement': 'bottom', 'always_visible': False}
        ),

        html.Label("Highlight Year Range:"),
        dcc.RangeSlider(
            id='year-range-slider',
            min=int(plot_data['year'].min()),
            max=int(plot_data['year'].max()),
            step=1,
            value=[int(plot_data['year'].min()), int(plot_data['year'].max())],
            marks=None,
            allowCross=False,
            tooltip={'placement': 'bottom', 'always_visible': False}
        ),

        html.Label("Outlier Management:"),
        html.Div([
            dcc.Input(
                id='outlier-count-input',
                type='number',
                min=0,
                max=100,
                step=1,
                value=0,
                placeholder='# of outliers',
                style={'width': '80px', 'marginRight': '8px'}
            ),
            dcc.Dropdown(
                id='outlier-method-dropdown',
                options=[
                    {'label': 'Keep All', 'value': 'none'},
                    {'label': 'Trim Extremes', 'value': 'trim'},
                    {'label': 'Cap to Percentile', 'value': 'cap'}
                ],
                value='none',
                style={'width': '130px', 'fontSize': '12px'},
                clearable=False
            )
        ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '4px'}),
        html.Div([
            html.P("Reduces impact of extreme events on color scale", 
                   style={'fontSize': '10px', 'color': 'gray', 'margin': '0'})
        ], style={'marginBottom': '8px'}),
    ]), "", plot_data.to_dict('records') 

# Update the threshold slider callback to remove the label
@app.callback(
    Output('slider-container', 'children'),
    Input('variable-dropdown', 'value'),
    Input('plot-data-store', 'data')
)
def update_slider(variable, plot_data_json):
    if plot_data_json is None:
        return None
    plot_data = pd.DataFrame(plot_data_json)
    values = plot_data.loc[plot_data['variable'] == variable, 'value']
    values_numeric = pd.to_numeric(values, errors='coerce')
    if not isinstance(values_numeric, pd.Series):
        values_numeric = pd.Series([values_numeric])
    values_numeric = values_numeric.dropna()
    return dcc.Slider(
        id='threshold-slider',
        min=0,
        max=100,
        step=0.1,
        value=95,
        marks={0: '0%', 50: '50%', 100: '100%'},
        tooltip={'placement': 'bottom', 'always_visible': False}
    )

@app.callback(
    Output('main-container', 'style'),
    Output('main-container', 'className'),
    Output('sidebar-container', 'style'),
    Output('value-3d-graph', 'style'),
    Input('dark-mode-toggle', 'value')
)
def update_dark_mode_styling(dark_mode):
    if dark_mode == 'dark':
        # Dark mode styles
        main_style = {
            'background': '#1e1e1e',
            'color': '#ffffff',
            'minHeight': '100vh'
        }
        main_class = 'dark-mode'
        
        sidebar_style = {
            'width': '220px',
            'minWidth': '180px',
            'maxWidth': '260px',
            'padding': '8px',
            'background': '#2d2d30',
            'borderRight': '1px solid #555',
            'height': '85vh',
            'overflowY': 'auto',
            'display': 'inline-block',
            'verticalAlign': 'top',
            'color': '#ffffff'
        }
        
        graph_style = {
            'height': '85vh',
            'width': '100%',
            'background': '#1e1e1e'
        }
    else:
        # Light mode styles (original)
        main_style = {
            'background': '#ffffff',
            'color': '#000000',
            'minHeight': '100vh'
        }
        main_class = ''
        
        sidebar_style = {
            'width': '220px',
            'minWidth': '180px',
            'maxWidth': '260px',
            'padding': '8px',
            'background': '#f8f8f8',
            'borderRight': '1px solid #ddd',
            'height': '85vh',
            'overflowY': 'auto',
            'display': 'inline-block',
            'verticalAlign': 'top'
        }
        
        graph_style = {
            'height': '85vh', 
            'width': '100%'
        }
    
    return main_style, main_class, sidebar_style, graph_style

@app.callback(
    [Output(f'{component_id}', 'style') for component_id in [
        'variable-dropdown', 'plot-style-dropdown', 'color-palette-dropdown', 
        'threshold-mode-dropdown', 'start-month-dropdown', 'outlier-method-dropdown'
    ]],
    [Output(f'{component_id}', 'style') for component_id in [
        'x-aspect-input', 'y-aspect-input', 'z-aspect-input', 'outlier-count-input'
    ]],
    Output('upload-button', 'style'),
    Input('dark-mode-toggle', 'value')
)
def update_control_styling(dark_mode):
    if dark_mode == 'dark':
        # Dark mode dropdown style - improved for better text contrast
        dropdown_style = {
            'marginBottom': '8px'
        }
        
        # Dark mode input style
        input_style = {
            'backgroundColor': '#3c3c3c',
            'color': '#ffffff',
            'border': '1px solid #555',
            'borderRadius': '4px'
        }
        
        # Dark mode button style
        button_style = {
            'width': '100%',
            'fontSize': 12,
            'padding': '4px',
            'backgroundColor': '#3c3c3c',
            'color': '#ffffff',
            'border': '1px solid #555',
            'borderRadius': '4px'
        }
    else:
        # Light mode styles
        dropdown_style = {'marginBottom': '8px'}
        input_style = {}
        button_style = {
            'width': '100%',
            'fontSize': 12,
            'padding': '4px'
        }
    
    # Return styles for dropdowns first, then inputs, then button
    dropdown_styles = [dropdown_style] * 6  # 6 dropdowns
    input_styles = [input_style] * 4  # 4 inputs
    
    return dropdown_styles + input_styles + [button_style]

@app.callback(
    Output('value-3d-graph', 'figure'),
    Input('variable-dropdown', 'value'),
    Input('start-month-dropdown', 'value'),
    Input('display-mode', 'value'),
    Input('surface-mode', 'value'),
    Input('threshold-slider', 'value'),
    Input('plane-toggle', 'value'),
    Input('threshold-mode-dropdown', 'value'),
    Input('x-aspect-input', 'value'),
    Input('y-aspect-input', 'value'),
    Input('z-aspect-input', 'value'),
    Input('color-palette-dropdown', 'value'),
    Input('plot-style-dropdown', 'value'),
    Input('enso-toggle', 'value'),
    Input('enso-opacity-slider', 'value'),
    Input('year-range-slider', 'value'),
    Input('outlier-count-input', 'value'),
    Input('outlier-method-dropdown', 'value'),
    Input('plot-data-store', 'data'),
    Input('dark-mode-toggle', 'value'),
    State('value-3d-graph', 'relayoutData')
)
def update_graph(selected_variable, start_month, display_mode, surface_mode, threshold_z, plane_toggle,
                 threshold_mode, x_aspect, y_aspect, z_aspect, color_palette, plot_style, enso_toggle, enso_opacity, year_range,
                 outlier_count, outlier_method, plot_data_json, dark_mode, relayout_data):
    if plot_data_json is None:
        return go.Figure()
    # Avoid unnecessary DataFrame copies
    plot_data = pd.DataFrame(plot_data_json)
    # Only keep relevant variable rows up front
    plot_data = plot_data[plot_data['variable'] == selected_variable].copy()
    # Precompute anchor_date once
    anchor_date = pd.to_datetime(f"2001-{start_month}")
    # Vectorized day_of_year calculation
    plot_data['day_of_year'] = (
        pd.to_datetime('2001-' + plot_data['md'], format='%Y-%m-%d', errors='coerce') - anchor_date
    ).dt.days % 365 + 1
    # Group and mean in one go
    df_var = plot_data.groupby(['day_of_year', 'year'], as_index=False)['value'].mean()
    # Vectorized anomaly calculation
    if display_mode == 'anomaly':
        climatology = df_var.groupby('day_of_year')['value'].mean()
        df_var = df_var.join(climatology, on='day_of_year', rsuffix='_clim')
        df_var['value'] = df_var['value'] - df_var['value_clim']
        df_var = df_var.drop(columns=['value_clim'])
    
    # Outlier management
    if outlier_count and outlier_count > 0 and outlier_method != 'none':
        if outlier_method == 'trim':
            # Remove the most extreme values (both high and low)
            sorted_values = df_var['value'].dropna().sort_values()
            if len(sorted_values) > outlier_count * 2:
                # Remove equal numbers from each end
                n_per_side = outlier_count // 2
                n_extra = outlier_count % 2  # handle odd numbers
                lower_cutoff = sorted_values.iloc[n_per_side]
                upper_cutoff = sorted_values.iloc[-(n_per_side + n_extra) - 1]
                df_var = df_var[(df_var['value'] >= lower_cutoff) & (df_var['value'] <= upper_cutoff)]
        
        elif outlier_method == 'cap':
            # Cap values to percentiles (preserves data points but normalizes extremes)
            sorted_values = df_var['value'].dropna().sort_values()
            if len(sorted_values) > outlier_count * 2:
                # Calculate percentile positions
                n_per_side = outlier_count // 2
                n_extra = outlier_count % 2
                lower_percentile = (n_per_side) / len(sorted_values) * 100
                upper_percentile = 100 - ((n_per_side + n_extra) / len(sorted_values) * 100)
                
                lower_cap = np.percentile(sorted_values, lower_percentile)
                upper_cap = np.percentile(sorted_values, upper_percentile)
                
                # Cap extreme values
                df_var['value'] = df_var['value'].clip(lower=lower_cap, upper=upper_cap)
    
    # Pivot to matrix
    pivot_table = df_var.pivot(index='day_of_year', columns='year', values='value')
    x = pivot_table.columns.values
    y = pivot_table.index.values
    z = pivot_table.values
    # Only smooth if needed
    mylar_surface = z if surface_mode == 'raw' else gaussian_filter(z, sigma=[3, 1.5], mode='nearest')
    # Month labels and days
    base_year = 2001
    anchor_date = pd.to_datetime(f"{base_year}-{start_month}")
    month_starts = [anchor_date + pd.DateOffset(months=i) for i in range(12)]
    month_labels = [d.strftime('%b %d') for d in month_starts]
    month_days = [(d - anchor_date).days + 1 for d in month_starts]
    # Hovertemplate data
    date_labels = [(anchor_date + pd.Timedelta(days=int(d)-1)).strftime('%b %d') for d in y]
    customdata_2d = np.tile(date_labels, (len(x), 1)).T
    customdata_1d = customdata_2d.flatten()
    camera = relayout_data['scene.camera'] if relayout_data and 'scene.camera' in relayout_data else None
    fig = go.Figure()

    # --- Plot Style Logic ---
    if plot_style == 'surface':
        # Use surface or smoothed surface
        z_data = z if surface_mode == 'raw' else mylar_surface
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z_data,
                colorscale=color_palette,
                colorbar=dict(title='Value', x=1.0),
                customdata=customdata_2d,
                hovertemplate='<b>Year:</b> %{x}<br><b>Date:</b> %{customdata}<br><b>Value:</b> %{z:.2f}<extra></extra>',
                name='Surface',
                showscale=True,
                opacity=1.0
            )
        )
    elif plot_style == 'wireframe':
        # Wireframe: Surface with low opacity and no fill
        z_data = z if surface_mode == 'raw' else mylar_surface
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z_data,
                colorscale=color_palette,
                colorbar=dict(title='Value'),
                customdata=customdata_2d,
                hovertemplate='<b>Year:</b> %{x}<br><b>Date:</b> %{customdata}<br><b>Value:</b> %{z:.2f}<extra></extra>',
                name='Wireframe',
                showscale=False,
                opacity=0.3,
                contours = {
                    "z": {"show": True, "start": np.nanmin(z_data), "end": np.nanmax(z_data), "size": (np.nanmax(z_data)-np.nanmin(z_data))/15, "color":"black"}
                }
            )
        )
    elif plot_style == 'heatmap':
        # 2D Heatmap (flattened)
        z_data = z if surface_mode == 'raw' else mylar_surface
        fig.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=z_data,
                colorscale=color_palette,
                colorbar=dict(title='Value'),
                customdata=customdata_2d,
                hovertemplate='<b>Year:</b> %{x}<br><b>Date:</b> %{customdata}<br><b>Value:</b> %{z:.2f}<extra></extra>',
                name='Heatmap'
            )
        )
    elif plot_style == 'scatter3d':
        # 3D Scatter
        z_data = z if surface_mode == 'raw' else mylar_surface
        xg, yg = np.meshgrid(x, y)
        fig.add_trace(
            go.Scatter3d(
                x=xg.flatten(),
                y=yg.flatten(),
                z=z_data.flatten(),
                mode='markers',
                marker=dict(
                    size=2,
                    color=z_data.flatten(),
                    colorscale=color_palette,
                    colorbar=dict(title='Value')
                ),
                customdata=customdata_1d,
                hovertemplate='<b>Year:</b> %{x}<br><b>Date:</b> %{customdata}<br><b>Value:</b> %{z:.2f}<extra></extra>',
                name='Scatter3D'
            )
        )

    # --- Threshold Plane ---
    values = plot_data.loc[plot_data['variable'] == selected_variable, 'value']
    values_numeric = pd.to_numeric(values, errors='coerce')
    if not isinstance(values_numeric, pd.Series):
        values_numeric = pd.Series([values_numeric])
    values_numeric = values_numeric.dropna()
    if values_numeric.empty:
        threshold_value = 0
    else:
        min_val = float(np.floor(values_numeric.min()))
        max_val = float(np.ceil(values_numeric.max()))
        threshold_value = min_val + (max_val - min_val) * (threshold_z / 100.0)

    if plane_toggle == 'show':
        z_plane = np.full_like(z, fill_value=threshold_value, dtype=float)
        
        if threshold_mode == 'heatmap_above':
            # Bin the number of instances above the threshold for each year (x-axis)
            # z shape: (len(y), len(x)), where x = years, y = day_of_year
            # For each year (column), count number of days above threshold
            band_counts = (z > threshold_value).sum(axis=0)  # shape: (len(x),)
            # Repeat band_counts for each y to make a 2D array for surfacecolor
            band_matrix = np.tile(band_counts, (len(y), 1))
            # Prepare customdata for tooltips: year, day, count
            year_labels = np.array(x)
            day_labels = np.array(y)
            customdata_plane = np.empty((len(y), len(x)), dtype=object)
            for i in range(len(y)):
                for j in range(len(x)):
                    customdata_plane[i, j] = [year_labels[j], day_labels[i], band_matrix[i, j]]
            fig.add_trace(
                go.Surface(
                    x=x,
                    y=y,
                    z=z_plane,
                    surfacecolor=band_matrix,
                    colorscale=color_palette,
                    colorbar=dict(title='Days Above Threshold (per Year)', x=1.13),
                    showscale=True,
                    opacity=1.0,
                    name='Threshold Bands Above',
                    customdata=customdata_plane,
                    hovertemplate='<b>Year:</b> %{customdata[0]}<br><b>Day of Year:</b> %{customdata[1]}<br><b>Days Above Threshold:</b> %{customdata[2]}<extra></extra>'
                )
            )
        elif threshold_mode == 'heatmap_below':
            # Bin the number of instances below the threshold for each year (x-axis)
            # z shape: (len(y), len(x)), where x = years, y = day_of_year
            # For each year (column), count number of days below threshold
            band_counts = (z < threshold_value).sum(axis=0)  # shape: (len(x),)
            # Repeat band_counts for each y to make a 2D array for surfacecolor
            band_matrix = np.tile(band_counts, (len(y), 1))
            # Prepare customdata for tooltips: year, day, count
            year_labels = np.array(x)
            day_labels = np.array(y)
            customdata_plane = np.empty((len(y), len(x)), dtype=object)
            for i in range(len(y)):
                for j in range(len(x)):
                    customdata_plane[i, j] = [year_labels[j], day_labels[i], band_matrix[i, j]]
            fig.add_trace(
                go.Surface(
                    x=x,
                    y=y,
                    z=z_plane,
                    surfacecolor=band_matrix,
                    colorscale=color_palette,
                    colorbar=dict(title='Days Below Threshold (per Year)', x=1.13),
                    showscale=True,
                    opacity=1.0,
                    name='Threshold Bands Below',
                    customdata=customdata_plane,
                    hovertemplate='<b>Year:</b> %{customdata[0]}<br><b>Day of Year:</b> %{customdata[1]}<br><b>Days Below Threshold:</b> %{customdata[2]}<extra></extra>'
                )
            )
        elif threshold_mode == 'penetration_clustering':
            # Create density heatmap of threshold penetrations
            # Find all penetration points (where data crosses threshold)
            penetration_mask = (z > threshold_value)  # Can be modified for below threshold too
            
            # Create a density matrix to accumulate clustering
            density_matrix = np.zeros_like(z, dtype=float)
            
            # Define clustering radius (in grid units)
            cluster_radius_y = 15  # days
            cluster_radius_x = 2   # years
            
            # For each penetration point, add a Gaussian blob
            y_indices, x_indices = np.where(penetration_mask)
            
            for i in range(len(y_indices)):
                yi, xi = y_indices[i], x_indices[i]
                
                # Create Gaussian kernel around this penetration
                for dy in range(-cluster_radius_y, cluster_radius_y + 1):
                    for dx in range(-cluster_radius_x, cluster_radius_x + 1):
                        ny, nx = yi + dy, xi + dx
                        
                        # Check bounds
                        if 0 <= ny < len(y) and 0 <= nx < len(x):
                            # Gaussian weight based on distance
                            dist_y = dy / cluster_radius_y
                            dist_x = dx / cluster_radius_x
                            weight = np.exp(-(dist_y**2 + dist_x**2) / 2)
                            density_matrix[ny, nx] += weight
            
            # Normalize density for better visualization
            if density_matrix.max() > 0:
                density_matrix = density_matrix / density_matrix.max()
            
            # Create customdata for tooltips
            year_labels = np.array(x)
            day_labels = np.array(y)
            customdata_plane = np.empty((len(y), len(x)), dtype=object)
            for i in range(len(y)):
                for j in range(len(x)):
                    customdata_plane[i, j] = [year_labels[j], day_labels[i], density_matrix[i, j]]
            
            fig.add_trace(
                go.Surface(
                    x=x,
                    y=y,
                    z=z_plane,
                    surfacecolor=density_matrix,
                    colorscale='Hot',  # Good for showing clustering intensity
                    colorbar=dict(title='Penetration Density', x=1.13),
                    showscale=True,
                    opacity=0.9,
                    name='Threshold Penetration Clustering',
                    customdata=customdata_plane,
                    hovertemplate='<b>Year:</b> %{customdata[0]}<br><b>Day of Year:</b> %{customdata[1]}<br><b>Cluster Density:</b> %{customdata[2]:.3f}<extra></extra>'
                )
            )
        else:  # threshold_mode == 'black'
            # Simple black plane
            fig.add_trace(
                go.Surface(
                    x=x,
                    y=y,
                    z=z_plane,
                    colorscale=[[0, 'black'], [1, 'black']],
                    showscale=False,
                    opacity=0.8,
                    name='Threshold Plane',
                    hovertemplate='<b>Threshold Value:</b> %{z:.2f}<extra></extra>'
                )
            )

    # --- ENSO overlays (monthly curtains; sign color, intensity -> opacity) ---
    if enso_toggle == 'show' and ENSO_DF is not None and len(x) > 1 and len(y) > 1:
        min_year, max_year = year_range

        # Y-axis anchor: reuse the same anchor_date used for the surface
        base_year = 2001
        anchor_date = pd.to_datetime(f"{base_year}-{start_month}")

        # Z extents to span the main surface
        z_min, z_max = float(np.nanmin(z)), float(np.nanmax(z))
        if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min == z_max:
            z_min, z_max = -1.0, 1.0  # fallback

        # Limit ENSO rows to visible years
        enso = ENSO_DF[(ENSO_DF["year"] >= min_year) & (ENSO_DF["year"] <= max_year)].copy()
        if not enso.empty:
            # Global multiplier from UI slider (0..1)
            global_mult = float(enso_opacity)

            # For y mapping we only need month boundaries relative to anchor.
            # We'll compute the band's start (month 1st) and end (following month 1st) in the anchored year (base_year).
            # If the band wraps across the 365-day boundary, we split into two curtains.
            def y_pos_for_month(m):
                d = pd.to_datetime(f"{base_year}-{int(m):02d}-01")
                return ((d - anchor_date).days % 365) + 1

            # group by (year, sign, intensity) to keep trace count modest
            for (yr, sign, intensity), g in enso.groupby(["year", "sign", "intensity"]):
                # Skip Neutral (opacity level 0 -> invisible)
                base_opacity = ENSO_OPACITY_LEVEL.get(intensity, 0.0)
                if base_opacity <= 0.0:
                    continue
                final_opacity = max(0.0, min(1.0, base_opacity * global_mult))
                if final_opacity <= 0.0:
                    continue

                # Only render if the year exists on the X axis
                if yr not in x:
                    continue

                # Color by sign
                color = ENSO_SIGN_COLOR.get(sign, "rgba(200,200,200,1.0)")
                # Make a tiny colorscale with the same color
                colorscale = [[0, color], [1, color]]

                # Build one trace per (year, sign, intensity) with multiple small rectangles (one per month present)
                # Since Surface can't vary opacity per cell, we keep a single trace with constant opacity,
                # adding many small 2x2 patches via list extension of add_surface (requires separate traces);
                # So we actually add one trace PER MONTH here to preserve sharp edges but still batch by (yr,sign,intensity).
                # To limit trace count, we still keep intensity & sign grouping.
                months = sorted(g["month"].unique())
                for m in months:
                    # Y start/end in anchored day-of-year coords
                    y_start = y_pos_for_month(m)
                    # next month (wrap to 1 after 12)
                    m_next = 1 if m == 12 else m + 1
                    y_end = y_pos_for_month(m_next)

                    def add_band(y0, y1):
                        # Build a 2x2 surface that spans z_min..z_max at fixed x=yr and y from y0..y1
                        x_grid = np.array([[yr, yr], [yr, yr]])
                        y_grid = np.array([[y0, y0], [y1, y1]])
                        z_grid = np.array([[z_min, z_max], [z_min, z_max]])
                        fig.add_trace(
                            go.Surface(
                                x=x_grid,
                                y=y_grid,
                                z=z_grid,
                                surfacecolor=np.zeros_like(z_grid),
                                colorscale=colorscale,
                                showscale=False,
                                opacity=final_opacity,
                                name=f"ENSO {yr} {sign} {intensity}",
                                hoverinfo='skip'  # no tooltips
                            )
                        )

                    # If the band wraps past 365, split it
                    if y_end > y_start:
                        add_band(y_start, y_end)
                    else:
                        # segment 1: y_start -> 365
                        add_band(y_start, y.max())
                        # segment 2: 1 -> y_end
                        add_band(y.min(), y_end)


    # Configure x-axis ticks based on year range to avoid overcrowding
    year_span = max(x) - min(x) if len(x) > 0 else 0
    if year_span <= 5:
        # Very small range: show every year but no tick labels to avoid overlap
        x_tick_vals = list(range(int(min(x)), int(max(x)) + 1))
        x_tick_text = [''] * len(x_tick_vals)  # Empty labels to prevent crowding
    elif year_span <= 15:
        # Small range: show every 2-3 years
        tick_step = 2 if year_span <= 10 else 3
        x_tick_vals = list(range(int(min(x)), int(max(x)) + 1, tick_step))
        x_tick_text = [str(year) for year in x_tick_vals]
    elif year_span <= 50:
        # Medium range: show every 5 years
        x_tick_vals = list(range(int(min(x)), int(max(x)) + 1, 5))
        x_tick_text = [str(year) for year in x_tick_vals]
    else:
        # Large range: show every 10 years
        x_tick_vals = list(range(int(min(x)), int(max(x)) + 1, 10))
        x_tick_text = [str(year) for year in x_tick_vals]

    # Configure scene styling based on dark mode
    if dark_mode == 'dark':
        # Dark mode scene colors
        scene_bgcolor = '#1e1e1e'
        axis_color = '#ffffff'
        gridline_color = '#555555'
        axis_bgcolor = '#2d2d30'
    else:
        # Light mode scene colors (default)
        scene_bgcolor = 'white'
        axis_color = '#000000'
        gridline_color = '#cccccc'
        axis_bgcolor = 'rgba(240,240,240,0.8)'

    scene_config = dict(
        bgcolor=scene_bgcolor,
        xaxis=dict(
            title=dict(text="", font=dict(size=1, color='rgba(0,0,0,0)')),
            tickmode='array',
            tickvals=x_tick_vals,
            ticktext=x_tick_text,
            color=axis_color,
            gridcolor=gridline_color,
            backgroundcolor=axis_bgcolor,
            showbackground=True
        ),
        yaxis=dict(
            title=dict(text="", font=dict(size=1, color='rgba(0,0,0,0)')),
            tickmode='array',
            tickvals=month_days,
            ticktext=month_labels,
            color=axis_color,
            gridcolor=gridline_color,
            backgroundcolor=axis_bgcolor,
            showbackground=True
        ),
        zaxis=dict(
            title=dict(text="", font=dict(size=1, color='rgba(0,0,0,0)')),
            color=axis_color,
            gridcolor=gridline_color,
            backgroundcolor=axis_bgcolor,
            showbackground=True
        ),
        aspectratio=dict(x=x_aspect, y=y_aspect, z=z_aspect)
    )
    if camera:
        scene_config['camera'] = camera

    # Configure dark/light mode plot styling
    if dark_mode == 'dark':
        plot_bgcolor = '#1e1e1e'
        paper_bgcolor = '#1e1e1e'
        font_color = '#ffffff'
    else:
        plot_bgcolor = 'white'
        paper_bgcolor = 'white'
        font_color = '#000000'

    fig.update_layout(
        scene=scene_config,
        margin=dict(l=0, r=0, b=0, t=40),
        title=f"{'Δ from Avg' if display_mode == 'anomaly' else 'Raw'} {selected_variable} by Day of Year",
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        font=dict(color=font_color)
    )

    return fig

import webbrowser

if __name__ == '__main__':
    webbrowser.open_new("http://127.0.0.1:8050/")
    app.run(debug=False)
