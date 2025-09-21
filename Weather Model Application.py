import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx
from scipy.ndimage import gaussian_filter
import base64
import io
import requests
import re
from datetime import datetime

# ---- Dash App ----
app = Dash(__name__)

# ---- App Layout ----
app.layout = html.Div([
    html.H1("Absolute Weather Value Viewer", style={'margin': '0 0 10px 0', 'fontSize': 24}),
    dcc.Store(id='plot-data-store'),  # <-- Add this line!
    html.Div([
        html.Div([
            dcc.Upload(
                id='upload-data',
                children=html.Button('Upload Weather CSV', style={'width': '100%', 'fontSize': 12, 'padding': '4px'}),
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
], style={'height': '100vh', 'margin': 0, 'padding': 0, 'fontFamily': 'Segoe UI, Arial, sans-serif'})

# ---- Helper: Parse Uploaded CSV ----
def parse_contents(contents):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        # Try utf-8 first
        try:
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        except UnicodeDecodeError:
            # Fallback to latin1 if utf-8 fails
            df = pd.read_csv(io.StringIO(decoded.decode('latin1')))
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
        return None, str(e)

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
        html.Label("Select Variable:"),
        dcc.Dropdown(
            id='variable-dropdown',
            options=[{'label': var, 'value': var} for var in sorted(plot_data['variable'].unique())],
            value=plot_data['variable'].unique()[0]
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
                step=0.1,
                value=2,
                style={'width': '60px', 'marginRight': '8px'},
                placeholder='X Axis'
            ),
            dcc.Input(
                id='y-aspect-input',
                type='number',
                min=1,
                max=6,
                step=0.1,
                value=4,
                style={'width': '60px', 'marginRight': '8px'},
                placeholder='Y Axis'
            ),
            dcc.Input(
                id='z-aspect-input',
                type='number',
                min=1,
                max=6,
                step=0.1,
                value=1,
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
            marks={int(y): str(int(y)) for y in sorted(plot_data['year'].unique())},
            allowCross=False,
            tooltip={'placement': 'bottom', 'always_visible': False}
        ),
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
    Output('value-3d-graph', 'figure'),
    Input('variable-dropdown', 'value'),
    Input('start-month-dropdown', 'value'),
    Input('display-mode', 'value'),
    Input('surface-mode', 'value'),
    Input('threshold-slider', 'value'),
    Input('plane-toggle', 'value'),
    Input('x-aspect-input', 'value'),
    Input('y-aspect-input', 'value'),
    Input('z-aspect-input', 'value'),
    Input('color-palette-dropdown', 'value'),
    Input('plot-style-dropdown', 'value'),
    Input('enso-toggle', 'value'),
    Input('enso-opacity-slider', 'value'),
    Input('year-range-slider', 'value'),  # <-- Add this line!
    Input('plot-data-store', 'data'),
    State('value-3d-graph', 'relayoutData')
)
def update_graph(selected_variable, start_month, display_mode, surface_mode, threshold_z, plane_toggle,
                 x_aspect, y_aspect, z_aspect, color_palette, plot_style, enso_toggle, enso_opacity, year_range,
                 plot_data_json, relayout_data):
    if plot_data_json is None:
        return go.Figure()
    plot_data = pd.DataFrame(plot_data_json)
    df = plot_data.copy()
    anchor_date = pd.to_datetime(f"2001-{start_month}")
    df['day_of_year'] = (
        pd.to_datetime('2001-' + df['md'], format='%Y-%m-%d', errors='coerce') - anchor_date
    ).dt.days % 365 + 1

    df_var = (
        df[df['variable'] == selected_variable]
        .groupby(['day_of_year', 'year'], as_index=False)['value']
        .mean()
    )

    if display_mode == 'anomaly':
        climatology = df_var.groupby('day_of_year')['value'].mean()
        df_var = df_var.copy()
        df_var['value'] = df_var.apply(lambda row: row['value'] - climatology.get(row['day_of_year'], 0), axis=1)

    pivot_table = df_var.pivot(index='day_of_year', columns='year', values='value')
    x = pivot_table.columns.values
    y = pivot_table.index.values
    z = pivot_table.values
    mylar_surface = gaussian_filter(z, sigma=[3, 1.5], mode='nearest')

    # Generate month labels for y-axis
    base_year = 2001
    anchor_date = pd.to_datetime(f"{base_year}-{start_month}")
    month_starts = [anchor_date + pd.DateOffset(months=i) for i in range(12)]
    month_labels = [d.strftime('%b %d') for d in month_starts]
    month_days = [(d - anchor_date).days + 1 for d in month_starts]

    # For hovertemplate, create date labels for each y value (relative to anchor_date)
    date_labels = [(anchor_date + pd.Timedelta(days=int(d)-1)).strftime('%b %d') for d in y]
    customdata = [date_labels for _ in range(len(x))]
    customdata_2d = np.array(customdata).T  # shape: (len(y), len(x))
    customdata_1d = customdata_2d.flatten() # for scatter3d

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
                colorbar=dict(title='Value'),
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
        z_plane = np.full_like(z, fill_value=threshold_value)
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z_plane,
                colorscale=[[0, 'black'], [1, 'black']],
                showscale=False,
                opacity=1.0,
                name='Threshold',
                hoverinfo='skip'
            )
        )

    # --- ENSO overlays as continuous 3D bands ---
    if enso_toggle == 'show' and len(x) > 1 and len(y) > 1:
        min_year, max_year = year_range
        for start, end, phase in ENSO_PHASES:
            start_year, start_month = map(int, start.split('-'))
            end_year, end_month = map(int, end.split('-'))
            for year in x:
                year = int(year)
                # Only shade if year is within the selected range
                if year < start_year or year > end_year or year < min_year or year > max_year:
                    continue
                # Start and end day-of-year for this year
                if year == start_year:
                    enso_start = pd.to_datetime(f"{year}-{start_month:02d}-01")
                    y_start = ((enso_start - anchor_date).days % 365) + 1
                else:
                    y_start = y.min()
                if year == end_year:
                    enso_end = pd.to_datetime(f"{year}-{end_month:02d}-01")
                    y_end = ((enso_end - anchor_date).days % 365) + 1
                else:
                    y_end = y.max()
                # Mask for y within this interval
                y_mask = (y >= y_start) & (y <= y_end)
                if not np.any(y_mask):
                    continue
                # Create a surface for this (year, y_mask) region
                y_band = y[y_mask]
                x_band = np.full_like(y_band, year)
                z_min, z_max = np.nanmin(z), np.nanmax(z)
                y_grid, z_grid = np.meshgrid(y_band, [z_min, z_max])
                x_grid = np.full_like(y_grid, year)
                fig.add_trace(
                    go.Surface(
                        x=x_grid,
                        y=y_grid,
                        z=z_grid,
                        showscale=False,
                        opacity=enso_opacity,
                        surfacecolor=np.full_like(y_grid, 0),
                        colorscale=[[0, ENSO_COLORS.get(phase, "rgba(255,255,255,0.15)")],
                                    [1, ENSO_COLORS.get(phase, "rgba(255,255,255,0.15)")]],
                        name=phase,
                        hoverinfo='skip'
                    )
                )

    scene_config = dict(
        xaxis_title='Year',
        yaxis_title=f'Day ({start_month} Start)',
        zaxis_title='Measured Value',
        aspectratio=dict(x=x_aspect, y=y_aspect, z=z_aspect)
    )
    if camera:
        scene_config['camera'] = camera

    fig.update_layout(
        scene=dict(
            **scene_config,
            yaxis=dict(
                tickmode='array',
                tickvals=month_days,
                ticktext=month_labels,
                title=f'Day ({start_month} Start)'
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        title=f"{'Δ from Avg' if display_mode == 'anomaly' else 'Raw'} {selected_variable} by Day of Year"
    )

    return fig

def load_enso_phases(filepath="enso_phases.csv"):
    df = pd.read_csv(filepath)
    return [(row['start'], row['end'], row['phase']) for _, row in df.iterrows()]

ENSO_PHASES = load_enso_phases()

# ENSO phase color mapping
ENSO_COLORS = {
    "El Niño": "rgba(255, 80, 80, 0.5)",    # Red-ish
    "La Niña": "rgba(80, 80, 255, 0.5)",    # Blue-ish
    "Neutral": "rgba(200, 200, 200, 0.3)"   # Gray-ish
}

import webbrowser
if __name__ == '__main__':
    webbrowser.open_new("http://127.0.0.1:8050/")
    app.run(debug=False)
