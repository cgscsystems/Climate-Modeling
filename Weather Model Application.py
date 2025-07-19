import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State
from tkinter import Tk, filedialog
from scipy.ndimage import gaussian_filter

# ---- File Picker ----
root = Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select Weather CSV File", filetypes=[("CSV files", "*.csv")])

# ---- Load and Clean Data ----
df = pd.read_csv(file_path)
df.columns = [col.strip().lower() for col in df.columns]
df.rename(columns={df.columns[0]: "date"}, inplace=True)  # Ensure first column is named "date"
df['date'] = pd.to_datetime(df['date'].astype(str).str.strip(), format='%Y-%m-%d', errors='coerce')
df = df.dropna(axis=1, how='all')
df['year'] = df['date'].apply(
    lambda d: d.year + 1 if d.month == 12 else d.year
).astype('Int64')
df = df[df['year'].notna()]

# ---- Melt and Prepare ----
df['md'] = df['date'].dt.strftime('%m-%d')
df_melted = df.melt(id_vars=['date', 'year', 'md'], var_name='variable', value_name='value')
df_melted.sort_values(by=['variable', 'date'], inplace=True)
plot_data = df_melted.dropna(subset=['value'])[['variable', 'year', 'md', 'date', 'value']]

# ---- Dash App ----
app = Dash(__name__)

# Add aspect ratio sliders to the layout
app.layout = html.Div([
    html.H1("Absolute Weather Value Viewer"),
    
    html.Label("Select Variable:"),
    dcc.Dropdown(
        id='variable-dropdown',
        options=[{'label': var, 'value': var} for var in sorted(plot_data['variable'].unique())],
        value=plot_data['variable'].unique()[0]
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

    html.Label("X Axis Size:"),
    dcc.Slider(
        id='x-aspect-slider',
        min=1, max=5, step=0.1, value=2,
        marks={i: str(i) for i in range(1, 6)}
    ),
    html.Label("Y Axis Size:"),
    dcc.Slider(
        id='y-aspect-slider',
        min=1, max=5, step=0.1, value=4,
        marks={i: str(i) for i in range(1, 6)}
    ),
    html.Label("Z Axis Size:"),
    dcc.Slider(
        id='z-aspect-slider',
        min=0.5, max=3, step=0.1, value=1,
        marks={i: str(i) for i in range(1, 4)}
    ),

    html.Div(id='slider-container'),

    dcc.Graph(id='value-3d-graph', style={'height': '80vh'})
], style={'height': '100vh', 'margin': 0, 'padding': 10})

# ---- Slider for Dynamic Range Based on Variable ----
@app.callback(
    Output('slider-container', 'children'),
    Input('variable-dropdown', 'value')
)
def update_slider(variable):
    values = plot_data.loc[plot_data['variable'] == variable, 'value']
    values_numeric = pd.to_numeric(values, errors='coerce')
    if not isinstance(values_numeric, pd.Series):
        values_numeric = pd.Series([values_numeric])
    values_numeric = values_numeric.dropna()
    return dcc.Slider(
        id='threshold-slider',
        min=float(np.floor(values_numeric.min())),
        max=float(np.ceil(values_numeric.max())),
        step=1,
        value=float(np.percentile(values_numeric, 95)),
        tooltip={'placement': 'bottom', 'always_visible': True}
    )

# ---- Main Plot Callback ----
@app.callback(
    Output('value-3d-graph', 'figure'),
    Input('variable-dropdown', 'value'),
    Input('start-month-dropdown', 'value'),
    Input('display-mode', 'value'),
    Input('surface-mode', 'value'),
    Input('threshold-slider', 'value'),
    Input('plane-toggle', 'value'),
    Input('x-aspect-slider', 'value'),
    Input('y-aspect-slider', 'value'),
    Input('z-aspect-slider', 'value'),
    State('value-3d-graph', 'relayoutData')
)
def update_graph(selected_variable, start_month, display_mode, surface_mode, threshold_z, plane_toggle,
                 x_aspect, y_aspect, z_aspect, relayout_data):
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

    # For hovertemplate, create date labels for each y value
    date_labels = [(anchor_date + pd.Timedelta(days=int(d)-1)).strftime('%b %d') for d in y]
    customdata = [date_labels for _ in range(len(x))]

    camera = relayout_data['scene.camera'] if relayout_data and 'scene.camera' in relayout_data else None

    fig = go.Figure()

    if surface_mode == 'raw':
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                colorscale='Turbo',
                colorbar=dict(title='Value'),
                customdata=customdata,
                hovertemplate='<b>Year:</b> %{x}<br><b>Date:</b> %{customdata}<br><b>Value:</b> %{z:.2f}<extra></extra>',
                name='Observed',
                showscale=True,
                opacity=1.0
            )
        )

    if surface_mode == 'smooth':
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=mylar_surface,
                colorscale='Turbo',
                colorbar=dict(title='Value'),
                customdata=customdata,
                hovertemplate='<b>Year:</b> %{x}<br><b>Date:</b> %{customdata}<br><b>Value:</b> %{z:.2f}<extra></extra>',
                name='Smoothed',
                showscale=True,
                opacity=1.0
            )
        )

    if plane_toggle == 'show':
        z_plane = np.full_like(z, fill_value=threshold_z)
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

import webbrowser
if __name__ == '__main__':
    webbrowser.open_new("http://127.0.0.1:8050/")
    app.run(debug=False)
