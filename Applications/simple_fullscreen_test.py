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
import webbrowser
import platform
from urllib.parse import urljoin
from typing import Dict, List, Tuple, Union, Optional
from datetime import datetime, timedelta

# Initialize Dash app
app = Dash(__name__)

# Simple fullscreen test
app.layout = html.Div([
    html.H1("Fullscreen Test"),
    html.Button('🔍 FULLSCREEN', id='fullscreen-btn', style={
        'position': 'fixed',
        'top': '20px',
        'right': '20px',
        'padding': '10px 15px',
        'background': '#ff0000',
        'color': 'white',
        'border': 'none',
        'borderRadius': '5px',
        'fontSize': '14px',
        'zIndex': '9999'
    }),
    html.Div("This is a test page. Click the red FULLSCREEN button.", style={
        'padding': '50px',
        'fontSize': '18px'
    })
])

@app.callback(
    Output('fullscreen-btn', 'children'),
    Input('fullscreen-btn', 'n_clicks'),
    prevent_initial_call=True
)
def toggle_fullscreen(n_clicks):
    return f"Clicked {n_clicks} times!"

if __name__ == '__main__':
    print("Simple Fullscreen Test")
    print("URL: http://127.0.0.1:8050/")
    webbrowser.open_new("http://127.0.0.1:8050/")
    app.run(debug=False)