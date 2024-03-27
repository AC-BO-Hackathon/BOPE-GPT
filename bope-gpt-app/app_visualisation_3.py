import flask
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import numpy as np
from flask import request
from flask_cors import CORS  # Step 1: Import CORS

# Initialize Flask server
server = flask.Flask(__name__)
CORS(server)  # Step 2: Apply CORS to your Flask server instance. This enables CORS for all routes.

# Initialize the Dash app
app = dash.Dash(__name__, server=server)

# App layout
app.layout = html.Div([
    dcc.Graph(id='sine-wave-graph'),  # Graph component to display the plot
    html.Label('Frequency:'),  # Label for the slider
    dcc.Slider(
        id='frequency-slider',
        min=1,  # Minimum value of the slider
        max=10,  # Maximum value of the slider
        value=2,  # Initial value of the slider
        marks={i: str(i) for i in range(1, 11)},  # Marks on the slider for readability
        step=0.1,  # Step size of the slider
    ),
])

# Flask route to handle POST requests
@app.server.route('/submit', methods=['POST'])
def handle_data():
    data = request.get_json()
    # Here, you would process the received data and potentially update Dash app components or data sources
    print(data)
    # For demonstration, just return a success status
    return {'status': 'success'}, 200

# Callback to update the graph based on the slider input
@app.callback(
    Output('sine-wave-graph', 'figure'),
    [Input('frequency-slider', 'value')]
)
def update_graph(frequency):
    # This function generates and returns a sine wave graph based on the slider's value
    x_values = np.linspace(0, 10, 500)
    y_values = np.sin(frequency * x_values)

    figure = {
        'data': [go.Scatter(x=x_values, y=y_values, mode='lines')],
        'layout': go.Layout(
            title='Dynamic Sine Wave',
            xaxis={'title': 'X Value'},
            yaxis={'title': 'Sin(X)'},
        ),
    }
    return figure

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, host="127.0.0.1", port=8080)
