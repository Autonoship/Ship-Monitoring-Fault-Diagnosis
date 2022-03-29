import dash
from dash import dcc, html, dash_table, State
import plotly
from dash.dependencies import Input, Output
import pandas as pd
from plotly.subplots import make_subplots
from collections import deque
import plotly.graph_objects as go
from joblib import load
import os

parameter_range = {
    'GAS OUT': [10, 600],
    'SCAV AIR': [20, 80],
    'LOAD': [0, 100],
    'RPM': [-70, 130]
}

# Set environnment variables
MODEL_DIR = os.environ["MODEL_DIR"]
MODEL_FILE_LDA = os.environ["MODEL_FILE_LDA"]
MODEL_FILE_NN = os.environ["MODEL_FILE_NN"]
MODEL_PATH_LDA = os.path.join(MODEL_DIR, MODEL_FILE_LDA)
MODEL_PATH_NN = os.path.join(MODEL_DIR, MODEL_FILE_NN)

# Loading LDA model
print("Loading model from: {}".format(MODEL_PATH_LDA))
inference_lda = load(MODEL_PATH_LDA)

# loading Neural Network model
print("Loading model from: {}".format(MODEL_PATH_NN))
inference_NN = load(MODEL_PATH_NN)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
raw_data = pd.read_csv('./test_ASC.csv')
exh_gas_names = raw_data.columns[:8]
exh_gas = [deque([]) for _ in range(len(exh_gas_names))]
cyl_scav_air_names = raw_data.columns[8:16]
cyl_scav_air = [deque([]) for _ in range(len(cyl_scav_air_names))]
load_rpm_names = raw_data.columns[16:18]
load_rpm = [deque([]) for _ in range(len(load_rpm_names))]
model_names = ['lda', 'nn', 'ground_truth']
model = [deque([]) for _ in range(len(model_names))]

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    html.Div(
        id = 'whole',
        children = 
        [
        html.H1('ML based Anomaly Prediction Simulation'),
        html.Hr(style={'color':'#fff'}),
        html.Div([
            html.H2('Settings',style={'margin-bottom':'20px'}),
            html.Div([
            html.H4('Duration',style={'display':'inline-block','margin-right':20}),
            dcc.Input(id='input-on-duration',type='number', min=0, placeholder='please enter a number', value = 0, style={'display':'inline-block', 'border': '1px solid black'}),
            html.H4('Step',style={'display':'inline-block','margin-left':20, 'margin-right':20}),
            dcc.Input(id='input-on-step',type='number', min=0, placeholder='please enter a number', value = 0, style={'display':'inline-block', 'border': '1px solid black'}),
            html.H6('milliseconds', style={'display':'inline-block','margin-left':10, 'margin-right':20})
            ], style={'display':'inline-block', 'width': '100%'}),
            html.Button('Simulate', id='submit-val', n_clicks=0, disabled=False, style={'display':'inline-block', 'margin-top':20, 'margin-left':580, 'backgroundColor': colors['background'], 'color': colors['text']})
        ]),
        html.Hr(style={'color':'#fff'}),
        html.Div([
            html.H2('Sensor Data', style={'margin-bottom':'0'}),
            html.Div([
                html.Div(dcc.Graph(id='live-update-graph1', animate = True), style={'width': '85%', 'display': 'inline-block', 'margin-top':'0px'}),
                html.Div(html.Div(id='live-update-text1'), style={'width': '15%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-top':'100px'})
            ]),
            html.Div([
                html.Div(dcc.Graph(id='live-update-graph2', animate = True), style={'width': '85%', 'display': 'inline-block', 'margin-top':'0px'}),
                html.Div(html.Div(id='live-update-text2'), style={'width': '15%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-top':'100px'})
            ]),
            html.Div([
                html.Div(dcc.Graph(id='live-update-graph3'), style={'width': '85%', 'display': 'inline-block', 'margin-top':'0px'}),
                html.Div(html.Div(id='live-update-text3'), style={'width': '15%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-top':'100px'})
            ]),
        ]),
        html.Hr(style={'color':'#fff'}),
        html.Div([
            html.H2('Anamoly prediction', style={'margin-bottom':'0'}),
            html.Div([
                html.Div(dcc.Graph(id='live-update-graph4'), style={'width': '85%', 'display': 'inline-block', 'margin-top':'0px'}),
                html.Div(html.Div(id='live-update-text4'), style={'width': '15%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-top':'100px'})
            ]),
        ]),
        dcc.Interval(
            id='interval-component',
            # interval=0, # in milliseconds
            interval=1*1000, # in milliseconds
            n_intervals=-1,
            # max_intervals = 0
            max_intervals = len(raw_data)
        )
    ]
    )
)

time_stamp = deque()

def min_max_normalize(df):
    normalizad_df = df.copy()
    for col in normalizad_df.columns:
        for key in parameter_range.keys():
            if key in col:
                normalizad_df[col] = (normalizad_df[col] - parameter_range[key][0]) / (parameter_range[key][1] - parameter_range[key][0])
                break
    return normalizad_df

@app.callback(
    [Output('interval-component', 'max_intervals'),
    Output('interval-component', 'interval'),
    Output('submit-val', 'disabled')],
    Input('submit-val', 'n_clicks'),
    State('input-on-duration', 'value'),
    State('input-on-step', 'value')
)
def update_output(n_clicks, duration, step):
    if n_clicks > 0:
        return int(duration), int(step), True
    return int(duration), int(step), False

@app.callback(Output('live-update-text1', 'children'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    if n == -1:
        return []
    features = exh_gas_names
    data = raw_data.loc[n]
    style = {'padding': '2px', 'fontSize': '14px'}
    current_data = []
    for i in range(len(features)):
        current_data.append(html.Div('{}: {:.4f}'.format(features[i], data[features[i]]), style=style))
    return current_data

@app.callback(Output('live-update-text2', 'children'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    if n == -1:
        return []
    features = cyl_scav_air_names
    data = raw_data.loc[n]
    style = {'padding': '2px', 'fontSize': '14px'}
    current_data = []
    for i in range(len(features)):
        current_data.append(html.Div('{}: {:.4f}'.format(features[i], data[features[i]]), style=style))
    return current_data

@app.callback(Output('live-update-text3', 'children'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    if n == -1:
        return []
    features = load_rpm_names
    data = raw_data.loc[n]
    style = {'padding': '2px', 'fontSize': '14px'}
    current_data = []
    for i in range(len(features)):
        current_data.append(html.Div('{}: {:.4f}'.format(features[i], data[features[i]]), style=style))
    return current_data

@app.callback(Output('live-update-text4', 'children'),
              Input('interval-component', 'n_intervals'))
def update_metrics(n):
    if n == -1:
        return []

    features = ['lda', 'nn', 'ground_truth']
    ground_truth = int(raw_data.loc[n]['label'])
    data = raw_data.loc[n].drop('label', axis = 0) 
    data = pd.DataFrame.from_dict({k: [v] for k, v in zip(data.index, data.values)})
    data = min_max_normalize(data)
    data = data.values.reshape(1, -1)
    prediction_lda = inference_lda.predict(data)
    prediction_nn = inference_NN.predict(data)

    results = []
    results.append(prediction_lda[0])
    results.append(prediction_nn[0])
    results.append(ground_truth)
    style = {'padding': '2px', 'fontSize': '14px'}
    current_data = []
    for i in range(len(features)):
        current_data.append(html.Div('{}: {}'.format(features[i], results[i]), style=style))
    return current_data

# Multiple components can update everytime interval gets fired.
@app.callback(
              Output('live-update-graph1', 'figure'),
              Output('live-update-graph2', 'figure'),
              Output('live-update-graph3', 'figure'),
              Output('live-update-graph4', 'figure'),
              State('input-on-step', 'value'),
              Input('interval-component', 'n_intervals'),
              )
def update_graph_live(step, n):
    if n == -1:
        return go.Figure(), go.Figure(), go.Figure(), go.Figure()
    for i in range(len(exh_gas)):
        exh_gas[i].append(raw_data.loc[n][exh_gas_names[i]])
    for i in range(len(cyl_scav_air_names)):
        cyl_scav_air[i].append(raw_data.loc[n][cyl_scav_air_names[i]])
    for i in range(len(load_rpm_names)):
        load_rpm[i].append(raw_data.loc[n][load_rpm_names[i]])

    data = raw_data.loc[n].drop('label', axis = 0) 
    data = data.values.reshape(1, -1)
    prediction_lda = inference_lda.predict(data)
    prediction_nn = inference_NN.predict(data)
    ground_truth = int(raw_data.loc[n]['label'])

    results = []
    results.append(prediction_lda[0])
    results.append(prediction_nn[0])
    results.append(ground_truth)
    
    for i in range(len(model_names)):
        model[i].append(results[i])

    if not time_stamp:
        time_stamp.append(0)
    else:
        time_stamp.append(time_stamp[-1] + step)

    fig1 = go.Figure()
    fig2 = go.Figure()
    fig3 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4 = go.Figure()

    for i in range(len(exh_gas_names)):
        fig1.add_trace(go.Scatter(
            x= list(time_stamp),
            y= list(exh_gas[i]),
            name= exh_gas_names[i],
            mode= 'lines+markers'
        
        ))
    fig1.update_xaxes(title_text="time")
    fig1.update_yaxes(title_text="temperature(C)")

    for i in range(len(cyl_scav_air_names)):
        fig2.add_trace(go.Scatter(
            x= list(time_stamp),
            y= list(cyl_scav_air[i]),
            name= cyl_scav_air_names[i],
            mode= 'lines+markers'
        ))

    fig2.update_xaxes(title_text="time")
    fig2.update_yaxes(title_text="temperature(C)")

    fig3.add_trace(
        go.Scatter(x = list(time_stamp), y=list(load_rpm[0]), name=load_rpm_names[0], mode= 'lines+markers'),
        secondary_y=False,
    )
    fig3.add_trace(
        go.Scatter(x = list(time_stamp), y=list(load_rpm[1]), name=load_rpm_names[1], mode= 'lines+markers'),
        secondary_y=True,
    )

    fig3['layout']['xaxis']['title']='time'
    fig3['layout']['yaxis']['title']='load(%)'
    fig3['layout']['yaxis2']['title']='RPM(rpm)'

    for i in range(len(model_names)):
        fig4.add_trace(go.Scatter(
            x= list(time_stamp),
            y= list(model[i]),
            name= model_names[i],
            mode= 'lines+markers'
        ))

    fig4.update_xaxes(title_text="time")

    fig1.update_layout(xaxis=dict(range=[min(time_stamp),max(time_stamp)]))
    fig2.update_layout(xaxis=dict(range=[min(time_stamp),max(time_stamp)]))
    fig3.update_layout(xaxis=dict(range=[min(time_stamp),max(time_stamp)]))
    fig4.update_layout(xaxis=dict(range=[min(time_stamp),max(time_stamp)]),yaxis = dict(range = [0,1]))
    return fig1, fig2, fig3, fig4

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8080)