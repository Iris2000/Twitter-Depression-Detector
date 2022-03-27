import pandas as pd 
import sqlalchemy as db
import dash_bootstrap_components as dbc
from dash import html, Input, Output, State, callback, dash_table
from flask_login import current_user
from dash.exceptions import PreventUpdate
from sqlalchemy import desc

layout = ([
    dbc.Card(id='patient-card', children=[
        html.Label('Patient List', style={'font-size':'20px', 'background':'#e79070', 'color':'white', 
                                            'font-weight':'600', 'padding':'0.5rem'}),
        html.Div(id='patient-list', children= [
            dash_table.DataTable(
                id='patient-table',
                style_table={
                    'height':'80vh', 
                    'overflowX': 'auto', 
                    'overflowY': 'auto'
                },
                style_cell={
                    'fontSize': 15
                },
                style_header={
                    'font-family': 'Source Serif Pro',
                    'border-bottom': '1px solid #e79070',
                    'border-left': 'none',
                    'border-right': 'none',
                    'color': '#7f4245',
                    'font-weight': '600',
                    'text-align': 'center'
                },
                style_data={
                    'whiteSpace': 'normal',
                    'height': 'auto',
                    'font-family': 'Source Serif Pro',
                    'border-bottom': '1px solid #e79070',
                    'border-left': 'none',
                    'border-right': 'none',
                    'text-align': 'center'
                },
                page_size=10
            )
        ])
    ], style={'textAlign':'center', 'height':'80vh', 'margin-top':'1rem'})
])

# update patient table
@callback (
    Output('patient-table', 'data'),
    Output('patient-table', 'columns'),
    Input('patient-table', 'data')
)

def update_patient_table(data):
    from app import Patient, engine

    df = pd.read_sql_query(
        sql = db.select([Patient.patientID, Patient.fullname, Patient.email]).where(Patient.doctorID==current_user.userID),
        con = engine
    )

    df[''] = 'view'

    data = df.to_dict('records')
    column = [{'name': i, 'id': i} for i in df.columns]
    return data, column

# highlight button
@callback(
    Output('patient-table', 'style_data_conditional'),
    Input('patient-table', 'derived_virtual_selected_row_ids')
)

def highlight(sel_row):
    val = [
        {
            'if': {
                'column_id': '',
            },
            'backgroundColor': '#e17751',
            'color': 'white'
        }
    ]   
    return val

# direct to patient record when button clicked
@callback(
    Output('patient-card', 'children'),
    Input('patient-table', 'active_cell'),
    State('patient-table', 'derived_viewport_data')
)

def cell_clicked(active_cell, data):
    if active_cell:
        row = active_cell['row']
        col = active_cell['column_id']

        if col == '': 
            patient = data[row]['patientID']

            record = [
                html.A('<', href='/patient-list', className='previous round', 
                        style={'width':'2rem', 'margin-top':'1rem', 'margin-bottom':'1rem'}),
                html.Label(patient, id='patient-id', style={'font-size':'20px', 'background':'#e79070', 'color':'white', 
                                            'font-weight':'600', 'padding':'0.5rem'}),
                html.Div(id='tweet-list', children= [
                    dash_table.DataTable(
                        id='tweet-table',
                        style_table={'height':'80vh', 'overflowX': 'auto', 'overflowY': 'auto'},
                        style_cell={'fontSize':15},
                        style_header={
                            'font-family': 'Source Serif Pro',
                            'border-bottom': '1px solid #e79070',
                            'border-left': 'none',
                            'border-right': 'none',
                            'color': '#7f4245',
                            'font-weight': '600',
                            'text-align': 'center'
                        },
                        style_data={
                            'whiteSpace': 'normal',
                            'height': 'auto',
                            'font-family': 'Source Serif Pro',
                            'border-bottom': '1px solid #e79070',
                            'border-left': 'none',
                            'border-right': 'none',
                            'text-align': 'center'
                        },
                        page_size=10,
                        sort_action='native'
                    )
                ])
            ]
            return record
        else:
            raise PreventUpdate
    raise PreventUpdate

# update tweet table
@callback (
    Output('tweet-table', 'data'),
    Output('tweet-table', 'columns'),
    Input('tweet-table', 'data'),
    State('patient-id', 'children')
)

def update_tweet_table(data, id):
    from app import Tweet, engine
    df = pd.read_sql_query(
        sql = db.select([Tweet.datetime, Tweet.tweet, Tweet.symptom, 
                        Tweet.target]).where(Tweet.patientID==id).order_by(desc(Tweet.datetime)),
        con = engine
    )

    data = df.to_dict('records')
    column = [{'name': i, 'id': i} for i in df.columns]
    return data, column

# highlight table rows
@callback(
    Output('tweet-table', 'style_data_conditional'),
    Input('tweet-table', 'derived_virtual_selected_row_ids')
)

def highlight(sel_row):
    val = [
        {
            'if': {
                'column_id': 'target',
                'filter_query': '{target} eq "depressed"',
            },
            'backgroundColor': '#FF4136',
            'color': 'white'
        }
    ]   
    return val