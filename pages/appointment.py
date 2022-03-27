import pandas as pd 
import sqlalchemy as db
import time
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback, dash_table
from flask_login import current_user
from dash.exceptions import PreventUpdate
from datetime import date, datetime

layout = ([
    dcc.Location(id='url-appt', refresh=True),
    html.Div(id='request-alert', style={'display':'none'}),
    html.Div(id='update-alert', style={'display':'none'}),
    dbc.Card(id='appointment-card', children=[
        html.Label('Appointment List', style={'font-size':'20px', 'background':'#e79070', 'color':'white', 
                                            'font-weight':'600', 'padding':'0.5rem'}),
        html.Div(id='appointment-list', children= [
            dash_table.DataTable(
                id='appointment-table',
                style_table={
                    'height':'70vh', 
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
                page_size=10,
                sort_action='native'
            ),
            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle('Accept Request')),
                dbc.ModalBody([
                    html.Div('Are you sure to accept this appointment request?')
                ]),
                dbc.ModalFooter([
                    dbc.Button('Yes', id='accept-button', n_clicks=0, n_clicks_timestamp=0),
                ])
            ], id='accept-request-modal', backdrop='static', keyboard='False'),

            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle('Reject Request')),
                dbc.ModalBody([
                    html.Div('Are you sure to reject this appointment request?\
                                If yes, please provide the reason.'),
                    dbc.Input(id='reject-remark', maxLength='50', placeholder='(optional)')
                ]),
                dbc.ModalFooter([
                    dbc.Button('Yes', id='reject-button', n_clicks=0, n_clicks_timestamp=0),
                ])
            ], id='reject-request-modal', backdrop='static', keyboard='False'),

            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle('Cancel Appointment')),
                dbc.ModalBody([
                    html.Div('Are you sure to cancel this appointment?\
                                If yes, please provide the reason.'),
                    dbc.Input(id='cancel-remark', maxLength='50', placeholder='(optional)')
                ]),
                dbc.ModalFooter([
                    dbc.Button('Yes', id='cancel-button', n_clicks=0, n_clicks_timestamp=0),
                ])
            ], id='cancel-appt-modal', backdrop='static', keyboard='False'),

            dbc.Modal([
                dbc.ModalHeader(dbc.ModalTitle('Appointment Completed')),
                dbc.ModalBody([
                    html.Div('Are you sure to mark this appointment as completed?'),
                ]),
                dbc.ModalFooter([
                    dbc.Button('Yes', id='complete-button', n_clicks=0, n_clicks_timestamp=0),
                ])
            ], id='complete-appt-modal', backdrop='static', keyboard='False')
        ])
    ], style={'textAlign':'center', 'margin-top':'1rem'}),
    html.Div([
        dbc.Button('Add Request', id='add-request', n_clicks_timestamp=0, style={'margin-top':'1rem'}),
        dbc.Modal([
            dbc.ModalHeader(dbc.ModalTitle('Request Appointment')),
            dbc.ModalBody([
                html.Div(id='modal-alert', style={'display':'none'}),
                html.Div([
                    dbc.Label('Patient'),
                    dcc.Dropdown(id='patient-appt')
                ], id='patient-div'),
                html.Div([
                    dbc.Label('Date'),
                    dcc.DatePickerSingle(
                        id='date',
                        min_date_allowed=datetime.now().strftime('%m-%d-%Y'),
                        max_date_allowed=date(2025, 1, 1),
                        initial_visible_month=datetime.now().strftime('%m-%d-%Y'),
                        placeholder='select a date',
                        className='form-control'
                    )
                ]),
                html.Br(),
                html.Div([
                    dbc.Label('Time'),
                    dbc.Input(
                        id='time',
                        type='time',
                        placeholder='select a time'
                    )
                ]),
                html.Br(),
                html.Div([
                    dbc.Label('Remark'),
                    dbc.Input(
                        id='remark',
                        maxLength='50',
                        placeholder='(optional)'
                    )
                ])
            ]),
            dbc.ModalFooter([
                dbc.Button('Confirm', id='confirm-request', n_clicks=0, n_clicks_timestamp=0)
            ])
        ], id='add-request-modal', backdrop='static', keyboard='False')
    ])
])

# update appointment table
@callback (
    Output('appointment-table', 'data'),
    Output('appointment-table', 'columns'),
    Input('appointment-table', 'data'),
    Input('request-alert', 'style'),
    Input('update-alert', 'style')
)

def update_appointment_table(data, request_style, update_style):
    from app import Appointment, engine

    if data == None or request_style != {'display':'none'} or update_style != {'display':'none'}:
        if current_user.role == 'doctor':
            df = pd.read_sql_query(
                sql = db.select([Appointment.id, Appointment.patientID, Appointment.request_from,
                                Appointment.date, Appointment.time, Appointment.remark, Appointment.status, 
                                Appointment.reject_reason]).where(Appointment.doctorID==current_user.userID),
                con = engine
            )

            df['action1'] = ''
            df['action2'] = ''

            for index, row in df.iterrows():
                id = row.id
                request_from = Appointment.query.filter_by(id=id).first().request_from
                if request_from == 'doctor':
                    if df.loc[index, 'status'] == 'pending':
                        df.loc[index, 'status'] = 'sent'
                if df.loc[index, 'status'] == 'pending':
                    df.loc[index, 'action1'] = 'accept'
                    df.loc[index, 'action2'] = 'reject'
                if df.loc[index, 'status'] == 'accepted':
                    df.loc[index, 'action1'] = 'cancel'
                    if current_user.role == 'doctor':
                        df.loc[index, 'action2'] = 'complete'
            
        else:
            df = pd.read_sql_query(
                sql = db.select([Appointment.id, Appointment.request_from, 
                                Appointment.date, Appointment.time, Appointment.remark, Appointment.status, 
                                Appointment.reject_reason]).where(Appointment.patientID==current_user.userID),
                con = engine
            )

            df['action1'] = ''
            df['action2'] = ''

            for index, row in df.iterrows():
                id = row.id
                request_from = Appointment.query.filter_by(id=id).first().request_from
                if request_from == 'patient':
                    if df.loc[index, 'status'] == 'pending':
                        df.loc[index, 'status'] = 'sent'
                if df.loc[index, 'status'] == 'pending':
                    df.loc[index, 'action1'] = 'accept'
                    df.loc[index, 'action2'] = 'reject'
                if df.loc[index, 'status'] == 'accepted':
                    df.loc[index, 'action1'] = 'cancel'
                    if current_user.role == 'doctor':
                        df.loc[index, 'action2'] = 'complete'

        data = df.to_dict('records')
        column = [{'name': i, 'id': i} for i in df.columns]
        return data, column
    raise PreventUpdate

# highlight table button
@callback(
    Output('appointment-table', 'style_data_conditional'),
    Input('appointment-table', 'derived_virtual_selected_row_ids')
)

def highlight(sel_row):
    val = [
        {
            'if': {
                'column_id': 'action1',
                'filter_query': '{action1} eq "accept"',
            },
            'backgroundColor': '#71cd50',
            'color': 'white'
        },
        {
            'if': {
                'column_id': 'action2',
                'filter_query': '{action2} eq "reject"',
            },
            'backgroundColor': '#cd5050',
            'color': 'white'
        },
        {
            'if': {
                'column_id': 'action1',
                'filter_query': '{action1} eq "cancel"',
            },
            'backgroundColor': '#cd5050',
            'color': 'white'
        },
        {
            'if': {
                'column_id': 'action2',
                'filter_query': '{action2} eq "complete"',
            },
            'backgroundColor': '#71cd50',
            'color': 'white'
        }
    ]   
    return val

# open modal for confirmation when action button clicked
@callback(
    Output('accept-request-modal', 'is_open'),
    Output('reject-request-modal', 'is_open'),
    Output('cancel-appt-modal', 'is_open'),
    Output('complete-appt-modal', 'is_open'),
    Output('update-alert', 'children'),
    Output('update-alert', 'style'),
    Input('appointment-table', 'active_cell'),
    Input('accept-button', 'n_clicks_timestamp'),
    Input('reject-button', 'n_clicks_timestamp'),
    Input('cancel-button', 'n_clicks_timestamp'),
    Input('complete-button', 'n_clicks_timestamp'),
    State('appointment-table', 'derived_viewport_data'),
    State('accept-request-modal', 'is_open'),
    State('reject-request-modal', 'is_open'),
    State('cancel-appt-modal', 'is_open'),
    State('complete-appt-modal', 'is_open'),
    State('reject-remark', 'value'),
    State('cancel-remark', 'value')
)

def show_confirmation_modal(active_cell, accept_clicks, reject_clicks, cancel_clicks, complete_clicks,
                            data, accept_open, reject_open, cancel_open, complete_open, reject_remark, 
                            cancel_remark):
    from app import appointment_table, engine
    conn = engine.connect()

    if active_cell:
        row = active_cell['row']
        col = active_cell['column_id']
        id = data[row]['id']

        if accept_open == True:
            update_appt = appointment_table.update().where(appointment_table.c.id==id).values(status='accepted')
            alert = dbc.Alert('The appointment request has been accepted.', color='success', duration=5000, 
                                style={'margin-top':'1rem'})
            conn.execute(update_appt)
            conn.close()
            return False, False, False, False, alert, {'display': 'block'}
        elif reject_open == True:
            update_appt = appointment_table.update().where(appointment_table.c.id==id).values(status='rejected')
            alert = dbc.Alert('The appointment request has been rejected.', color='danger', duration=5000,
                                style={'margin-top':'1rem'})
            if reject_remark != None and reject_remark != '':
                update_remark = appointment_table.update().where(appointment_table.c.id==id).values(reject_reason=reject_remark)
                conn.execute(update_remark)
            conn.execute(update_appt)
            conn.close()
            return False, False, False, False, alert, {'display': 'block'}
        elif cancel_open == True:
            update_appt = appointment_table.update().where(appointment_table.c.id==id).values(status='canceled')
            alert = dbc.Alert('The appointment has been cancelled.', color='danger', duration=5000,
                                style={'margin-top':'1rem'})
            if cancel_remark != None and cancel_remark != '':
                update_remark = appointment_table.update().where(appointment_table.c.id==id).values(reject_reason=cancel_remark)
                conn.execute(update_remark)
            conn.execute(update_appt)
            conn.close()
            return False, False, False, False, alert, {'display': 'block'}
        elif complete_open == True:
            update_appt = appointment_table.update().where(appointment_table.c.id==id).values(status='completed')
            alert = dbc.Alert('The appointment has been marked as completed.', color='success', duration=5000,
                                style={'margin-top':'1rem'})
            conn.execute(update_appt)
            conn.close()
            return False, False, False, False, alert, {'display': 'block'}
        elif col == 'action1': 
            action = data[row][col]
            if action == 'accept':
                return True, False, False, False, '', {'display':'none'}
            elif action == 'cancel':
                return False, False, True, False, '', {'display':'none'}
            else:
                raise PreventUpdate
        elif col == 'action2':
            action = data[row][col]
            if action == 'reject':
                return False, True, False, False, '', {'display':'none'}
            elif action == 'complete':
                return False, False, False, True, '', {'display':'none'}
            else:    
                raise PreventUpdate
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate

# update patient name dropdown
@callback(
    Output('patient-appt', 'options'),
    Input('patient-appt', 'options')
)

def update_dropdown(option):
    if current_user.role == 'doctor':
        from app import Patient
        patient = Patient.query.filter_by(doctorID=current_user.userID).all()
        dropdown = [
            {'label': i.fullname, 'value': i.patientID} for i in patient
        ]
        return dropdown
    raise PreventUpdate

# show modal & save request appointment into db
@callback(
    Output('add-request-modal', 'is_open'),
    Output('request-alert', 'children'),
    Output('request-alert', 'style'),
    Output('patient-div', 'style'),
    Input('add-request', 'n_clicks_timestamp'),
    Input('confirm-request', 'n_clicks_timestamp'),
    State('patient-appt', 'value'),
    State('date', 'date'),
    State('time', 'value'),
    State('remark', 'value'),
)

def show_modal(add_clicks, confirm_clicks, id, date, time, remark):
    if add_clicks > confirm_clicks:
        if current_user.role == 'doctor':
            return True, '', {'display':'none'}, {'display':'block'}
        else:
            return True, '', {'display':'none'}, {'display':'none'}
    elif confirm_clicks > add_clicks:
        from app import appointment_table, Patient, engine
        if date != None and time != None:
            date = datetime.strptime(date, '%Y-%m-%d')
            time = datetime.strptime(time, '%H:%M')
            if current_user.role == 'doctor':
                insert_appt = appointment_table.insert().values(patientID=id, doctorID=current_user.userID,
                                request_from='doctor', date=date.date(), time=time.time(), remark=remark, status='pending')
            else:
                doctorID = Patient.query.filter_by(patientID=current_user.userID).first().doctorID
                insert_appt = appointment_table.insert().values(patientID=current_user.userID, doctorID=doctorID,
                                request_from='patient', date=date.date(), time=time.time(), remark=remark, status='pending')
            conn = engine.connect()
            conn.execute(insert_appt)
            conn.close()

            alert = dbc.Alert('The appointment request has been sent successfully.', color='success', duration=5000,
                                style={'margin-top':'1rem'})
            return False, alert, {'display':'block', 'margin-top':'1rem'}, {'display':'none'}
        raise PreventUpdate
    raise PreventUpdate

# check request appointment form
@callback(
    Output('modal-alert', 'children'),
    Output('modal-alert', 'style'),
    Input('confirm-request', 'n_clicks'),
    State('patient-appt', 'value'),
    State('date', 'date'),
    State('time', 'value')
)

def required_field(n_clicks, name, date, time):
    if n_clicks > 0:
        # return name_border, date_border, True
        if name == None or date == None or time == None:
            alert = dbc.Alert('Please complete the form before proceeding.', 
                                color='warning', dismissable=True, style={'margin-top':'0.5rem'})
            alert_style = {'display':'block'}
        else:
            alert = ''
            alert_style = {'display':'none'}

        return alert, alert_style
    raise PreventUpdate

