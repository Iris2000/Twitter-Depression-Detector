import re
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
from werkzeug.security import generate_password_hash

layout = dbc.Row([
            dbc.Col(id='acc-type', children=[
                dbc.Spinner(
                    html.Div([
                        html.H2('Choose Account Type', style={'color': '#7f4245', 'margin-top':'5rem',
                            'font-family':'Source Serif Pro', 'font-weight':'700'}),
                        html.Div(id='acc-alert', style={'display':'none'}),
                        html.H6('Hello there!', 
                                style={'margin-top':'2rem', 'font-family':'Source Serif Pro', 'color': '#adb5bd'}),
                        html.H6('Please choose your account type to get started',
                                style={'font-family':'Source Serif Pro', 'color': '#adb5bd'}),
                        html.Br(),
                        html.Button(id='patient', n_clicks_timestamp=0,
                                    children=[
                                        html.Img(src='assets/patient.png', width='100px'), 
                                        html.Br(),
                                        html.H6(html.Strong('Patient'))
                                    ], style={'className':'me-1', 'margin-right':'20px', 
                                                'color': '#7f4245','borderColor':'#7f4245'}),
                        html.Button(id='public', n_clicks_timestamp=0,
                                    children=[
                                        html.Img(src='assets/public_user.png', width='100px'), 
                                        html.Br(),
                                        html.H6(html.Strong('Public User'))
                                    ], style={'className':'me-1', 'color': '#7f4245','borderColor':'#7f4245'}),
                        html.Br(),
                        dbc.Button('Confirm', id='confirm', n_clicks=0,
                                    style={'margin-top':'2rem'}),
                        html.Br(),
                        html.Div(dcc.Link('I am already member', className='link', href='/login')
                        , style={'margin-top': '1rem'})  
                    ]), size='lg', color='#e79070', type='grow'
                )
            ], width=7, style={'text-align':'center', 'height':'100vh', 'padding-right':'0px'}),

            dbc.Col([
                html.Div([
                    html.Img(src='/assets/poster1.png', style={'height':'100vh', 'max-width':'100%'})
                ], style={'text-align':'right'})
            ], width=5, style={'padding':'0px'})
        ], style={'margin-right':'0px'})

@callback(
    Output('acc-type', 'children'), 
    Output('acc-alert', 'children'),
    Output('acc-alert', 'style'),
    Input('confirm', 'n_clicks'),
    State('patient', 'n_clicks_timestamp'),
    State('public', 'n_clicks_timestamp')
)

def sign_up(n_clicks, patient_ts, public_ts):
    if n_clicks > 0:
        if patient_ts > public_ts:
            patient_sign_up = [
                html.A('<', href='/create', className='previous round'),
                html.H2('Sign Up', style={'color': '#7f4245', 'margin-top':'4rem',
                    'font-family':'Source Serif Pro', 'font-weight':'700'}),
                html.H6('Hello there!', 
                        style={'margin-top':'2rem', 'font-family':'Source Serif Pro', 'color': '#adb5bd'}),
                html.H6('Please fill up the form below to get started',
                        style={'font-family':'Source Serif Pro', 'color': '#adb5bd'}),
                html.Br(),
                html.Div([
                    html.Div(id='register-alert', style={'display':'none'}),
                    html.Div([
                        dbc.Input(id='patientID', placeholder='Patient ID', type='text'),
                        dbc.FormFeedback('', id='id-check', type='invalid')
                    ]),
                    html.Div([
                        dbc.Input(id='username-pa', placeholder='Username', type='text'),
                        dbc.FormFeedback('Username already exists', type='invalid')
                    ]),
                    html.Div([
                        dbc.Input(id='twitter-pa', placeholder='Twitter Username', type='text'),
                        dbc.FormFeedback('Twitter username already exists', type='invalid')
                    ]),
                    html.Div([
                        dbc.Input(id='password-pa', placeholder='Password', type='password', minLength=8),
                        dbc.FormFeedback('Password must be at least 8 characters', type='invalid')
                    ]),
                    dbc.Button(id='register-pu', n_clicks=0, style={'display':'none'}),
                    dbc.Input(id='full-name', value=None, type='hidden'),
                    dbc.Input(id='username-pu', value=None, type='hidden'),
                    dbc.FormFeedback('', id='email-check', type='invalid'),
                    dbc.Input(id='email-pu', value=None, type='hidden'),
                    dbc.Input(id='twitter-pu', value=None, type='hidden'),
                    dbc.Input(id='password-pu', value=None, type='hidden'),
                    html.Br(),
                    dbc.Button('Register', id='register-pa', n_clicks=0, style={'margin-top':'2rem'})
                ], style={'width':'fit-content', 'display':'inline-block'}),
                html.Br(),
                html.Div([
                    dcc.Link('I am already member', className='link', href='/login')
                ], style={'margin-top': '1rem'}) 
            ]
            return patient_sign_up, None, None
        elif patient_ts < public_ts:
            public_sign_up = [
                html.A('<', href='/create', className='previous round'),
                html.H2('Sign Up', style={'color': '#7f4245', 'margin-top':'5rem',
                    'font-family':'Source Serif Pro', 'font-weight':'700'}),
                html.H6('Hello there!', 
                        style={'margin-top':'2rem', 'font-family':'Source Serif Pro', 'color': '#adb5bd'}),
                html.H6('Please fill up the form below to get started',
                        style={'font-family':'Source Serif Pro', 'color': '#adb5bd'}),
                html.Br(),
                html.Div([
                    html.Div(id='register-alert', style={'display':'none'}),
                    dbc.Input(id='full-name', placeholder='Full Name', type='text'),
                    html.Div([
                        dbc.Input(id='username-pu', placeholder='Username', type='text'),
                        dbc.FormFeedback('Username already exists', type='invalid')
                    ]),
                    html.Div([
                        dbc.Input(id='email-pu', placeholder='Email', type='email'),
                        dbc.FormFeedback('', id='email-check', type='invalid')
                    ]),
                    html.Div([
                        dbc.Input(id='twitter-pu', placeholder='Twitter Username', type='text'),
                        dbc.FormFeedback('Twitter username already exists', type='invalid')
                    ]),
                    html.Div([
                        dbc.Input(id='password-pu', placeholder='Password', type='password', minLength=8),
                        dbc.FormFeedback('Password must be at least 8 characters', type='invalid')
                    ]),
                    dbc.Button(id='register-pa', n_clicks=0, style={'display':'none'}),
                    dbc.Input(id='patientID', value=None, type='hidden'),
                    dbc.Input(id='username-pa', value=None, type='hidden'),
                    dbc.Input(id='twitter-pa', value=None, type='hidden'),
                    dbc.Input(id='password-pa', value=None, type='hidden'),
                ], style={'width':'fit-content', 'display':'inline-block'}),
                html.Br(),
                dbc.Button('Register', id='register-pu', n_clicks=0,
                            style={'margin-top':'2rem'}),
                html.Br(),
                html.Div(dcc.Link('I am already member', className='link', href='/login')
                , style={'margin-top': '1rem'}) 
            ]
            return public_sign_up, None, None
        else:
            account_type = [dbc.Spinner(
                    html.Div([
                        html.H2('Choose Account Type', style={'color': '#7f4245', 'margin-top':'5rem',
                            'font-family':'Source Serif Pro', 'font-weight':'700'}),
                        html.Div(id='acc-alert', style={'padding':'0 2rem'}),
                        html.H6('Hello there!', 
                                style={'margin-top':'1rem', 'font-family':'Source Serif Pro', 'color': '#adb5bd'}),
                        html.H6('Please choose your account type to get started',
                                style={'font-family':'Source Serif Pro', 'color': '#adb5bd'}),
                        html.Br(),
                        html.Button(id='patient', n_clicks_timestamp=0,
                                    children=[
                                        html.Img(src='assets/patient.png', width='100px'), 
                                        html.Br(),
                                        html.H6(html.Strong('Patient'))
                                    ], style={'className':'me-1', 'margin-right':'20px', 
                                                'color': '#7f4245','borderColor':'#7f4245'}),
                        html.Button(id='public', n_clicks_timestamp=0,
                                    children=[
                                        html.Img(src='assets/public_user.png', width='100px'), 
                                        html.Br(),
                                        html.H6(html.Strong('Public User'))
                                    ], style={'className':'me-1', 'color': '#7f4245','borderColor':'#7f4245'}),
                        html.Br(),
                        dbc.Button('Confirm', id='confirm', n_clicks=0,
                                    style={'margin-top':'2rem'}),
                        html.Br(),
                        html.Div(dcc.Link('I am already member', className='link', href='/login')
                        , style={'margin-top': '1rem'})  
                    ]), size='lg', color='#e79070', type='grow'
                )]
            alert = dbc.Alert('Please select an account type!', color='danger')
            return account_type, alert, {'display':'block', 'margin-left': '3rem'}
    else:
        raise PreventUpdate 

@callback(
    Output('patientID', 'required'),
    Output('username-pa', 'required'),
    Output('twitter-pa', 'required'),
    Output('password-pa', 'required'),
    Input('register-pa', 'n_clicks'),
)

def required_patient(n_clicks):
    if n_clicks > 0:
        return True, True, True, True
    else: 
        raise PreventUpdate

@callback(
    Output('patientID', 'valid'),
    Output('patientID', 'invalid'),
    Output('id-check', 'children'),
    Input('patientID', 'value')
)

def patientID_validity(patientID):
    from app import Patient, Role
    id = Patient.query.filter_by(patientID=patientID).first()
    check = Role.query.filter_by(userID=patientID).first()
    if patientID != None:
        if check:
            return False, True, 'Patient ID already registered'
        elif not id:
            return False, True, 'Patient ID does not exist'
        else:
            return True, False, ''
    else:
        return False, False, ''

@callback(
    Output('username-pa', 'valid'),
    Output('username-pa', 'invalid'),
    Input('username-pa', 'value')
)

def username_pa_validity(pa_username):
    from app import Role
    username = Role.query.filter_by(username=pa_username).first()
    if pa_username != None and pa_username != '':
        if username:
            return False, True
        else:
            return True, False
    else:
        return False, False

@callback(
    Output('twitter-pa', 'valid'),
    Output('twitter-pa', 'invalid'),
    Input('twitter-pa', 'value')
)

def twitter_pa_validity(pa_twitter):
    from app import Patient
    twitter = Patient.query.filter_by(twitter=pa_twitter).first()
    if pa_twitter != None and pa_twitter != '':
        if twitter:
            return False, True
        else:
            return True, False
    else:
        return False, False

@callback(
    Output('password-pa', 'valid'),
    Output('password-pa', 'invalid'),
    Input('password-pa', 'value'),
)

def password_pa_validity(pa_password):
    if pa_password != None and pa_password != '':
        if len(pa_password) >= 8:
            return True, False
        else:
            return False, True
    else:
        return False, False

@callback(
    Output('register-pa', 'value'),
    Input('patientID', 'valid'),
    Input('username-pa', 'valid'),
    Input('twitter-pa', 'valid'),
    Input('password-pa', 'valid')
)

def patient_register_validitiy(patientID, username, twitter, password):
    if patientID == True and username == True and twitter == True and password == True:
        return 'valid'
    else:
        raise PreventUpdate

@callback(
    Output('full-name', 'required'),
    Output('username-pu', 'required'),
    Output('email-pu', 'required'),
    Output('twitter-pu', 'required'),
    Output('password-pu', 'required'),
    Input('register-pu', 'n_clicks'),
)

def required_public(n_clicks):
    if n_clicks > 0:
        return True, True, True, True, True
    else: 
        raise PreventUpdate

@callback(
    Output('full-name', 'valid'),
    Output('full-name', 'invalid'),
    Input('full-name', 'value')
)

def full_name_validity(pu_name):
    if pu_name != None and pu_name != '':
            return True, False
    else:
        return False, False

@callback(
    Output('username-pu', 'valid'),
    Output('username-pu', 'invalid'),
    Input('username-pu', 'value')
)

def username_pu_validity(pu_username):
    from app import Role
    username = Role.query.filter_by(username=pu_username).first()
    if pu_username != None and pu_username != '':
        if username:
            return False, True
        else:
            return True, False
    else:
        return False, False

@callback(
    Output('email-pu', 'valid'),
    Output('email-pu', 'invalid'),
    Output('email-check', 'children'),
    Input('email-pu', 'value')
)

def email_pu_validity(pu_email):
    from app import Public
    email = Public.query.filter_by(email=pu_email).first()
    format = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    if pu_email != None and pu_email != '':
        if email:
            return False, True, 'Email already exists'
        elif re.fullmatch(format, pu_email) == None:
            return False, True, 'Invalid email format'
        elif re.fullmatch(format, pu_email):
            return True, False, ''
        else: 
            return False, False, ''
    else:
        return False, False, ''

@callback(
    Output('twitter-pu', 'valid'),
    Output('twitter-pu', 'invalid'),
    Input('twitter-pu', 'value')
)

def twitter_pu_validity(pu_twitter):
    from app import Public
    twitter = Public.query.filter_by(twitter=pu_twitter).first()
    if pu_twitter != None and pu_twitter != '':
        if twitter:
            return False, True
        else:
            return True, False
    else:
        return False, False

@callback(
    Output('password-pu', 'valid'),
    Output('password-pu', 'invalid'),
    Input('password-pu', 'value'),
)

def password_pu_validity(pu_password):
    if pu_password != None and pu_password != '':
        if len(pu_password) >= 8:
            return True, False
        else:
            return False, True
    else:
        return False, False

@callback(
    Output('register-pu', 'value'),
    Input('full-name', 'valid'),
    Input('username-pu', 'valid'),
    Input('email-pu', 'valid'),
    Input('twitter-pu', 'valid'),
    Input('password-pu', 'valid')
)

def public_register_validitiy(name, username, email, twitter, password):
    if name == True and username == True and email == True and twitter == True and password == True:
        return 'valid'
    else:
        raise PreventUpdate

@callback(
    Output('register-alert', 'children'),
    Output('register-alert', 'style'),
    Input('register-pa', 'n_clicks'),
    Input('register-pu', 'n_clicks'),
    State('patientID', 'value'),
    State('username-pa', 'value'),
    State('twitter-pa', 'value'),
    State('password-pa', 'value'),
    State('full-name', 'value'),
    State('username-pu', 'value'),
    State('email-pu', 'value'),
    State('twitter-pu', 'value'),
    State('password-pu', 'value'),
    State('register-pa', 'value'),
    State('register-pu', 'value')
)

def register_user(pa_clicks, pu_clicks, patientID, pa_username, pa_twitter, pa_password,
                     pu_name, pu_username, pu_email, pu_twitter, pu_password, pa_value, pu_value):
    from app import role_table, patient_table, public_table, Public, engine
    if pa_clicks > 0:
        if pa_value == 'valid':
            patientID = patientID
            username = pa_username
            twitter = pa_twitter
            password = generate_password_hash(pa_password, method='sha256')

            insert_role = role_table.insert().values(userID=patientID, username=username, role='patient',
                                               password=password)
            update_twitter = patient_table.update().where(patient_table.c.patientID==patientID).values(twitter=twitter)
            
            conn = engine.connect()
            conn.execute(insert_role)
            conn.execute(update_twitter)
            conn.close()

            alert = dbc.Alert('Registration Successful!', color='success')
            return alert, {'display':'block'}
        else:
            raise PreventUpdate
    elif pu_clicks > 0:
        if pu_value == 'valid':
            name = pu_name
            username = pu_username
            email = pu_email
            twitter = pu_twitter
            password = generate_password_hash(pu_password, method='sha256')

            insert_public = public_table.insert().values(fullname=name, email=email, twitter=twitter)

            conn = engine.connect()
            conn.execute(insert_public)

            publicID = Public.query.filter_by(email=email).first().id
            insert_role = role_table.insert().values(userID=publicID, username=username, role='public',
                                                    password=password)
            conn.execute(insert_role)
            conn.close()

            alert = dbc.Alert('Registration Successful!', color='success')
            return alert, {'display':'block'}
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate