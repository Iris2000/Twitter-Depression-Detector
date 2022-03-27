import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
from flask_login import login_user
# password hashing
from werkzeug.security import check_password_hash

layout = html.Div([
            dcc.Location(id='url-login', refresh=True),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Img(src='/assets/poster2.png', style={'height':'100vh', 'max-width':'100%'})
                    ], style={'text-align':'left'})
                ], width=5, style={'padding-right':'0px'}),

                dbc.Col([
                    html.Div([
                        html.H2('Login', style={'color': '#7f4245', 'margin-top':'4rem',
                            'font-family':'Source Serif Pro', 'font-weight':'700'}),
                        html.Br(),
                        html.Div(id='login-alert', style={'display':'none'}),
                        html.Div([
                            dbc.Input(id='username', placeholder='Username', type='text'),
                            dbc.Input(id='password', placeholder='Password', type='password'),
                            dbc.Button('Login', id='login', n_clicks=0, style={'margin-top':'2rem'}),
                        ], style={'width':'fit-content', 'display':'inline-block', 'margin-top':'2rem'}),
                        html.Br(),
                        html.Div([
                            dcc.Link('Create an account', className='link', href='/create'),
                            html.Div(style={'width':'5rem'}),
                            dcc.Link('Forgot password', className='link', href='/forgot-password')
                        ], style={'margin-top':'1rem', 'display':'inline-flex'}) 
                    ])
                ], width=7, style={'text-align':'center', 'height':'100vh'})
            ], style={'margin-right':'0px'})
        ])

@callback(
    Output('username', 'required'),
    Output('password', 'required'),
    Input('login', 'n_clicks')
)

def required_login(n_clicks):
    if n_clicks > 0:
        return True, True
    else: 
        raise PreventUpdate

@callback(
    Output('url-login', 'pathname'),
    Output('login-alert', 'children'),
    Output('login-alert', 'style'),
    Input('login', 'n_clicks'),
    State('username', 'value'),
    State('password', 'value'),
    State('url', 'pathname')
)

def user_login(n_clicks, username, password, pathname):
    from app import Role
    user = Role.query.filter_by(username=username).first()
    if n_clicks > 0 and username != None and username != '' and password != None and password != '':
        if user:
            if check_password_hash(user.password, password):
                login_user(user)
                if user.role == 'admin':
                    return '/admin', None, None
                else:
                    return '/dashboard', None, None
            else:
                alert = dbc.Alert('OOpps! Wrong Password!', color='danger', dismissable=True)
                return None, alert, {'display':'block'}
        else:
            alert = dbc.Alert('OOpps! User does not exist!', color='danger', dismissable=True)
            return None, alert, {'display':'block'}
    elif pathname == '/logout':
        alert = dbc.Alert('Logout Successful!', color='success', dismissable=True)
        return None, alert, {'display':'block'}
    else:
        raise PreventUpdate