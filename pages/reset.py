import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, State, callback
from dash.exceptions import PreventUpdate
from flask_login import login_user
# password hashing
from werkzeug.security import generate_password_hash

layout = html.Div([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Img(src='/assets/poster2.png', style={'height':'100vh', 'max-width':'100%'})
                    ], style={'text-align':'left'})
                ], width=5, style={'padding-right':'0px'}),

                dbc.Col([
                    html.Div([
                        html.H2('Password Reset', style={'color': '#7f4245', 'margin-top':'4rem',
                            'font-family':'Source Serif Pro', 'font-weight':'700'}),
                        html.Br(),
                        html.Div(id='reset-alert', style={'display':'none'}),
                        html.Div([
                            html.Div([
                                dbc.Input(id='username-reset', placeholder='Username', type='text'),
                            ]),
                            html.Div([
                                dbc.Input(id='password-reset', placeholder='Password', type='password', minLength=8),
                                dbc.FormFeedback('Password must be at least 8 characters', type='invalid')
                            ]),
                            dbc.Button('Reset', id='reset', n_clicks=0, style={'margin-top':'2rem'}),
                        ], style={'width':'fit-content', 'display':'inline-block', 'margin-top':'2rem'}),
                        html.Br(),
                        html.Div([
                            dcc.Link('Back to login', className='link', href='/login'),
                        ], style={'margin-top':'1rem', 'display':'inline-flex'}) 
                    ])
                ], width=7, style={'text-align':'center', 'height':'100vh'})
            ], style={'margin-right':'0px'})
        ])

@callback(
    Output('username-reset', 'required'),
    Output('password-reset', 'required'),
    Input('reset', 'n_clicks')
)

def required_reset(n_clicks):
    if n_clicks > 0:
        return True, True
    else: 
        raise PreventUpdate

@callback(
    Output('password-reset', 'valid'),
    Output('password-reset', 'invalid'),
    Input('password-reset', 'value'),
)

def password_validity(password):
    if password != None:
        if len(password) >= 8:
            return True, False
        else:
            return False, True
    else:
        return False, False

@callback(
    Output('reset-alert', 'children'),
    Output('reset-alert', 'style'),
    Input('reset', 'n_clicks'),
    State('username-reset', 'value'),
    State('password-reset', 'value'),
    State('password-reset', 'valid')
)

def password_reset(n_clicks, username, password, password_valid):
    from app import Role, role_table, engine
    user = Role.query.filter_by(username=username).first()
    if n_clicks > 0 and password_valid == True:
        if user:
            password = generate_password_hash(password, method='sha256')
            update_password = role_table.update().where(role_table.c.username==user.username).values(password=password)
            
            conn = engine.connect()
            conn.execute(update_password)
            conn.close()

            alert = dbc.Alert('Password reset successfully!', color='success', dismissable=True)
        else:
            alert = dbc.Alert('OOpps! User does not exist!', color='danger', dismissable=True)
        return alert, {'display':'block'}
    else:
        raise PreventUpdate