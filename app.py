# config server
import os
import warnings
import re

# flask and dash dependencies
import flask
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output
from dash.exceptions import PreventUpdate
from pages import dashboard, appointment, patient, login, register, reset
from flask import flash

# manage db and users
import sqlite3
from sqlalchemy import Table, create_engine
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, logout_user, current_user, UserMixin
from flask_admin import Admin
from flask_admin.contrib.sqla import ModelView

# password hashing
from werkzeug.security import generate_password_hash

# ignore warnings
warnings.filterwarnings('ignore')

# connect to db
conn = sqlite3.connect('data.sqlite')
engine = create_engine('sqlite:///data.sqlite')
db = SQLAlchemy()

# class for role table
class Role(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    userID = db.Column(db.String(10), nullable=False)
    username = db.Column(db.String(15), nullable=False)
    role = db.Column(db.String(20), nullable=False)
    password = db.Column(db.String(80), nullable=False)

role_table = Table('role', Role.metadata)

# class for patient table
class Patient(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patientID = db.Column(db.String(10), unique=True, nullable=False)
    doctorID = db.Column(db.String(10), db.ForeignKey('doctor.doctorID'), nullable=False)
    doctor = db.relationship('Doctor', backref='doctor')
    fullname = db.Column(db.String(30), nullable=False)
    email = db.Column(db.String(80), unique=True, nullable=False)
    twitter = db.Column(db.String(30), default='')

patient_table = Table('patient', Patient.metadata)

# class for patient tweets
class Tweet(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patientID = db.Column(db.String(10), db.ForeignKey('patient.patientID'), nullable=False)
    patient = db.relationship('Patient', backref='patient')
    datetime = db.Column(db.DateTime, nullable=False)
    tweet = db.Column(db.String(200), nullable=False)
    symptom = db.Column(db.String(50), nullable=False)
    target = db.Column(db.String(10), nullable=False)

tweet_table = Table('tweet', Tweet.metadata)

# class for public_user table
class Public(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    fullname = db.Column(db.String(30), nullable=False)
    email = db.Column(db.String(80), unique=True, nullable=False)
    twitter = db.Column(db.String(30), unique=True, nullable=False)

public_table = Table('public', Public.metadata)

# class for doctor table
class Doctor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doctorID = db.Column(db.String(10), unique=True, nullable=False)
    fullname = db.Column(db.String(30), nullable=False)
    email = db.Column(db.String(80), unique=True, nullable=False)

doctor_table = Table('doctor', Doctor.metadata)

# class for appointment table
class Appointment(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    patientID = db.Column(db.String(10), db.ForeignKey('patient.patientID'), nullable=False)
    patient = db.relationship('Patient', backref='patient_appt')
    doctorID = db.Column(db.String(10), db.ForeignKey('doctor.doctorID'), nullable=False)
    doctor = db.relationship('Doctor', backref='doctor_appt')
    request_from = db.Column(db.String(10), nullable=False)
    date = db.Column(db.Date, nullable=False)
    time = db.Column(db.Time, nullable=False)
    remark = db.Column(db.String(50))
    status = db.Column(db.String(10), nullable=False)
    reject_reason = db.Column(db.String(50))

appointment_table = Table('appointment', Appointment.metadata)

# create tables
def create_table():
    Role.metadata.create_all(engine)
    Patient.metadata.create_all(engine)
    Tweet.metadata.create_all(engine)
    Public.metadata.create_all(engine)
    Doctor.metadata.create_all(engine)
    Appointment.metadata.create_all(engine)

# Patient.__table__.drop(engine)

# create the table
# create_table()

# add admin to role table
# admin = role_table.insert().values(userID='A01', username='admin01', role='admin', 
#                                     password=generate_password_hash('admin01', method='sha256'))
# conn = engine.connect()
# conn.execute(admin)
# conn.close()

# config server and app
server = flask.Flask(__name__)
server.config.update(
    # configuration the secret key to encrypt the user session cookie
    SECRET_KEY=os.getenv('SECRET_KEY'),
    # db file path
    SQLALCHEMY_DATABASE_URI='sqlite:///data.sqlite',
    # deactivate Flask-SQLAlchemy track modifications
    SQLALCHEMY_TRACK_MODIFICATIONS=False
)
app = Dash(__name__, server=server, url_base_pathname='/',
           external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
app.title = 'Depression Detection'

# initialiaze sqlite database
db.init_app(server) 

# admin accessibility
class MyModelView(ModelView):
    def is_accessible(self):
        if current_user.is_authenticated:
            if current_user.role == 'admin':
                return current_user.is_authenticated
            else:
                return not current_user.is_authenticated
        else:
            raise PreventUpdate 

    # password hashing on admin-flask update
    def on_model_change(self, form, model, is_created):
        if hasattr(model, 'password'):
            model.password = generate_password_hash(model.password, method='sha256')
        return super().on_model_change(form, model, is_created)

    def validate_form(self, form):
        # validate email format
        if hasattr(form, 'email'):
            if form.email.data != None:
                format = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                if re.fullmatch(format, form.email.data) == None:
                    flash('Invalid email format')
                    return False
        # validate if doctor ID in role table
        elif hasattr(form, 'userID'):
            if form.userID.data != None:
                id_doctor = Doctor.query.filter_by(doctorID=form.userID.data).first()
                id_role = Role.query.filter_by(userID=form.userID.data).first()
                if not id_doctor:
                    flash('Doctor ID not exists')
                    return False
                elif id_role:
                    flash('Doctor account already exists')
                    return False
        # validate if role == 'doctor'
        elif hasattr(form, 'role'):
            if form.userID.data != None and form.userID.data != 'doctor':
                flash('You can only create account for doctor')
                return False
        # validate fields in doctor table
        return super().validate_form(form)
        
admin = Admin(server)
admin.add_view(MyModelView(Role, db.session))
admin.add_view(MyModelView(Patient, db.session))
admin.add_view(MyModelView(Public, db.session))
admin.add_view(MyModelView(Doctor, db.session))
admin.add_view(MyModelView(Tweet, db.session))
admin.add_view(MyModelView(Appointment, db.session))

# setup LoginManager, allowing dash app to work with flaks-login
login_manager = LoginManager()
# configure for login
login_manager.init_app(server) 
# define the redirection path when login required
login_manager.login_view = '/login' 

CONTENT_STYLE = {
    'margin-left': '2rem',
    'margin-right': '2rem',
    'font-size': 'small'
}

navbar = dbc.NavbarSimple(
    id='navbar',
    children=[
        dbc.NavLink('Dashboard', href='/dashboard', active='exact'),
        dbc.NavLink('Appointment', href='/appointment', active='exact'),
        dbc.NavLink('Patient List', href='/patient-list', active='exact'),
        html.Span(style={'margin-left':'2rem'}),
        dbc.NavLink('', id='nav-username', active='disabled'),
        dbc.NavLink('Logout', href='/logout', active='exact')
    ],
    brand='Depression Detection',
    brand_href='#',
    color='#e79070',
    dark=True,
)

content = dbc.Container([html.Div(id='page-content', children=[], style=CONTENT_STYLE)], fluid=True, 
                        style={'height':'100vh', 'padding': '0px'})

app.layout = html.Div([dcc.Location(id='url', refresh=False), navbar, content], className='bg-light')

# reload user object
@login_manager.user_loader
def load_user(user_id):
    return Role.query.get(int(user_id))

@app.callback(Output('page-content', 'children'), [Input('url', 'pathname')])
def render_page_content(pathname):
    if pathname == '/' or pathname == '/login':
        return login.layout
    elif pathname == '/forgot-password':
        return reset.layout
    elif pathname == '/create':
        return register.layout
    elif pathname == '/dashboard':
        if current_user.is_authenticated and current_user.role != 'admin':
            return dashboard.layout
        else:
            return forbidden()
    elif pathname == '/appointment':
        if current_user.is_authenticated and (current_user.role == 'doctor' or current_user.role == 'patient'):
            return appointment.layout
        else:
            return forbidden()
    elif pathname == '/patient-list':
        if current_user.is_authenticated and current_user.role == 'doctor':
            return patient.layout
        else:
            return forbidden()
    elif pathname == '/logout':
        if current_user.is_authenticated:
            logout_user()
            return login.layout
        else:
            return login.layout
    # If the user tries to reach a different page, return a 404 message
    else:
        return dbc.Container(
            [
                html.H1('404: Not found', className='text-danger'),
                html.Hr(),
                html.P(f'The pathname {pathname} was not recognised...'),
            ]
        )

def forbidden():
    message = [
        dbc.Container(
        [
            html.H1('403: Forbidden', className='text-danger', style={'margin-top':'1rem'}),
            html.Hr(),
            html.P(f'You are not allowed to visit this page'),
        ])
    ]
    return message

@app.callback(
    Output('navbar', 'children'), 
    Output('page-content', 'style'), 
    Input('url', 'pathname')
)

def render_nav_content(pathname):
    if (pathname == '/' or pathname == '/login' or pathname == '/create' or pathname == '/logout'
        or pathname == '/forgot-password'):
        children = [
            dbc.NavLink('Sign Up', id='sign-up-nav', href='/create', active='exact'),
            dbc.NavLink('Login', id='login-nav', href='/login', active='exact')
        ]
        style = {
            'margin-left': '0',
            'margin-right': '0',
            'font-size': 'small'
        }
        return children, style
    else: 
        if current_user.is_authenticated:
            if current_user.role == 'patient':
                children = [
                        dbc.NavLink('Dashboard', href='/dashboard', active='exact'),
                        dbc.NavLink('Appointment', href='/appointment', active='exact'),
                        html.Span(style={'margin-left':'2rem'}),
                        dbc.NavLink(current_user.username, active='disabled'),
                        dbc.NavLink('Logout', href='/logout', active='exact')
                    ]
            elif current_user.role == 'public' or current_user.role == 'admin':
                children = [
                        dbc.NavLink(current_user.username, active='disabled'),
                        dbc.NavLink('Logout', href='/logout', active='exact')
                    ]   
            else:
                raise PreventUpdate
            return children, CONTENT_STYLE
        else:
            children = [
                dbc.NavLink('Sign Up', href='/create', active='exact'),
                dbc.NavLink('Login', href='/login', active='exact')
            ]
        return children, CONTENT_STYLE

@app.callback(
    Output('sign-up-nav', 'active'), 
    Output('login-nav', 'active'), 
    Input('url', 'pathname')
)

def nav_active(pathname):
    if pathname == '/login':
        return None, 'exact'
    elif pathname == '/create':
        return 'exact', None
    raise PreventUpdate

@app.callback(
    Output('nav-username', 'children'),
    Input('nav-username', 'children')
)

def update_username(username):
    if current_user.is_authenticated:
        if username == '':
            return current_user.username
        raise PreventUpdate
    raise PreventUpdate
    
if __name__ == '__main__':
    server.run(debug=True)