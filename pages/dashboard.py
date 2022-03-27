# import packages
from flask import session
import pandas as pd
import re 
import unidecode
import string
import itertools
import contractions
import liwc
import gensim
import pickle
import numpy as np
import nltk
import dash_bootstrap_components as dbc
import plotly.express as px
import twint
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nrclex import NRCLex
from collections import Counter
from keras.models import model_from_json
from sentence_transformers import SentenceTransformer
from dash import html, dcc, Input, Output, State, callback, dash_table, no_update
from flask_login import current_user
from datetime import date, datetime
from dash.exceptions import PreventUpdate
from io import BytesIO
from wordcloud import WordCloud
import base64

layout = dbc.Spinner(
    html.Div([
        # session storage
        dcc.Store(id='session', storage_type='session'),

        html.Div(id='select-alert', style={'display':'none'}),

        html.Div(id='alert', style={'display':'none'}),

        dbc.Row(html.Div(children=[
                dbc.Col([
                    html.Div(id='selection', children=[
                        html.Label('Patient\'s Name'),
                        dcc.Dropdown(id='patient-name'),
                        html.Label('Date Range', style={'margin-top':'5px'}),
                        html.Div([
                            dcc.DatePickerSingle(
                                id='start-date',
                                min_date_allowed=date(2015, 1, 1),
                                max_date_allowed=datetime.now().strftime('%m-%d-%Y'),
                                initial_visible_month=date(2022, 1, 1),
                                placeholder='start date',
                            ),
                            dcc.DatePickerSingle(
                                id='end-date',
                                min_date_allowed=date(2015, 1, 1),
                                max_date_allowed=datetime.now().strftime('%m-%d-%Y'),
                                initial_visible_month=date(2022, 1, 1),
                                placeholder='end date',
                            ),
                        ])
                    ], style={'padding': 10, 'flex': 1})
                ]),

                dbc.Col([
                    html.Div(children=[
                        dbc.Button('Analyze', color='primary', id='analyze', n_clicks=0, value=0,
                                    style={'font-size': 'small'})
                    ], style={'padding': 10, 'flex': 1}),
                ], align='end'),

                dbc.Col([
                    dbc.Card([
                        dbc.CardBody(children=[
                            html.Label('Depression Score', style={'color':'white'}),
                            html.H2(id='score', style={'color':'white'}),
                        ], style={'padding': 10, 'flex': 1, 'textAlign': 'center'})
                    ], color='#e79070')
                ], align='center')
        ], style={'display': 'flex', 'flex-direction': 'row'})),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.Label('Twitter Posts'),
                    html.Div(id='table-container', children= [
                        dash_table.DataTable(
                            id='table',
                            style_table={'height':'65vh', 'overflowX': 'auto', 'overflowY': 'auto'},
                            style_cell={'fontSize':12},
                            style_data={
                                'whiteSpace': 'normal',
                                'height': 'auto'
                            },
                            page_size=10
                        )
                    ], style={'display':'none'})
                ], style={'textAlign':'center', 'height':'73vh'})
            ], width=5),

            html.Br(),
            dbc.Col([
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            html.Label('Sentiment Over Time'),
                            html.Div(id='graph-container', children= [
                                dcc.Graph(id='line-chart', config={'displayModeBar': False})
                            ], style={'display':'none'})
                        ], style={'textAlign':'center', 'height':'35vh'})
                    ])
                ]),

                html.Br(),
                dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            html.Label('Positive Word Cloud'),
                            html.Img(id='pos-wordcloud', style={'height':'34vh'}, src='')
                        ], style={'textAlign':'center', 'height':'35vh'})
                    ], width=6),
                    dbc.Col([
                        dbc.Card([
                            html.Label('Negative Word Cloud'),
                            html.Img(id='neg-wordcloud', style={'height':'34vh'}, src='')
                        ], style={'textAlign':'center', 'height':'35vh'})
                    ], width=6)
                ])
            ], width=7)
        ], style={'margin-top':'5px'})
    ])
, size='lg', color='#e79070', type='grow')

# clear the session data
@callback(
    Output('session', 'clear_data'),
    Input('analyze', 'n_clicks'),
    State('url', 'pathname')
)

def clear_session(n_clicks, pathname):
    if n_clicks > 0 and pathname == '/dashboard':
        return True
    raise PreventUpdate

# validate form and number of tweets
@callback(
    Output('select-alert', 'children'),
    Output('select-alert', 'style'),
    Input('analyze', 'n_clicks'),
    State('patient-name', 'value'),
    State('start-date', 'date'),
    State('end-date', 'date'),
    State('analyze', 'value')
)

def required_field(n_clicks, name, start, end, value):
    if n_clicks > 0:
        if name == None or start == None or end == None:
            alert = dbc.Alert('Please complete the form before proceeding.', 
                                color='warning', dismissable=True, style={'margin-top':'1rem'})
            alert_style = {'display':'block'}
        elif end < start:
            alert = dbc.Alert('Ending date should not be earlier than starting date.', 
                    color='warning', dismissable=True, style={'margin-top':'1rem'})
            alert_style = {'display':'block'}
        elif value == 1:
            alert = dbc.Alert('No tweets available. Please select different date.', 
                    color='warning', dismissable=True, style={'margin-top':'1rem'})
            alert_style = {'display':'block'}
        else:
            alert = ''
            alert_style = {'display':'none'}
        return alert, alert_style
    raise PreventUpdate

@callback(
    Output('table-container', 'style'),
    Output('table', 'data'),
    Output('table', 'columns'),
    Output('score', 'children'),
    Output('graph-container', 'style'),
    Output('line-chart', 'figure'),
    Output('pos-wordcloud', 'src'),
    Output('neg-wordcloud', 'src'),
    Output('alert', 'children'),
    Output('alert', 'style'),
    Output('session', 'data'),
    Output('analyze', 'value'),
    Output('select-alert', 'is_open'),
    Input('analyze', 'n_clicks'),
    State('patient-name', 'value'),
    State('start-date', 'date'),
    State('end-date', 'date'),
    State('session', 'data')
)

def update_output(n_clicks, name, start_date, end_date, session_data):
    if session_data is not None:
        if current_user.username != session_data['username']:
            session_data.clear()
            raise PreventUpdate
        else:         
            return (session_data['table_style'], session_data['table_data'], session_data['table_columns'], 
                    session_data['score'], session_data['graph_style'], session_data['graph_figure'], 
                    session_data['pos_word'], session_data['neg_word'], session_data['alert'], 
                    session_data['alert_style'], session_data, 0, no_update)
    elif n_clicks > 0:
        if name != None and start_date != None and end_date != None:
            # scrape tweets
            df = scrape(name, start_date, end_date)
            # check if df is empty
            if df.empty:
                return (no_update, no_update, no_update, no_update, no_update, no_update, 
                        no_update, no_update, no_update, no_update, no_update, 1, no_update)
                # raise PreventUpdate
            # import csv
            df = extract(df)
            df = duplicate(df)
            df = decode(df)
            tweet = tweet_list(df)
            tweet = clean(tweet)
            tweet = emojis(tweet)
            tweet = lowercase(tweet)
            tweet = contracts(tweet)
            tweet = punctuations(tweet)
            tweet = lengthening(tweet)
            tweet = jargons(tweet)
            tweet = numbers(tweet)
            tweet = replace(tweet)
            tweet = whitespace(tweet)
            # extract pos tag
            pos_tagged = [nltk.pos_tag(nltk.word_tokenize(x)) for x in tweet]
            # replace the original pos tag with simpler naming
            wordnet_tagged = [list(map(lambda x: (x[0], pos_tagger(x[1])), x)) for x in pos_tagged]
            tweet = lemmatize(wordnet_tagged)
            tweet = remove_stopwords(tweet)
            tweet = lowercase(tweet)
            df = combine(tweet, df)
            df = less_words(df)
            df = vader(df)
            df = nrclex(df)
            df = count_liwc(df)
            gsdmm, docs_cleaned = train_topic(df)
            # number of documents per topic
            doc_count = np.array(gsdmm.cluster_doc_count)
            # topics sorted by the number of document they are allocated to
            top_topic = doc_count.argsort()[-10:][::-1]
            df = extract_topic(df, gsdmm, docs_cleaned, top_topic)
            predicted_df = predict(df)
            df = predicted_df[['datetime', 'tweet', 'symptoms', 'target']]

            # save tweets to db
            save_tweets(name, df)

            # update table
            data = df.to_dict('records')
            columns = [{'name': i, 'id': i} for i in df.columns]

            # update score
            score = len(df[df['target']=='depressed']) / len(df) * 100
            score = round(score, 2)

            # update sentiment over time
            fig_line = px.line(predicted_df, x='datetime', y='tweet_compound', height=190)

            # update positive wordcloud
            pos_wordcloud = update_wordcloud(predicted_df, 'pos')

            # update negative wordcloud
            neg_wordcloud = update_wordcloud(predicted_df, 'neg')

            # update alert
            alert = update_alert(score)
            alert_style = {'display':'block', 'margin-top':'5px'}

            # update session_data
            session_data = {}
            session_data['username'] = current_user.username
            session_data['name'] = name
            session_data['start_date'] = start_date
            session_data['end_date'] = end_date
            session_data['score'] = str(score)+'%'
            session_data['table_style'] = {'display':'block'}
            session_data['table_data'] = data
            session_data['table_columns'] = columns
            session_data['graph_style'] = {'display':'block'}
            session_data['graph_figure'] = fig_line
            session_data['pos_word'] = pos_wordcloud
            session_data['neg_word'] = neg_wordcloud
            session_data['alert'] = alert
            session_data['alert_style'] = alert_style

            return ({'display':'block'}, data, columns, str(score)+'%', {'display':'block'}, fig_line, 
                    pos_wordcloud, neg_wordcloud, alert, alert_style, session_data, 0, False)
        else:
            raise PreventUpdate
    else:
        raise PreventUpdate   

# analytics process after click 'predict' button
# scrape tweets
def scrape(name, start_date, end_date):
    # configuration
    config = twint.Config()
    config.Username = name
    config.Since = start_date
    config.Until = end_date
    config.Limit = 100
    config.Pandas = True
    config.Lang = 'en'
    # running search
    twint.run.Search(config)
    df = twint.output.panda.Tweets_df
    # rename date column
    df.rename(columns={'date': 'datetime'}, inplace=True)
    return df

# extract required columns
def extract(df):
    df = df[['username', 'datetime', 'tweet']]
    return df

# remove duplicate tweets
def duplicate(df):
    # drop duplicate tweets
    df = df.drop_duplicates(subset=['tweet'])
    df = df.reset_index(drop=True)
    return df

# remove non-english character
def decode(df):
    decode_data = []
    encoded_data = [x.encode('ascii', 'ignore') for x in df['tweet']]
    for data in encoded_data:
        decode_data.append(data.decode())
    df['tweet'] = decode_data
    return df

# create a tweet list from dataframe for cleaning
def tweet_list(df):
    tweet = df['tweet'].iloc[:]
    return tweet

# text cleaning
def clean(tweet):
    # remove links
    tweet = [re.sub(r'https?://\S+|www\.\S+', '', x) for x in tweet]
    # remove emails
    tweet = [re.sub(r'\S*@\S*\s?', '', x) for x in tweet]
    # remove mentions
    tweet = [re.sub(r'@', '', x) for x in tweet]
    # remove hashtags
    tweet = [re.sub(r'#[A-Za-z0-9_]+', '', x) for x in tweet]
    # remove unicode, change quotation marks, ™ to (tm), and so on
    tweet = [unidecode.unidecode(x) for x in tweet]
    # remove specific character / word
    tweet = [re.sub(r'&amp;', '', x) for x in tweet]
    tweet = [re.sub(r'"', '', x) for x in tweet]
    tweet = [re.sub(r'\(tm\)','', x) for x in tweet]
    tweet = [re.sub(r'\(c\)','', x) for x in tweet]
    return tweet

# convert emojis to text
def emojis(tweet):
    # encode html entities - some of them are emojis
    tweet = [re.sub("&gt;",">", x) for x in tweet]
    tweet = [re.sub("&lt;","<", x) for x in tweet]
    # emojis to text
    tweet = [re.sub(";\)","wink", x) for x in tweet]
    tweet = [re.sub(";-\)","wink", x) for x in tweet]
    tweet = [re.sub(";D","wink", x) for x in tweet]
    tweet = [re.sub(":\)","smiley", x) for x in tweet]
    tweet = [re.sub(":>","smiley", x) for x in tweet]
    tweet = [re.sub(":-\)","smiley", x) for x in tweet]
    tweet = [re.sub(":^\)","smiley", x) for x in tweet]
    tweet = [re.sub(":]","smiley", x) for x in tweet]
    tweet = [re.sub("=\)","smiley", x) for x in tweet]
    tweet = [re.sub(":'\)","happy", x) for x in tweet]
    tweet = [re.sub(">v<","happy", x) for x in tweet]
    tweet = [re.sub(":D","grinning", x) for x in tweet]
    tweet = [re.sub("\^\^","joy", x) for x in tweet]
    tweet = [re.sub("\^__\^","joy", x) for x in tweet]
    tweet = [re.sub("xD","laughing", x) for x in tweet]
    tweet = [re.sub("XD","laughing", x) for x in tweet]
    tweet = [re.sub(":p","cheeky", x) for x in tweet]
    tweet = [re.sub(":/","hesitant", x) for x in tweet]
    tweet = [re.sub("<3","love", x) for x in tweet]
    tweet = [re.sub(">_>","devious", x) for x in tweet]
    tweet = [re.sub(">.>","devious", x) for x in tweet]
    tweet = [re.sub(":\(","sad", x) for x in tweet]
    tweet = [re.sub("T__T","crying", x) for x in tweet]
    tweet = [re.sub(";-;","crying", x) for x in tweet]
    tweet = [re.sub("_\(:3","tired", x) for x in tweet]
    return tweet

# convert text to lowercase        
def lowercase(tweet):
    tweet = [x.lower() for x in tweet]
    return tweet

# expand the contractions
def contracts(tweet):
    tweet = [contractions.fix(x, slang=True) for x in tweet]
    return tweet

# remove punctuations
def punctuations(tweet):
    for character in string.punctuation:
        tweet = [x.replace(character, ' ') for x in tweet]
    return tweet

# fix word lengthening
def lengthening(tweet):
    tweet = [''.join(''.join(s)[:2] for _, s in itertools.groupby(x)) for x in tweet]
    return tweet

# expand jargons
def jargons(tweet):
    tweet = [re.sub(r'\bppl\b','people', x) for x in tweet]
    tweet = [re.sub(r'\brn\b','right now', x) for x in tweet]
    tweet = [re.sub(r'\bw\b','with', x) for x in tweet]
    tweet = [re.sub(r'\br\b','are', x) for x in tweet]

    jargons = {
    'asap': 'as fast as possible',
    'brb': 'be right back',
    'lol': 'laugh',
    'lolz': 'laugh',
    'lmao': 'laugh',
    'lmaoo': 'laugh',
    'tbh': 'to be honest',
    'abt': 'about',
    'smh': 'somehow',
    'btw': 'by the way',
    'gatchu': 'got you',
    'atm': 'at the moment',
    'txt': 'text',
    'yrs': 'years',
    'fyi': 'for your information',
    'gna': 'gonna',
    'gn': 'good night',
    'yt': 'youtube',
    'vid': 'video',
    'vids': 'videos',
    'pic': 'picture',
    'thru': 'through',
    'tho': 'though',
    'imma': 'i am going to',
    'bff': 'best friend forever',
    'dm': 'direct message',
    'cmon': 'come on',
    'bcz': 'because',
    'bc': 'because',
    'rly': 'really',
    'idk': 'i do not know',
    'ikr': 'i know right',
    'plz': 'please',
    'pls': 'please',
    'bro': 'brother',
    'bruh': 'brother',
    'dunno': 'do not know'
    }

    for i in range(len(tweet)):
        for word in tweet[i].split():
            if word in jargons:
                tweet[i] = tweet[i].replace(word, jargons[word])
    return tweet

# remove word with numbers -> such as datetime
def numbers(tweet):
    tweet = [re.sub(r'\w+\d\w+', '', x) for x in tweet]
    tweet = [re.sub(r'\d+', '', x) for x in tweet]
    return tweet

# change 's -> is, delete s as some become s only
def replace(tweet):
    tweet = [re.sub(r'\bs\b','', x) for x in tweet]
    tweet = [re.sub(r'\bid\b','i had', x) for x in tweet]
    return tweet

# remove whitespace
def whitespace(tweet):
    tweet = [re.sub(' +',' ', x) for x in tweet]
    tweet = [x.rstrip().lstrip() for x in tweet]
    return tweet

# WordNetLemmatizer with appropriate pos tags
def pos_tagger(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:         
        return None

# lemmatize tweets with the pos tag created
def lemmatize(wordnet_tagged):
    lemmatizer = WordNetLemmatizer()
    lemmatized_tweet = []

    for i in range (len(wordnet_tagged)):
        lemmatized_sentence = []
        for word, tag in wordnet_tagged[i]:
            if tag is None:
                # if there is no available tag, append the token without lemmatize it
                lemmatized_sentence.append(word)
            else:       
                # use the tag to lemmatize the word
                lemmatized_sentence.append(lemmatizer.lemmatize(word, tag))
        lemmatized_join = ' '.join(lemmatized_sentence)
        lemmatized_tweet.append(lemmatized_join)
    return lemmatized_tweet

# remove stopwords
def remove_stopwords(tweet):
    # reserve 1st 2nd 3rd person pronoun and "no"
    stop_words = stopwords.words('english')
    stop_words = ['what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are','was', 'were', 
                  'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 
                  'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 
                  'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 
                  'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 
                  'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
                  'other', 'some', 'such', 'nor', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 
                  'will', 'just', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y',
                 ]

    text_tokens = [word_tokenize(x) for x in tweet]
    
    combine_without_sw = []
    combine_text = []
    
    for text in text_tokens:
        text_without_sw = [word for word in text if word not in stop_words]
        combine_without_sw.append(text_without_sw)
    
    for word in combine_without_sw:
        combine_join = ' '.join(word)
        combine_text.append(combine_join)
    return combine_text

# combine tweet and df
def combine(tweet, df):
    # convert tweet list to dataframe
    cleaned_tweet = pd.DataFrame(tweet, columns=['cleaned_tweet'])
    # concat df with cleaned_tweet
    df = pd.concat([df, cleaned_tweet], axis=1)
    # reordering the columns
    df = df[['username', 'datetime', 'tweet', 'cleaned_tweet']]
    return df

# remove tweets with <= 1 words 
def less_words(df):
    # check the tweets with <= words
    count = 0
    delete = []

    for i in range (len(df)):
        if len(df['cleaned_tweet'][i].split()) <= 1:
            print(df['cleaned_tweet'][i])
            delete.append(i)
            count += 1

    # remove empty elements
    df = df.drop(df.index[delete])
    # reset index
    df = df.reset_index(drop=True)
    return df

# get sentiment polarity score with VADER
def vader(df):
    analyzer = SentimentIntensityAnalyzer()
    df['tweet_compound'] = df['tweet'].apply(lambda x:analyzer.polarity_scores(x)['compound'])
    return df

# get emotion lexicon with nrclex
def nrclex(df):
    # all emotions affect frequencies in one column
    df['emotions'] = df['cleaned_tweet'].apply(lambda x: NRCLex(x).affect_frequencies)
    # split emotions into respective columns
    df = pd.concat([df.drop(['emotions'], axis=1), df['emotions'].apply(pd.Series)], axis=1)
    # remove extra column
    df = df.drop('anticip', 1)
    # replace NaN with 0
    df['anticipation'] = df['anticipation'].fillna(0)
    return df

# get liwc with custom dictionary
def count_liwc(df):
    # load the liwc dictionary
    liwcPath = 'C://Users//lvlip//Documents//BCSI Sem 6//FYP 4202//CSV//liwc_dic.dic'
    # parse the dictionary
    parse, category_names = liwc.load_token_parser(liwcPath)
    # tokenize the tweets
    text_tokens = [word_tokenize(x) for x in df['cleaned_tweet']]
    
    # count the frequency of each category
    count = 0
    combine_liwc = []

    for text in text_tokens:
        count += 1
        gettysburg_counts = Counter(category for word in text for category in parse(word))
        combine_liwc.append(gettysburg_counts)
    
    # extract counter dictionary - previous result Counter({}), extract the {}
    combine_dic = []

    for i in range (len(combine_liwc)):
        dic = {}
        for k, v in combine_liwc[i].items():
            dic[k] = v
        combine_dic.append(dic)
        
    # convert dictionary into dataframe
    liwc_df = pd.DataFrame(combine_dic)
    
    # replace NaN with 0
    liwc_df = liwc_df.fillna(0)
    
    # concat df with liwc
    df = pd.concat([df, liwc_df], axis=1)

    # all liwc column
    columns = ['second/thirdperson', 'firstperson', 'sadness/loneliness', 'suicidal', 'primarysupport',
              'troubleconcentrate', 'worthlessness/guilt', 'housing', 'disturbedsleep', 'occupational',
              'fatigue/lossenergy', 'weight/appetite', 'agitation/retardation']
    
    # if no symptoms detected, assign 0 to the column
    for column in columns:
        if column not in df:
            df[column] = 0

    # summarize symptoms detected for each tweet
    # extract liwc columns
    liwc_df = df[['sadness/loneliness', 'suicidal', 'primarysupport', 'troubleconcentrate',
                  'worthlessness/guilt', 'housing', 'disturbedsleep', 'occupational', 
                  'fatigue/lossenergy', 'weight/appetite', 'agitation/retardation']]

    # extract symptoms from eact tweet into list 
    symptoms = liwc_df.apply(lambda x: x.index[x != 0.0].to_list(), axis=1).to_list()

    # replace empty list with 'None'
    for i in range (len(symptoms)):
        if not symptoms[i]:
            symptoms[i] = 'None'

    # combine symptoms with df
    df['symptoms'] = symptoms

    return df

# topic modeling
def train_topic(df):
    # tokenize tweets into words
    docs = [word_tokenize(x) for x in df['cleaned_tweet']]
    
    # remove stopwords from docs
    docs_cleaned = []
    stop_words = gensim.parsing.preprocessing.STOPWORDS
    
    for doc in docs:
        temp = []
        for word in doc:
            if word not in stop_words:
                temp.append(word)
        docs_cleaned.append(temp)

    # create dictionary of all words in docs_cleaned
    dictionary = gensim.corpora.Dictionary(docs_cleaned)
    
    # find the length of dictionary
    dic_length = len(dictionary)
    
    # initialize GSDMM
    with open('gsdmm.pkl', 'rb') as topic:
        gsdmm = pickle.load(topic)

    # fit GSDMM model
    gsdmm.fit(docs_cleaned, dic_length)
    return gsdmm, docs_cleaned

# extract topic for each tweet
def extract_topic(df, gsdmm, docs_cleaned, top_topic):
    topic = []
    prob = [gsdmm.choose_best_label(x) for x in docs_cleaned]

    threshold = 0.3
    
    # confirm the topic only when the prob is >= 0.3, else assign 10 as other topic
    for data in prob:
        if data[1] >= threshold:
            topic.append(top_topic[data[0]])
        else:
            topic.append(10)
        
    # add the topic column to df
    df['topic'] = topic
    return df

# prediction
def predict(predict_df):
    # build bert model with distilbert - ligther version of bert
    bert_model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    # create embeddings with bert - with 'cleaned_tweet' only
    embeddings = bert_model.encode(predict_df['cleaned_tweet'])
    # combine embeddings and other features
    embeddings2 = np.hstack((embeddings, 
                             predict_df[predict_df.columns.difference(['username', 'datetime',
                                                                       'tweet', 'cleaned_tweet',
                                                                       'symptoms'])]))
    # define row and column number
    row_num = embeddings2.shape[0]
    col_num = embeddings2.shape[1]
    # reshape to 3 dimension for prediction
    predict_data = embeddings2.reshape(row_num, col_num, 1)
    # loading the model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # loading the best model
    model.load_weights('neural_network2.hdf5')
    # predict the target
    score = model.predict(predict_data)
    # encode score into target
    label = list(map(lambda x: 'normal' if x<0.5 else 'depressed', score))
    label_df = pd.DataFrame(label, columns=['target'])
    # combine target with other columns
    predicted_df = pd.concat([predict_df, label_df], axis=1)
    return predicted_df 

def save_tweets(twitter, df):
    if current_user.role == 'doctor':
        from app import tweet_table, engine, Patient, Tweet
        patientID = Patient.query.filter_by(twitter=twitter).first().patientID
        conn = engine.connect()
        df['datetime'] = pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')
        for index, row in df.iterrows():
            datetime_check = Tweet.query.filter_by(datetime=row.datetime).first()
            if not datetime_check:
                if row.symptoms != 'None':
                    row.symptoms = ','.join(row.symptoms)
                insert_tweet = tweet_table.insert().values(patientID=patientID,
                                datetime=row.datetime, tweet=row.tweet, symptom=row.symptoms, target=row.target)
                conn.execute(insert_tweet)
        conn.close()
    return

# update positive wordcloud
def plot_wordcloud(df, sentiment):
    stop_words = set(stopwords.words('english'))

    # create corpus from cleaned_tweet
    patient_tweet = df['cleaned_tweet']
    join_tweet = ' '.join(x for x in patient_tweet)

    # remove stopwords
    filter_tweet = [x for x in join_tweet.split(' ') if x not in stop_words]

    # get sentiment score for each word
    analyzer = SentimentIntensityAnalyzer()
    polarity = [analyzer.polarity_scores(x)['compound'] for x in filter_tweet]

    # assign positive and negative word
    word_list = []

    if sentiment == 'pos':
        for x in range (len(polarity)):
            if polarity[x] > 0:
                word_list.append(filter_tweet[x])
    else:
        for x in range (len(polarity)):
            if polarity[x] <= 0:
                word_list.append(filter_tweet[x])
            
    # join word 
    word_list = ' '.join(x for x in word_list)

    # render wordcloud
    wc = WordCloud(max_font_size=70, max_words=50, 
                   colormap='Dark2', collocations = False, 
                   background_color="white").generate(word_list)
    return wc.to_image()

# update wordcloud
def update_wordcloud(df, sentiment):
    img = BytesIO()
    plot_wordcloud(df, sentiment).save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())

# update alert
def update_alert(score):
    if current_user.role == 'doctor':
        if score >= 70:
            alert = dbc.Alert('The average depression score for this patient is in a higher range.\
                                The patient is more likely to have severe depression symptoms.\
                                It is recommended to schedule a consultation with the patient\
                                as soon as possible.', color='danger', dismissable=True, style={'margin-top':'1rem'})
        elif score >= 40:
            alert = dbc.Alert('The average depression score for this patient is in a middle range.\
                                The patient may experience depression symptoms.\
                                It is recommended that the patient continue to be examined over the next few days.', 
                                color='warning', dismissable=True, style={'margin-top':'1rem'})
        elif score > 0:
            alert = dbc.Alert('Good news! The average depression score for this patient is in a lower range.', 
                                color='info', dismissable=True, style={'margin-top':'1rem'})
    elif current_user.role == 'patient':
        if score >= 70:
            alert = dbc.Alert('We notice you have been on an emotional roller coaster lately.\
                                We strongly recommend that you make an appointment with your doctor for\
                                support. Remember, we are here for you, and it is okay not to be okay!', 
                                color='danger', dismissable=True, style={'margin-top':'1rem'})
        elif score >= 40:
            alert = dbc.Alert('We captured your mood swings at some point. \
                                It is recommended that you call someone you feel comfortable with when\
                                you can\'t stand it. Remember, there is always a way.', 
                                color='warning', dismissable=True, style={'margin-top':'1rem'})
        elif score > 0:
            alert = dbc.Alert('Good news! Your average depression score is in a lower range. Keep it up!', 
                                color='info', dismissable=True, style={'margin-top':'1rem'})
    elif current_user.role == 'public':
        if score >= 70:
            alert = dbc.Alert('We notice you have been on an emotional roller coaster lately.\
                                We strongly recommend that you call 03-2780 6803 (MMHA) for professional\
                                help. Remember, we are here for you, and it is okay not to be okay!', 
                                color='danger', dismissable=True, style={'margin-top':'1rem'})
        elif score >= 40:
            alert = dbc.Alert('We captured your mood swings at some point.\
                                It is recommended that you call someone you feel comfortable with when\
                                you can\'t stand it, or call 03-7956 8145 (BEFRIENDERS) for emotional support.\
                                Remember, there is always a way.', color='warning', dismissable=True,
                                style={'margin-top':'1rem'})
        elif score > 0:
            alert = dbc.Alert('Good news! Your average depression score is in a lower range. Keep it up!', 
                                color='info', dismissable=True, style={'margin-top':'1rem'})
    return alert

# highlight table rows
@callback(
    Output('table', 'style_data_conditional'),
    Input('table', 'derived_virtual_selected_row_ids')
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

# update user dashboard selection
@callback(
    Output('selection', 'children'),
    Input('selection', 'children')
)

def dashboard_selection(children):
    if current_user.role == 'patient' or current_user.role == 'public':
        from app import Patient, Public
        if current_user.role == 'patient':
            twitter = Patient.query.filter_by(patientID=current_user.userID).first().twitter
        else:
            twitter = Public.query.filter_by(id=current_user.userID).first().twitter

        children = [
            dbc.Input(id='patient-name', value=twitter, style={'display':'none'}),
            html.Label('Date Range', style={'margin-top':'5px'}),
            html.Div([
                dcc.DatePickerSingle(
                    id='start-date',
                    min_date_allowed=date(2015, 1, 1),
                    max_date_allowed=datetime.now().strftime('%m-%d-%Y'),
                    initial_visible_month=date(2022, 1, 1),
                    placeholder='start date'
                ),
                dcc.DatePickerSingle(
                    id='end-date',
                    min_date_allowed=date(2015, 1, 1),
                    max_date_allowed=datetime.now().strftime('%m-%d-%Y'),
                    initial_visible_month=date(2022, 1, 1),
                    placeholder='end date'
                ),
            ])
        ]
        return children   
    raise PreventUpdate  

# update patient name dropdown
@callback(
    Output('patient-name', 'options'),
    Input('patient-name', 'options')
)

def update_dropdown(option):
    if current_user.role == 'doctor':
        from app import Patient
        patient = Patient.query.filter_by(doctorID=current_user.userID).all()
        dropdown = [
            {'label': i.fullname, 'value': i.twitter} for i in patient
        ]
        return dropdown
    raise PreventUpdate

# output the session data
@callback(
    Output('patient-name', 'value'),
    Output('start-date', 'date'),
    Output('end-date', 'date'),
    Input('session', 'data')
)

def on_data(data):
    if data is None:
        raise PreventUpdate
    else:
        data = data or {}
        if current_user.username == data['username']:
            return (data['name'], data['start_date'], data['end_date'])
        raise PreventUpdate

