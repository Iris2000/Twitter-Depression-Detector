# import packages
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
import gsdmm
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nrclex import NRCLex
from collections import Counter
from keras.models import model_from_json
from sentence_transformers import SentenceTransformer
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

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
    
    for column in columns:
        if column not in df:
            df[column] = 0

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
                                                                       'tweet', 'cleaned_tweet'])]))
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
    label = list(map(lambda x: 0 if x<0.5 else 1, score))
    label_df = pd.DataFrame(label, columns=['target'])
    # combine target with other columns
    predicted_df = pd.concat([predict_df, label_df], axis=1)
    return predicted_df

@app.route('/')
def home():
    return render_template('dashboard.py')

@app.route('/predict2',methods=['POST'])
def predict2():

    # import csv
    df = pd.read_csv (r'C://Users//lvlip//Documents//BCSI Sem 6//FYP 4202//CSV//testing.csv', engine='python')
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
    return render_template('index.html', prediction_text=predicted_df.to_html())

if __name__ == "__main__":
    app.run(debug=True)