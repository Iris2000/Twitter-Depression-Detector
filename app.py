from json import dump
import numpy as np
from flask import Flask, request, jsonify, render_template
from keras.models import model_from_json
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# build bert model with distilbert - ligther version of bert
bert_model = SentenceTransformer('distilbert-base-nli-mean-tokens')

# loading the model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# loading the best model
model.load_weights('neural_network2.hdf5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    tweet = request.form['tweet']
    # create embeddings with bert - with 'cleaned_tweet' only
    embeddings = bert_model.encode(tweet)
    # combine embeddings and other features
    #embeddings2 = np.hstack((embeddings, predict_df[predict_df.columns.difference(['cleaned_tweet'])]))
    # define row and column number
    row_num = embeddings.shape[0]
    col_num = 1
    # reshape to 3 dimension for prediction
    predict_data = embeddings.reshape(row_num, col_num, 1)
    score = model.predict(predict_data)
    output = map(lambda x: 'normal' if x<0.5 else 'depressed', score)

    return render_template('index.html', prediction_text='This tweet shows {}'.format(output))

@app.route('/results',methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)