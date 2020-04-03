import numpy as np
np.random.seed(42)
import tensorflow as tf
# tf.set_random_seed(42)
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.layers import LSTM, Dropout
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras.layers import Dense, Activation, Dropout, RepeatVector
from tensorflow.keras.optimizers import RMSprop
import matplotlib.pyplot as plt
import pickle
import sys
import heapq

from pylab import rcParams

import pandas as pd
from flask import Flask, request, render_template
import requests
import json


app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
 	return render_template('home.html')
@app.route('/process',methods=['GET', 'POST'])
def nextFn():

#     sentences = []
#     next_chars = []
#     for i in range(0, len(text) - SEQUENCE_LENGTH, step):
#         sentences.append(text[i: i + SEQUENCE_LENGTH])
#         next_chars.append(text[i + SEQUENCE_LENGTH])
#     X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
#     y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
#     for i, sentence in enumerate(sentences):
#         for t, char in enumerate(sentence):
#             X[i, t, char_indices[char]] = 1
#         y[i, char_indices[next_chars[i]]] = 1
    # Dataset from - https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
#     df = pd.read_table('smsspamcollection/SMSSpamCollection',
#                    sep='\t',
#                    header=None,
#                    names=['label', 'sms_message'])
#     df['label'] = df.label.map({'ham':0, 'spam':1})
#     count_vector = CountVectorizer()

#     X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
#                                                     df['label'],
#                                                     random_state=1)

#     count_vector = CountVectorizer()

#     # Fit the training data and then return the matrix
#     training_data = count_vector.fit_transform(X_train)
#     naive_bayes = MultinomialNB()
#     naive_bayes.fit(training_data, y_train)
#     user_text= request.form.get('raw')
#     t_data= count_vector.transform([str(user_text)])
#     p = naive_bayes.predict(t_data)
    q=request.form.get('raw')


    

    seq = q[:40].lower()
    if len(seq)<40:
        n=40-len(seq)
        seq=n*' '+seq
# #     seq = q.lower()
#     print(seq)
#     print()


    output= str(predict_completions(seq, 5))
    output = output.lstrip('[').rstrip(']')



    return render_template('response.html', result1=seq,result2=output)
    return output


    
def prepare_input(text):
    SEQUENCE_LENGTH = 40
    step = 3
    global chars
    chars=['\n', ' ', '!', '"', '$', '%', '&', "'", '(', ')', '*', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '?', '@', '[', '\\', ']', '_', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '|', '}', '£', '¥', '§', '©', '®', '°', '»', 'é', '—', '‘', '’', '“', '”', '€', 'ﬁ', 'ﬂ']
    global char_indices
    with open('char_indices.p', 'rb') as fp:
        char_indices = pickle.load(fp)
        
    x = np.zeros((1, SEQUENCE_LENGTH, len(chars)))
    for t, char in enumerate(text):
        x[0, t, char_indices[char]] = 1.
        
    return x

def sample(preds, top_n=3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    return heapq.nlargest(top_n, range(len(preds)), preds.take)

def predict_completion(text):
    original_text = text
    generated = text
    completion = ''
    while True:
        x = prepare_input(text)
        preds = model.predict(x, verbose=0)[0]
        next_index = sample(preds, top_n=1)[0]
        next_char = indices_char[next_index]
        text = text[1:] + next_char
        completion += next_char
        
        if len(original_text + completion) + 2 > len(original_text) and next_char == ' ':
            return completion

def predict_completions(text, n=3):
    global graph
    graph = tf.compat.v1.get_default_graph()
    global model
    model= load_model('keras_model_gita.h5')
    history = pickle.load(open("history_gita.p", "rb"))
    global indices_char
    with open('indices_char.p', 'rb') as fp:
        indices_char = pickle.load(fp)    
    x = prepare_input(text)
    preds = model.predict(x, verbose=0)[0]
    next_indices = sample(preds, n)
    return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]        

if __name__=='__main__':
    app.run()