"""Reddit prediction model Flask App"""

from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('home.html')


@app.route("/predict", )
def predict():
    text = 'I want to play call of duty'
    nn = pickle.load(open('model.pkl', 'rb'))
    vect = pickle.load(open('countvectorizer.pkl', 'rb'))
    subset = pd.read_pickle("./dataset.pkl")
    new = vect.transform([text])
    
    x = nn.kneighbors(new.todense())
    
    y = x[1][0][0]
    z = subset.iloc[y][0]
    
    return z
    # return render_template('home.html')



@app.route("/about")
def preds():
    return render_template('about.html')
