import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from preprocessing import *

#from joblib import load
app = Flask(__name__)
model = pickle.load(open('project1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend',methods=['POST'])
def recommend():
    '''
    For rendering results on HTML GUI
    '''
    user_ingredients = [[str(x) for x in request.form.values()]]

    recipe_prediction, recipe_link, imageLink = user_input(user_ingredients)
    print(imageLink)
    return render_template('index.html', recipe_pred = recipe_prediction, recipe_link = recipe_link,ImageLink = imageLink)

if __name__ == "__main__":
    app.run(debug=True)
