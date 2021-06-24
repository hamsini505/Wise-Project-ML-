import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, request, jsonify, render_template
import pickle


cols = ['Title', 'Ingredients','WebLinks', 'ImageLinks']
data  = pd.read_csv('dataset.csv',names = cols)

stopwords.words('english')

unused = ['fresh','cup','cups','pound','pounds', 'teaspoon', 'tablespoon', 'g', 'kg', 'ml', 'l', 'litre', 'kmobs','slices', 'small', 'handful', 'large', 'bunch', 'extra', 'virgin', 'such', 'as','broad', 'raw', 'heart', 'bulb', 'sticks', 'tins', 'mixed', 'wild', 'quality', 'single', 'jar', 'new', 'season', 'broad', 'black', 'regular', 'plus','whole','pinch','serve', 'dusting', 'tablespoons', 'teaspoons', 'super-ripe', 'sustainable','sources',  'scrubbed', 'hard']

def text_process(mess):
    #print(mess)
    nopunc = [char for char in mess if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    nopunc = [word for word in nopunc.split() if word.lower() not in stopwords.words('english') and word.isalpha()]
    return [word for word in nopunc if word.lower() not in unused]

features = data['Ingredients'].apply(text_process)

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    words = ""
    for word in text:
        words += lemmatizer.lemmatize(word) + " "
    return words

features = features.apply(lemmatization)

features = data['Ingredients']

vectorizer = CountVectorizer()
vectorizer.fit(features)
vector = vectorizer.transform(features)
columnNames = vectorizer.get_feature_names()

resArray = vector.toarray()

finalVector = []
for item in resArray:
    li = list(item)
    finalVector.append(li)

df_add = pd.DataFrame(data=finalVector,columns=columnNames)
data = pd.concat([data,df_add], axis=1)

df = data.drop(columns='Ingredients')
df = data.drop(columns='WebLinks')
df = data.drop(columns='ImageLinks')
df1 = pd.pivot_table(data=df,index=['Title'])


features_matrix = csr_matrix(df1.values)

model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(features_matrix)

def calc(target):
    distances, indices = model_knn.kneighbors(target, n_neighbors = 3)
    outputRecipe = df1.index[indices.flatten()[0]]
    outputLink = data[data['Title'] == outputRecipe]
    outputLink2 = outputLink.iloc[0,2]
    imageLink = outputLink.iloc[0,3]
    return outputRecipe, outputLink2, imageLink

def user_input(user_ingred1):
    zero, one = 0, 1
    list2 = []    
    
    user_ingred = str(user_ingred1)
    ingred = preprocess(user_ingred)
    
    for item in columnNames:
        if item in ingred:
            list2.append(one)
        else:
            list2.append(zero)

    arr = pd.DataFrame([list2])
    return calc(arr.values.reshape(1, -1))
    
def preprocess(user_ingred):
    list1 = user_ingred.split(",")
    ingredients = []
    
    for item in list1:
        list2 = item.split()
        for item1 in list2:
            ingredients.append(item1.lower())
    return ingredients

pickle.dump(model_knn,open('project1.pkl','wb'))