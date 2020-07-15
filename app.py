# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 23:12:08 2020

@author: rakes
"""
from flask import Flask, request,jsonify,render_template
import pickle
import pandas as pd
import numpy as np 

app = Flask(__name__)
filename = 'diabetes-prediction.pkl'
model = pickle.load(open(filename, 'rb'))

@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    data5 = request.form['e']
    data6 = request.form['f']
    data7 = request.form['g']
    data8 = request.form['h']
    arr = np.array([[data1, data2, data3, data4,data5,data6,data7,data8]])
    pred = model.predict(arr)
    return render_template('after.html',data=pred)


if __name__ == "__main__":
    app.run(debug=True)
