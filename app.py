# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 03:02:49 2024

@author: heman
"""
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
app =Flask(__name__)
model=pickle.load(open("model.pkl","rb"))
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    int_feature=[int(x) for x in request.form.values()]
    final_fetaure=[np.array(int_feature)]
    prediction=model.predict(final_fetaure)
    output=round(prediction[0],2)
    return render_template("index.html",prediction_text="employee salary should be ${}".format(output))

if __name__=="__main__":
    app.run(debug=True)

