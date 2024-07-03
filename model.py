# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import pickle
dataset= pd.read_csv(r"C:\Users\heman\OneDrive\Documents\Python Scripts\hiring.csv")
dataset["test_score"].fillna(dataset["test_score"].mean(),inplace=True)
dataset["experience"]=dataset["experience"].map({"five":5,"two":2,"seven":7,"three":3,"ten":10,"eleven":11})
dataset["experience"].fillna(0, inplace=True)
x=dataset.iloc[:,:3]
y=dataset.iloc[:,-1]
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(x,y)
pickle.dump(LR,open("model.pkl","wb"))
model=pickle.load(open("model.pkl","rb"))
print(model.predict([[2,9,6]]))