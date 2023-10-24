import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

df=pd.read_csv("winequality-red.csv")
df=df.dropna()
df=df[["fixed acidity","citric acid","volatile acidity","chlorides","sulphates","quality"]]
ttle=st.title("Welcome to Wine Quality Test")
summary=st.write("Welcome to the Wine Quality Evaluator! 🍷📊"
                 "Discover the true quality of your wine with our Python-based application. "
                 "Utilizing a Kaggle dataset, this app simplifies the process of assessing wine quality. "
                 "Just input your wine's key characteristics, and we'll provide an insightful analysis. "
                 "While we can't predict the exact accuracy, we're here to make your wine experience more enjoyable and informed. Cheers! 🍇"
                  "Access the application's code on my GitHub page [here] (https://github.com/zaferErbil/WineTaste")


fixaci=st.slider("Select Fixed Acid Value: ", min_value=4.6,max_value=15.9,value=5.0,step=0.1)
citaci=st.slider("Select Citric Acid Value: ", min_value=0.0,max_value=1.0,step=0.01)
volaci=st.slider("Select Volatile Acidity Value: ", min_value=0.120,max_value=1.580,step=0.005)
chlords=st.number_input("Enter Chloride Values(0.012 - 0.611): ", min_value=0.010,max_value=0.615,step=0.001)
st.write(f"Value Entered: {chlords}")
slphats=st.slider("Select Sulphates Value: ", min_value=0.30,max_value=2.0,step=0.01)

X= df.drop("quality",axis=1)
y=df["quality"]

X_train,X_test, y_train,y_test=train_test_split(X,y,test_size=32,random_state=62)
model=RandomForestClassifier(random_state=32)
model.fit(X_train,y_train)

user_input=np.array([[fixaci, citaci, volaci,chlords,slphats]])
prediction=model.predict(user_input)
emojis = "🍷" * prediction[0]
prediction_text = f"Predicted Wine Quality: {prediction[0]}  {emojis}"
st.write(prediction_text)
