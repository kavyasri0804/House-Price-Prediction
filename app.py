import flask
from flask import Flask,render_template,request
import pickle
import numpy as np
with open("Model.pkl","rb") as f:
    m=pickle.load(f)
app=Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict",methods=["GET","POST"])
def predict():
    a = [float(i) for i in request.form.values()]
    r=m.predict([a])
    return render_template("index.html", prediction=float(r[0]))
if __name__=="__main__":
    app.run(debug=True)
