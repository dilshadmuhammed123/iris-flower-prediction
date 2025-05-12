from flask import Flask, render_template, request
import pickle
import numpy as np

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html',prediction_text='')

@app.route('/predict', methods=['POST'])
def prediction():
    #retrive the  values from the form
    SL=float(request.form['sepel_length'])
    sw=float(request.form['sepel_width'])
    PL=float(request.form['petel_length'])
    pw=float(request.form["petel_width"])
    #preapre the input data foe prediction
    input=np.array([[SL,sw,PL,pw]])
    model=pickle.load(open("model.pkl","rb"))
    result=model.predict(input)
    if result==0:
        prediction_text="setosa"
    elif result==1:
        prediction_text="versicolor"
    elif result==2:
        prediction_text="virginica"
    else:
        prediction_text="unknown species"
        
        prediction_text="virginica"
    return render_template('index.html',prediction_text=prediction_text)



if __name__ == "__main__":
    app.run(debug=True)  








    


