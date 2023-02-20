from flask import Flask,url_for,render_template,request
import pickle
import numpy as np

model=pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def welcome():
    return render_template('index.html')

@app.route('/submit',methods=['POST','GET'])
def predict():
    gender=int(request.form['gender'])
    married=int(request.form['married'])
    dependents=int(request.form['dependents'])
    education=int(request.form['education'])
    selfemp=int(request.form['selfemp'])
    app=int(request.form['app'])
    coapp=int(request.form['coapp'])
    amount=int(request.form['amount'])
    term=int(request.form['term'])
    cred=int(request.form['cred'])
    area=int(request.form['area'])
    a=np.array([[gender,married,dependents,education,selfemp,app,coapp,amount,term,cred,area]])
    prediction=model.predict(a)
    if prediction==1:
        return render_template('result1.html')
    else:
        return render_template('result0.html')


if __name__=="__main__":
    app.run(debug=True)