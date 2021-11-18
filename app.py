from flask import Flask,request,render_template
import joblib

saved_model = joblib.load('titanic_model.pkl')

flask_obj = Flask(__name__)

@flask_obj.route('/')
def home1():
    print("Homepage")
    return render_template('homepage.html')

@flask_obj.route('/predict' , methods = ['POST'])
def home():
    Pclass = request.form.get('Pclass')
    Sex = request.form.get('Sex')
    Age = request.form.get('Age')
    SibSp = request.form.get('SibSp')
    Parch = request.form.get('Parch')
    Fare = request.form.get('Fare')
    Embarked = request.form.get('Embarked')
    pred = saved_model.predict([[Pclass,Sex,Age,SibSp,Parch,Fare,Embarked]])
    if pred == 0:
        value = "Not Survived"
    else:
        value = "Survived"
    return render_template('homepage.html' ,value = value ) 

if __name__=='__main__':
    flask_obj.run(debug=True)