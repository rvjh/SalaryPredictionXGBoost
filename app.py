## if we use sklearn then the libraries may mismatch which will create an issue so careful about versions
from flask import Flask,render_template,request
#import pandas as pd
import numpy as np
import pickle
from xgboost import XGBClassifier
import sklearn


app = Flask(__name__)
@app.route('/',methods=['GET'])
def homepage():
    return render_template('home.html')


@app.route('/predict',methods=['POST','GET'])
def home():
    if request.method == 'POST':
        try:
            #------------for age----------------------
            d1 = float(request.form['a'])
            #------------for workclass----------------
            req1 = request.form['b']
            work1 = ['Private','Self-emp-not-inc','Self-emp-inc','Federal-gov','Local-gov','State-gov','Never-worked']
            d2 = []
            for i in work1:
                if req1 == i:
                    d2.append(1)
                else:
                    d2.append(0)
            ##------------for fnlwgt--------------------
            d3 = float(request.form['c'])
            ##------------for education-----------------
            req2 = request.form['d']
            work2=['Bachelors','HS-grad','Primary','Masters','Some-college','Assoc-acdm','Assoc-voc','Doctorate','Prof-school','Preschool']
            d4 = []
            for i in work2:
                if req2 == i:
                    d4.append(1)
                else:
                    d4.append(0)
            ##-------------for education number-----------
            d5 = float(request.form['e'])
            ##-------------for marital-status-------------
            req3 = request.form['f']
            work3 = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
            d6 = []
            for i in work3:
                if req3 == i:
                    d6.append(1)
                else:
                    d6.append(0)
            ##-------------for occupation-----------------
            req4 = request.form['g']
            work4 = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty','Handlers-cleaners','Machine-op-inspect','Adm-clerical','Farming-fishing','Transport-moving','Priv-house-serv','Protective-serv','Armed-Forces']
            d7 = []
            for i in work4:
                if req4 == i:
                    d7.append(1)
                else:
                    d7.append(0)
            ##-------------for relationship----------------
            req5 = request.form['h']
            work5 = ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']
            d8 = []
            for i in work5:
                if req5 == i:
                    d8.append(1)
                else:
                    d8.append(0)
            ##-------------for race-----------------------
            req6 = request.form['i']
            work6 = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
            d9 = []
            for j in work6:
                if req6 == j:
                    d9.append(1)
                else:
                    d9.append(0)
            ##--------------for sex-----------------------
            req7 = request.form['j']
            work7 = ['Female','Male']
            d10 = []
            for i in work7:
                if req7 == i:
                    d10.append(1)
                else:
                    d10.append(0)
            ##-------------for capital gain---------------
            d11 = float(request.form['k'])
            ##-------------for capital loss---------------
            d12 = float(request.form['l'])
            ##-------------for hours per week---------------
            d13 = float(request.form['m'])
            ##------------for native country---------------
            req8 = request.form['n']
            work8 = ['North_America', 'Central_America','South_America','EU','Asian']
            d14 = []
            for i in work8:
                if req8 == i:
                    d14.append(1)
                else:
                    d14.append(0)

            #arr = np.array([[d1,d3,d5,d11,d12,d13,d2[0],d2[1],d2[2],d2[3],d2[4],d2[5],d2[6],d4[0],d4[1],d4[2],d4[3],d4[4],d4[5],d4[6],d4[7],d4[8],d4[9],d6[0],d6[1],d6[2],d6[3],d6[4],d6[5],d6[6],d7[0],d7[1],d7[2],d7[3],d7[4],d7[5],d7[6],d7[7],d7[8],d7[9],d7[10],d7[11],d7[12],d7[13],d8[0],d8[1],d8[2],d8[3],d8[4],d8[5],d9[0],d9[1],d9[2],d9[3],d9[4],d10[0],d10[1],d14[0],d14[1],d14[2],d14[3],d14[4]]])
            #-----------------------------------------#

            # Loading the saved models into memory
            filename_scaler = 'scaler_model_rd.pickle'
            filename = 'xgboost_model_rd.pickle'
            scaler_model = pickle.load(open(filename_scaler, 'rb'))
            loaded_model = pickle.load(open(filename, 'rb'))

            scaled_data = scaler_model.transform([[d1,d3,d5,d11,d12,d13,d2[0],d2[1],d2[2],d2[3],d2[4],d2[5],d2[6],d4[0],d4[1],d4[2],d4[3],d4[4],d4[5],d4[6],d4[7],d4[8],d4[9],d6[0],d6[1],d6[2],d6[3],d6[4],d6[5],d6[6],d7[0],d7[1],d7[2],d7[3],d7[4],d7[5],d7[6],d7[7],d7[8],d7[9],d7[10],d7[11],d7[12],d7[13],d8[0],d8[1],d8[2],d8[3],d8[4],d8[5],d9[0],d9[1],d9[2],d9[3],d9[4],d10[0],d10[1],d14[0],d14[1],d14[2],d14[3],d14[4]]])
            #loaded_model = pickle.load(open(filename, 'rb'))  # loading the model file from the storage
            # predictions using the loaded model file
            prediction = loaded_model.predict(scaled_data)
            print('prediction is', prediction)
            # showing the prediction results in a UI
            return render_template('after.html', data=prediction)
        except Exception as e:
            print('The Exception message is: ', e)
            return 'Opps!! you are supposed to give a number not character..'

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run()


