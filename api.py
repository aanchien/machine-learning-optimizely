import sys
import numpy as np
import pickle
import json
from sklearn.preprocessing import PolynomialFeatures
from flask import Flask, request, jsonify, Response

poly = PolynomialFeatures(degree=2)

filename = 'finalized_model.sav'
model = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

#Testing URL
@app.route('/hello/', methods=['GET', 'POST'])
def hello_world():
    return 'Hello, World!'

@app.route('/classifier/predict/', methods=['POST'])
def classifier():
    data = request.get_json()
    input_data = data['input']
    Xnew = input_data.split(',')
    Xnew = np.array([Xnew])
    Xnew = Xnew.astype(np.float64)
    variant0 = np.append ([[0]], Xnew)
    variant1 = np.append ([[1]], Xnew)
    Xnew = np.vstack((variant0, variant1))
    Xnew = poly.fit_transform(Xnew)
    ynew = model.predict(Xnew)
    if ynew[0] >= ynew[1]:
        d = {}
        d['variant'] = "0"
        return json.dumps(d)
    else:
        d= {}
        d['variant'] = "1"
        return json.dumps(d)

if __name__ == '__main__':
   app.run(threaded=True)
