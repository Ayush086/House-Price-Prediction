import pickle
from flask import Flask, request, app, jsonify, url_for, render_template

import numpy as np
import pandas as pd

# create app
app = Flask(__name__) # it defines the starting point of the app

# load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))
std = pickle.load(open('scaling.pkl', 'rb'))

# defining a route and a function which will be executed when given route is reached.
@app.route('/')
def home():
    return render_template('home.html')


# APIs
@app.route('/predict_api', methods = ['POST'])  
def predict_api():
    data = request.json['data'] # input will be in json format, contained in data key
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = std.transform(np.array(list(data.values())).reshape(1, -1)) # standardized data

    # applying regression
    output = regmodel.predict(new_data)
    print(output[0])

    return jsonify(output[0])


@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()] # converting each value into float and saving them in a list
    final_ip = std.transform(np.array(data).reshape(1, -1)) # transforming data
    print(final_ip)

    output = regmodel.predict(final_ip)[0]

    return render_template("home.html", prediction_text = "Predicted House Price: {}".format(output))




# main function
if __name__ == '__main__':
    app.run(debug = True)




