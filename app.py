from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

lin_model = pickle.load(open('linmodel-712.pkl', 'rb'))  # read linear model
svr_model = pickle.load(open('svrmodel-712.pkl', 'rb'))  # read svr model

app = Flask(__name__)
# initializing Flask app


@app.route("/", methods=['GET'])
def index():
    return render_template('index.html')


@app.route("/documentation", methods=['GET'])
def documentation():
    return render_template('documentation.html')


@app.route("/predict", methods=['POST'])
def predict():
    # Must assign variables here using method below, else file views them as strings
    Squat = request.form.get('Squat', type=float)
    Benchpress = request.form.get('Benchpress', type=float)
    Deadlift = request.form.get('Deadlift', type=float)
    Weight = request.form.get('Weight', type=float)

    # Convert from lbs to kgs, if user inputed in lbs
    if (request.form['Units'] == 'lbs'):
        Weight = Weight / 2.205
        Squat = Squat / 2.205
        Benchpress = Benchpress / 2.205
        Deadlift = Deadlift / 2.205

    x = Weight
    W = Squat + Benchpress + Deadlift

    if (request.form['Gender'] == 'F'):
        # Wilks score variables and formula. Variables differ depending on gender.
        a = 594.31747775582
        b = -27.23842536447
        c = 0.82112226871
        d = -9.30733913e-3
        e = 4.731582e-5
        f = -9.054e-8
        Wilks = (W * 500) / (a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5)
    else:  # If male or other
        a = -216.0475144
        b = 16.2606339
        c = -2.388645e-3
        d = -1.13732e-3
        e = 7.01863e-6
        f = -1.291e-8
        Wilks = (W * 500) / (a + b*x + c*x**2 + d*x**3 + e*x**4 + f*x**5)

    # Use linear model is Wilks is less than 150. Else, use Support Vector Regression.
    if (Wilks < 150):
        prediction = lin_model.predict([[Wilks]])
        prediction = round(prediction[0], 1)  # Adjust to one decimal point
    if (Wilks >= 500):  # Due to lin model returning negative and SVR creating an upward curve beyond 500, a Wilks of 500 or more will return a static 'First' place prediction.
        prediction = 'First'
    else:
        prediction = svr_model.predict([[Wilks]])
        prediction = round(prediction[0], 1)

    if (type(prediction) != str):  # Only set bounds if 'First' is not predicted
        if (prediction < 4):     # Set lower bound to 1st if subtracting 3 would lead to a negative
            lower = 1
            upper = prediction + 3
            prediction = "Between " + str(lower) + " and " + str(upper)
        else:  # Set bounds of 3 distance
            lower = prediction - 3
            upper = prediction + 3
            prediction = "Between " + str(lower) + " and " + str(upper)

    return render_template('index.html', prediction=prediction)


if __name__ == "__main__":
    app.run()
