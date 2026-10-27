from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "titanic_model.pkl")
scaler_path = os.path.join(BASE_DIR, "titanic_scaler.pkl")

model = pickle.load(open(model_path, "rb"))
scaler = pickle.load(open(scaler_path, "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # Get values from HTML form
    pclass = int(request.form["pclass"])
    sex = request.form["sex"]
    age = float(request.form["age"])
    sibsp = int(request.form["sibsp"])
    parch = int(request.form["parch"])
    fare = float(request.form["fare"])
    embarked = request.form["embarked"]

    # Convert categorical values into numbers
    sex_male = 1 if sex == "male" else 0

    embarked_q = 1 if embarked == "Q" else 0
    embarked_s = 1 if embarked == "S" else 0

    # IMPORTANT:
    # Same order as the columns used during training
    input_data = np.array([[
        pclass,
        age,
        sibsp,
        parch,
        fare,
        sex_male,
        embarked_q,
        embarked_s
    ]])

    # Scale the input
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)[0]

    # Get probability
    probability = model.predict_proba(input_scaled)[0][1] * 100

    # Result
    if prediction == 1:
        result = "Passenger is predicted to SURVIVE"
    else:
        result = "Passenger is predicted NOT to survive"

    return render_template(
        "index.html",
        prediction=result,
        probability=round(probability, 2)
    )


if __name__ == "__main__":
    app.run(debug=True)