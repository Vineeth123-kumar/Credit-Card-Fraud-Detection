from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict_page")
def predict_page():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    values = [float(x) for x in request.form.values()]

    final_input = np.array(values).reshape(1, -1)

    prediction = model.predict(final_input)

    if prediction[0] == 1:
        result = "Fraud Transaction Detected"
    else:
        result = "Genuine Transaction"

    return render_template("result.html", prediction_text=result)


if __name__ == "__main__":
    app.run(debug=True)