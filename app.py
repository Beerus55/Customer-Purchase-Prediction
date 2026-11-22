from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("YoHoHo.html", shyam=None)


@app.route("/predict", methods=["POST"])
def predict():

    age = int(request.form["age"])
    gender = int(request.form["gender"])
    salary = float(request.form["salary"])
    file=open("purhcase_pred1.pkl", "rb")
    model = pickle.load(file)
    file.close()

    # Prepare input
    features = np.array([[age, gender, salary]])

    # Prediction
    prediction = model.predict(features)[0]

    # Display result
    if prediction == 1:
        shyam = "Customer will purchase!"
    else:
        shyam = "Customer will not purchase."

    return render_template("YoHoHo.html", shyam=shyam)


if __name__ == "__main__":
    app.run(debug=False)