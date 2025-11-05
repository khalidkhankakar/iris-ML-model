from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load saved model
with open("iris_model.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    # render index.html file
    return app.send_static_file("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    # Extract features from request
    features = [
        data["sepal_length"],
        data["sepal_width"],
        data["petal_length"],
        data["petal_width"]
    ]

    prediction = model.predict([features])[0]
    print(f"Prediction: {prediction}")
    classes = ["Setosa", "Versicolor", "Virginica"]

    print(f"Predicted class: {classes[prediction]}")

    return jsonify({"prediction": classes[prediction]})

if __name__ == "__main__":
    app.run(debug=True)
