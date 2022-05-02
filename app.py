import os
import logging

from flask import Flask, request, jsonify, render_template

from model import TranslatorClassifier

app = Flask(__name__)

# define model path
model_path = 'machine_translation_model.h5'

# create instance
model = TranslatorClassifier(model_path)
logging.basicConfig(level=logging.INFO)


@app.route("/")
def index():
    """Provide simple health check route."""
    return render_template("index.html")


@app.route('/predict',methods=['POST'])
def predict():
    """Provide main prediction API route. Responds to both GET and POST requests."""
    logging.info("Predict request received!")

    inputted_text = [i for i in request.form.values()]

    prediction = model.predict(inputted_text[0])

    logging.info("prediction from model= {}".format(prediction))

    return render_template("index.html", prediction_text="The Predicted sentence in french is {}".format(prediction))



def main():
    """Run the Flask app."""
    app.run(host="0.0.0.0", port=8006, debug=True)


if __name__ == "__main__":
    main()