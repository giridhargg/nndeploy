from flask import Flask, jsonify, request, url_for, redirect, render_template
import pandas as pd
# from flask_cors import CORS
import pickle
from model import get_model

from pathlib import Path
THIS_FOLDER = Path(__file__).parent.resolve()
# my_file = THIS_FOLDER / "nn_model.pkl"
# model = pickle.load(open(my_file, 'rb'))

model = get_model()

weights_file = THIS_FOLDER / "weights.h5"
# weights = pickle.load(open(weights_file, 'rb'))

model.load_weights(weights_file)

app = Flask(__name__)
# CORS(app)

@app.route('/')
def home():
    return render_template("ui.html")

@app.route('/predict', methods = ['POST'])
def predict():
    input_data = request.json['input']
    input_data = [[float(i) for i in row] for row in input_data]
    
    y_pred_val = model.predict(input_data)
    
    y_pred_val_labels = (y_pred_val >= 0.5).astype(int)
    for i in range(len(input_data)):
        input_data[i].append(int(y_pred_val_labels[i][0]))
    
    return jsonify(input_data)


if __name__ == '__main__':
    app.run(port=5000)