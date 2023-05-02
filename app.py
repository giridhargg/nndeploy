from flask import Flask, jsonify, request, render_template
from model import get_model

from pathlib import Path
THIS_FOLDER = Path(__file__).parent.resolve()

model = get_model()

weights_file = THIS_FOLDER / "weights.h5"

model.load_weights(weights_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("ui.html")

@app.route('/predict', methods = ['POST'])
def predict():
    input_data = [[float(i) for i in row] for row in request.json['input']]
    y_pred_val = model.predict(input_data)
    y_pred_val_labels = (y_pred_val >= 0.5).astype(int)
    for i in range(len(input_data)):
        input_data[i].append(int(y_pred_val_labels[i][0]))
    return jsonify(input_data)


if __name__ == '__main__':
    app.run(port=5000)