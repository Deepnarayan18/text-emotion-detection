import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of packages to install
packages = [
    "tensorflow",
    "numpy",
    "flask",
    "pickle5"
]

for package in packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

# Import necessary modules
try:
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.models import load_model
except ImportError:
    import keras
    from keras.preprocessing.sequence import pad_sequences
    from keras.models import load_model

from flask import Flask, request, render_template
import numpy as np
import pickle

app = Flask(__name__)

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as file:
    tokenizer = pickle.load(file)

# Load the label encoder
with open('label_encoder.pickle', 'rb') as file:
    label_encoder = pickle.load(file)

# Load the model
model = load_model('text_classification_model.h5')

max_length = 100  # Use the same max_length used during training

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['input_text']
        input_sequence = tokenizer.texts_to_sequences([input_text])
        padded_input_sequence = pad_sequences(input_sequence, maxlen=max_length)
        prediction = model.predict(padded_input_sequence)
        predicted_label = label_encoder.inverse_transform([np.argmax(prediction[0])])
        return render_template('index.html', prediction=predicted_label[0])

if __name__ == '__main__':
    app.run(debug=True)
