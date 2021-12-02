from __future__ import division, print_function

import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

from time import time
# Define a flask app
app = Flask(__name__)

# Model guardado con Keras model.save()
MODEL_PATH = 'models/covid19_model_DenseNet201_full.h5'

# Carga el modelo entrenado
model = load_model(MODEL_PATH)
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224,3))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    x = x.astype('float32')
    x /= 255
    preds = model.predict(x)[0][0]
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        starter_time=time()
        preds = model_predict(file_path, model)
        THRESHOLD=0.5
        prediction = 1 if (preds >= THRESHOLD) else 0
        clases = ['Normal', 'Covid19+']
        resultado = clases[prediction]
        elapsed_time=time()-starter_time
        print("Probabilidad: ",preds)
        print("Tiempo de ejecución: %.10f seconds." % elapsed_time)
        return resultado
    return None


if __name__ == '__main__':
    app.run(debug=True)

