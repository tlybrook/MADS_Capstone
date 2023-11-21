from flask import Flask, render_template, request, redirect, url_for, abort, send_from_directory
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from flask import Flask, jsonify
import urllib.request
import threading
import requests
import json

app = Flask(__name__)
upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = upload_folder
model_path = './static/model_from_s3.keras'

loaded_model = load_model(model_path)

def preprocess_image(image_path):
    #This document helped with preproccessing steps below -> https://towardsdatascience.com/how-to-predict-an-image-with-keras-ca97d9cd4817
    #resize & convert to gray scale
    image_resize = (256, 256)
    im = image.load_img(image_path, target_size=image_resize)
    grayscaled = tf.image.rgb_to_grayscale(im)

    #convert to array and save as a batch of one
    img_array = image.img_to_array(grayscaled)
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    img_normalized = normalization_layer(img_array)
    processed_image = np.expand_dims(img_array, axis=0)

    return processed_image

@app.route('/')
def load_page():
    return render_template('loading.html')

@app.route('/app', methods=['GET', 'POST'])
def upload_file():
    filename = None
    prediction = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = f"{app.config['UPLOAD_FOLDER']}/file1.jpg"
            file.save(filename)
            processed_image = preprocess_image(filename)
            prediction_result = loaded_model.predict(processed_image)

            class_dict = {0: 'adenocarcinoma', 1: 'large cell carcinoma', 2: 'normal', 3: 'squamous cell carcinoma'}
            counter = 0
            highest = 0
            prediction = None
            for i in list(prediction_result[0]):
                if i > highest:
                    prediction = class_dict[counter]
                counter += 1

    return render_template('index.html', filename=filename, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=4000)