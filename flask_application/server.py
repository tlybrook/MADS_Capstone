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
from torchvision import transforms
from PIL import Image
import pickle

app = Flask(__name__)
upload_folder = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = upload_folder
model_path = './static/model_pytorch_base_cnn.h5'
pickled_model = pickle.load(open(model_path, 'rb'))

image_size = (300, 400)

general_transform = transforms.Compose([
    transforms.Resize(image_size),  # Resize the images to a consistent size
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=(0.5), std=(0.5))  # Normalize the images to have a mean and std between 0, 1
])

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
            isExist = os.path.exists(upload_folder)
            if not isExist:
                # Create a new directory because it does not exist
                os.makedirs(upload_folder)
            filename = f"{app.config['UPLOAD_FOLDER']}/file1.jpg"
            file.save(filename)
            im = Image.open(filename)
            processed_image = general_transform(im)
            prediction_result = pickled_model.predict(processed_image)

            class_dict = {0: 'adenocarcinoma', 1: 'large cell carcinoma', 2: 'normal', 3: 'squamous cell carcinoma'}
            counter = 0
            highest = 0
            prediction = None
            for i in list(prediction_result[0]):
                if i > highest:
                    prediction = class_dict[counter]
                counter += 1
            print(prediction_result)

    return render_template('index.html', filename=filename, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, port=4000)