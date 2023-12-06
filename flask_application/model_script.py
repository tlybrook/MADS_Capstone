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

model_path = './static/model_pytorch_base_cnn.h5'

def download_model():
    if os.path.exists(model_path):
        return
    else:
        s3_public_url = "https://mads-flask-app.s3.us-east-2.amazonaws.com/model_pytorch_base_cnn.h5" 
        urllib.request.urlretrieve(s3_public_url, model_path)
        return

download_model()