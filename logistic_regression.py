#%%
from keras.models import Sequential 
from keras.layers import Dense, Activation 
from PIL import Image
import os
import imagesize
import pandas as pd
import matplotlib.pyplot  as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
#from data_read import img_preprocess

root_folder = "./chest_scan_data/"
outcome_folders = ["/large.cell.carcinoma", "/normal", "/squamous.cell.carcinoma", "/adenocarcinoma"]

#%%
def img_preprocess(split_folder, root_folder=root_folder, outcome_folders=outcome_folders):
    imgs = []
    values = []
    for i in outcome_folders:
        folder_concat = root_folder + str(split_folder) + i
        files = os.listdir(folder_concat)
        for file in files:
            im = Image.open(f"{folder_concat}/{file}")
            new_image = im.resize((400, 300))
            image_array  = np.asarray(new_image.convert('RGB')).astype('float32')
            imgs.append(image_array)

            if i != "/normal":
                values.append(1)
            else:
                values.append(0)
    zipped = list(zip(imgs, values))
    random.shuffle(zipped)
    imgs, values = zip(*zipped)

    return imgs, values

input_size = 400*300*3 #these are the dimensions of the resized images

#%%
X_train, y_train = img_preprocess("train") 
X_train = np.array(X_train)
X_train = X_train.reshape(X_train.shape[0], -1) / 255
y_train = np.array(y_train)

X_test, y_test = img_preprocess("test")
X_test = np.array(X_test)
X_test = X_test.reshape(X_test.shape[0], -1) / 255
y_test = np.array(y_test)

X_val, y_val = img_preprocess("valid")
X_val = np.array(X_val)
X_val = X_val.reshape(X_val.shape[0], -1) / 255
y_val = np.array(y_val)

#should try to shuffle the data
#%%
num_classes = 2

model = Sequential() 
model.add(Dense(1, input_dim=input_size, activation='sigmoid')) 

# Hyperparameters
batch_size = 25
learning_rate = 1e-5
nb_epoch = 5
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=["accuracy", "Recall"]) 
history = model.fit(X_train, y_train, validation_data=(X_val, y_val)) 
score = model.evaluate(X_test, y_test, verbose=0) 
print('Test score:', score[0]) 
print('Test accuracy:', score[1])
print('Test recall:', score[2])


# %%
