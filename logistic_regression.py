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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

#from data_read import img_preprocess

root_folder = "./chest_scan_data/"
outcome_folders = ["/large.cell.carcinoma", "/normal", "/squamous.cell.carcinoma", "/adenocarcinoma"]
split_folders = ["train", "test", "valid"]
#%%
def img_preprocess(split_folders=split_folders, root_folder=root_folder, outcome_folders=outcome_folders):
    imgs = []
    values = []
    del_count = 0
    for i in split_folders:
        for j in outcome_folders:
            folder_concat = root_folder + i + j
            files = os.listdir(folder_concat)
            for file in files:
                im = Image.open(f"{folder_concat}/{file}")
                new_image = im.resize((400, 300))
                image_array  = np.asarray(new_image.convert('RGB')).astype('float32')
                if len(imgs) != 0 and np.any(np.all(image_array == imgs, axis=1)):
                    del_count += 1
                else:
                    imgs.append(image_array)

                    if j != "/normal":
                        values.append(1)
                    else:
                        values.append(0)

    zipped = list(zip(imgs, values))
    random.shuffle(zipped)
    imgs, values = zip(*zipped)
    # Notes for MATT below:
    # maybe train test split the data here? or do it in its own function with the 
    #reshaping and casting to as array that we do below

    return imgs, values

input_size = 400*300*3 #these are the dimensions of the resized images

#%%
images, values = img_preprocess()

#%%
#Notes for MATT Below
# this section needs to be reworked to split the data and the function calls are wrong
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

#%%

clf = LogisticRegression(random_state=0, max_iter=500).fit(X_train, y_train)
preds = clf.predict(X_test)
probs = clf.predict_proba(X_test)
score = clf.score(X_test, y_test)
recall = recall_score(y_test, preds)


#%%
