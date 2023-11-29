#%%
from PIL import Image, ImageOps
import os
import pandas as pd
import matplotlib.pyplot  as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, accuracy_score
from sklearn.model_selection import train_test_split
import random

root_folder = "./final_dataset"
#%%
# This function preprocesses the data to prepare for logistic regression train/test/valid split.
# Here we resize image, convert to grayscale, convert to array, and normalize array values to be between 0 and 1. 
def log_reg_preprocess(root_folder=root_folder):
    imgs = []
    labels = []
    image_size = (256, 256)
    dict_keys = {'adenocarcinoma': 0, 'large.cell.carcinoma': 1, 'normal': 2, 'squamous.cell.carcinoma': 3}

    folders = sorted(os.listdir(f"{root_folder}"))
    for i in folders:
        files = os.listdir(f"{root_folder}/{i}")
        label_key = dict_keys[i]

        for file in files:
            im = Image.open(f"{root_folder}/{i}/{file}")
            new_image = im.resize(image_size)
            gray_image = ImageOps.grayscale(new_image) 
            image_array  = np.asarray(gray_image).astype('float32')
            normalized_arr = image_array / 255
            imgs.append(normalized_arr)
            labels.append(label_key)

    return imgs, labels

# Here we are splitting the data into train/test 70/30
def split_data(imgs, values):
    zipped = list(zip(imgs, values))
    random.shuffle(zipped)
    imgs, values = zip(*zipped)
    
    # In the first step we will split the data in training and remaining dataset
    X_train, X_test, y_train, y_test = train_test_split(imgs, values, train_size=0.7, random_state=42)

    X_list = [X_train, X_test] 
    y_list = [y_train, y_test]
    final_list = []

    for i in X_list:
        temp = np.array(i)
        temp_reshape = temp.reshape(temp.shape[0], -1)
        final_list.append(temp_reshape)
    for j in y_list:
        temp = np.array(j)
        final_list.append(temp)

    return final_list

def logistic_reg(split_list):
    #X_train, X_test, y_train, y_test
    clf = LogisticRegression(random_state=42, max_iter=500).fit(split_list[0], split_list[2])
    y_preds = clf.predict(split_list[1])
    acc_score = accuracy_score(split_list[3], y_preds)
    recall = recall_score(split_list[3], y_preds, average="macro")

    return (acc_score, recall)

#%%
imgs, values = log_reg_preprocess()
split_list = split_data(imgs, values)
print(logistic_reg(split_list))

# %%
if __name__ == '__main__':
    imgs, values = log_reg_preprocess()
    split_list = split_data(imgs, values)
    print(logistic_reg(split_list))