"""
Functions with a main for logistic regression preprocessing, spliting, training and evaluation. 
"""
from PIL import Image, ImageOps
import os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from processes.visualization import (confusion_matrix_viz)
import joblib

root_folder = "./final_dataset"

def log_reg_preprocess(root_folder=root_folder):
    """Preprocess the data to prepare for logistic regression train/test split. 
    All files passed in will be resized to the standard image size previously chosen in EDA, converted to 
    grayscale, converted to an array, and then all pixels will be normalized to be between 0, 1.

    ** ONLY FOR LOGISTIC REGRESSION! 

    Parameters
    -----------
    root_folder: location of the directory containing all the images

    Return
    -----------
    A tuple (imgs, labels) where imgs is a list of transformed image arrays and labels is the 
    corresponding class labels.
    """
    imgs = []
    labels = []
    image_size = (400, 300)
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

def split_data(imgs, values):
    """Split a list transformed image arrays and the corresponding class labels into train and test set.
    And converts train and test sets into the right format for Logistic Regression. 
    The determined split is 70% train and 30% test. No validation set in this case as no hyperparameter
    tuning happening. Logistic Regression is just a base model. 

    ** ONLY FOR LOGISTIC REGRESSION!

    Parameters
    -----------
    imgs: a list of transformed image arrays
    values: a list of class labels (ints)

    Return
    -----------
    A list containing the train and test sets in the following order: [X_train, X_test, y_train, y_test]
    """
    # In the first step we will split the data in training and remaining dataset
    X_train, X_test, y_train, y_test = train_test_split(imgs, values, train_size=0.7, random_state=42, shuffle=True)

    X_list = [X_train, X_test] 
    y_list = [y_train, y_test]
    final_list = []

    # Getting the data into the right format for the Logistic Regression.
    for i in X_list:
        temp = np.array(i)
        temp_reshape = temp.reshape(temp.shape[0], -1)
        final_list.append(temp_reshape)
    for j in y_list:
        temp = np.array(j)
        final_list.append(temp)

    return final_list

def logistic_reg(split_list):
    """Run a base logistic regression model on a split dataset.
    
    Parameters
    -----------
    split_list: A list containing the train and test sets in the following order: [X_train, X_test, y_train, y_test]

    Return
    -----------
    Prints the train and test accuracy and recall and returns a tuple (the y_true and y_preds) for the test set.
    """
    clf = LogisticRegression(random_state=42, max_iter=1000, C=0.0005).fit(split_list[0], split_list[2])
    y_preds_test = clf.predict(split_list[1])
    y_preds_train = clf.predict(split_list[0])
    acc_score_test = accuracy_score(split_list[3], y_preds_test)
    acc_score_train = accuracy_score(split_list[2], y_preds_train)
    recall_test = recall_score(split_list[3], y_preds_test, average="macro")
    recall_train = recall_score(split_list[2], y_preds_train, average="macro")
    
    print(f"Train Accuracy: {acc_score_train}\n\
    Test Accuracy: {acc_score_test}\n\
    Train Recall: {recall_train}\n\
    Test Recall: {recall_test}")

    return split_list[3], y_preds_test

if __name__ == '__main__':
     imgs, values = log_reg_preprocess()
     split_list = split_data(imgs, values)
     y_true, y_pred = logistic_reg(split_list)
     confusion_matrix_viz(y_true, y_pred, "log_reg_cm", "Logistic Regression Test Confusion Matrix" )
