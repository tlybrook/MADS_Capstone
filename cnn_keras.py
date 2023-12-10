"""
Script to build CNN model in Keras.
"""
# Import packages
import os
import logging
import pathlib
import dill as pickle
import matplotlib.pyplot  as plt
import keras_metrics
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import tensorflow_datasets as tfds
from keras.layers import (
    Conv2D,
    Dense,
    MaxPool2D, 
    MaxPooling2D,
    Flatten,
    BatchNormalization,
    Dropout, 
    InputLayer
)

from keras.callbacks import ModelCheckpoint, EarlyStopping                                                                 
from keras.models import Sequential, load_model, save_model
from keras.losses import categorical_crossentropy
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import preprocess_input
from keras.src.metrics.confusion_metrics import activations
# Use legacy Adam optimizer to avoid slow runtimes on Mac M1/M2 chips
from keras.optimizers.legacy import Adam
from processes.preprocessing import (
    split_data,
    normalize_data,
    data_augmentation
)
from processes.utils import (
    get_model_tracker,
    get_model_summary,
    get_steps_per_epoch,
    get_key
)
from logs.logger_settings import *

MODEL_TRACKING = True
logger = logging.getLogger(__name__)

#Read in data and split into Train/Test/Valid
data_dir = pathlib.Path('./final_dataset/')
batch_size = 32
train_ds, val_ds, test_ds = split_data(data_dir, batch_size=batch_size)

#Normalize the data
train_ds = normalize_data(train_ds)
val_ds = normalize_data(val_ds)
test_ds = normalize_data(test_ds)

# Data augmentation in the training set
new_train_ds = data_augmentation(train_ds, rotation_val=0.5, flip_orientation="horizontal")

# Define input shape and num classes
num_classes = 4

# Get model Tracker
model_tracker = get_model_tracker(file='model_tracker.pickle', folder_path=None)

# Build design for the model
model = Sequential()
model.add(Conv2D(input_shape=(256,256,1),filters=32, kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=1024,activation="relu"))
model.add(Dense(units=512,activation="relu"))
model.add(Dense(units=256,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=4, activation="softmax"))

opt = Adam(learning_rate=0.001)
model.compile(
    optimizer=opt, 
    loss=categorical_crossentropy, 
    metrics=['accuracy', tf.keras.metrics.Recall()]
)

model_summary = get_model_summary(model=model)


#%% Fit the model to our training data
early = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, mode='auto')

results = model.fit(new_train_ds.repeat(),
                    steps_per_epoch=get_steps_per_epoch(new_train_ds, batch_size=batch_size),
                    epochs=100,
                    validation_data=val_ds.repeat(),
                    validation_steps=get_steps_per_epoch(val_ds, batch_size=batch_size),
                    callbacks=[early]
                    )

# Perform model tracking
key = get_key(model_output=model_tracker)
logger.debug(f"Model Key from most recent training: {key}")
model_tracker[key] = []
model_tracker[key].append(model_summary)
model_tracker[key].append(results.history)

with open('model_tracker.pickle', 'wb') as handle:
    pickle.dump(model_tracker, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Define the dataset to get predictions and observed values for
dataset = val_ds

class_headers = ['0', '1', '2', '3']
preds = model.predict(dataset)
y_pred = pd.DataFrame(preds, columns=class_headers)

y_true = pd.DataFrame(columns=class_headers)
for x, y in dataset:
    labels = tfds.as_numpy(y)
    labels = pd.DataFrame(labels, columns=class_headers)
    y_true = pd.concat([y_true, labels])
y_true.reset_index(drop=True, inplace=True)

def get_pred_class_label(row):
    """Gets predicted class labels from predicted probabilities.
    Input arg 'y_pred' is a dataframe with 4 columns containing probabilities.
    """
    index = np.argmax(row)
    new_row = [0] * len(row)
    new_row[index] = 1
    return new_row

def shrink_pred_matrix(row):
    """Converts 4 column pred matrix into single column.
    """
    ret_val = np.argmax(row)
    return ret_val

y_pred_2 = y_pred.apply(get_pred_class_label, axis=1, result_type="expand")
y_pred_class = y_pred_2.apply(shrink_pred_matrix, axis=1, result_type="expand")
y_true_class = y_true.apply(shrink_pred_matrix, axis=1, result_type="expand")


cm = confusion_matrix(y_true_class, y_pred_class)
print(cm)

acc = accuracy_score(y_true_class, y_pred_class)
print(acc)

plt.plot(results.history['loss'], label='train loss')
plt.plot(results.history['val_loss'], label='val loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.title('model Loss')
plt.legend()
plt.show()
plt.savefig('Val_loss')

plt.plot(results.history['accuracy'], label='train accuracy')
plt.plot(results.history['val_accuracy'], label='val accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.title('model accuracy')
plt.legend()
plt.show()
plt.savefig('Val_acc')

plt.plot(results.history['recall'], label='train recall')
plt.plot(results.history['val_recall'], label='val recall')
plt.ylabel('recall')
plt.xlabel('epoch')
plt.title('model recall')
plt.legend()
plt.show()
plt.savefig('Val_recall')
