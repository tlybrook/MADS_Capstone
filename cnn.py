"""
Script to build CNN model in Keras
"""
#%%
# Import packages
import os
import logging
import pathlib
import dill as pickle
import matplotlib.pyplot  as plt
import keras_metrics
import tensorflow as tf
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
from preprocessing import (
    split_data,
    normalize_data,
    data_augmentation
)
from utils import (
    get_model_tracker,
    get_model_summary,
    get_steps_per_epoch,
    get_key
)
from logger_settings import *

# Set to False to not add models to tracking file
MODEL_TRACKING = True
logger = logging.getLogger(__name__)

#%% Read in data and split into Train/Test/Valid
data_dir = pathlib.Path('./final_dataset/')
batch_size = 32
train_ds, val_ds, test_ds = split_data(data_dir, batch_size=batch_size)

#%% normalize the data
train_ds = normalize_data(train_ds)
val_ds = normalize_data(val_ds)
test_ds = normalize_data(test_ds)

#%% Data augmentation in the training set
new_train_ds = data_augmentation(train_ds, rotation_val=0.5, flip_orientation="horizontal")

# Define input shape and num classes
num_classes = 4

#%% Get model Tracker
model_tracker = get_model_tracker(file='model_tracker.pickle', folder_path=None)

#%% Build design for the model
model = Sequential()

model.add(Conv2D(input_shape=(256,256,1),filters=32, kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# Pass data to dense layers
# Good idea to have fully connected layers at the end after flattening
model.add(Flatten())
# model.add(Dense(units=2048,activation="relu"))
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
# checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
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

model.save('./model_objects/model_13.keras')
# model = load_model(f'./model_objects/model_{key}.keras')

with open('model_tracker.pickle', 'wb') as handle:
    pickle.dump(model_tracker, handle, protocol=pickle.HIGHEST_PROTOCOL)

#%%
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

#%%

# key = 10

# plt.plot(model_tracker[key][1]['loss'], label='train loss')
# plt.plot(model_tracker[key][1]['val_loss'], label='val loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.title(f'model {key} Loss')
# plt.legend()
# plt.show()
# plt.savefig('Val_loss')

# plt.plot(model_tracker[key][1]['accuracy'], label='train accuracy')
# plt.plot(model_tracker[key][1]['val_accuracy'], label='val accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.title(f'model {key} accuracy')
# plt.legend()
# plt.show()
# plt.savefig('Val_acc')

# plt.plot(model_tracker[key][1]['recall'], label='train recall')
# plt.plot(model_tracker[key][1]['val_recall'], label='val recall')
# plt.ylabel('recall')
# plt.xlabel('epoch')
# plt.title(f'model {key} recall')
# plt.legend()
# plt.show()
# plt.savefig('Val_recall')

# %%
