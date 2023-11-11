"""
Script to build CNN model in Keras
"""
#%%
# Import packages
from PIL import Image, ImageOps
import os
import shutil
import pandas as pd
import matplotlib.pyplot  as plt
import numpy as np
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

from keras.models import Sequential
from keras.losses import categorical_crossentropy
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import preprocess_input
from keras.src.metrics.confusion_metrics import activations
# Use legacy Adam optimizer to avoid slow runtimes on Mac M1/M2 chips
from keras.optimizers.legacy import Adam

# Read in data 
import pathlib
from sklearn.model_selection import train_test_split
data_dir = pathlib.Path('./final_dataset/')
def split_data(data_dir):

    batch_size = 32
    image_resize = (256, 256)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        labels="inferred",
        label_mode='categorical',
        color_mode="grayscale",
        subset="training",
        seed=42,
        image_size=image_resize)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        labels="inferred",
        label_mode='categorical',
        color_mode="grayscale",
        subset="validation",
        seed=42,
        image_size=image_resize)

    subset_size = int(0.5 * len(val_ds))

    test_ds = val_ds.take(subset_size)

    # Create a new dataset for the second subset
    val_ds = val_ds.skip(subset_size)
    
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = split_data(data_dir)

#Let's normalize the data
def normalize_data(ds):
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_ds = ds.map(lambda x, y: (normalization_layer(x), y))
    return normalized_ds

train_ds = normalize_data(train_ds)
val_ds = normalize_data(val_ds)
test_ds = normalize_data(test_ds)

# print(len(train_ds.class_labels), len(val_ds), len(test_ds))

#%%
#Data augmentation in the training set
def data_augmentation(train_ds):
    flip_layer = tf.keras.layers.RandomFlip("horizontal", input_shape=(256, 256, 1), seed=42)
    rotation_layer = tf.keras.layers.RandomRotation(0.5, seed=42)
    aug_train_ds = train_ds.map(lambda x, y: (flip_layer(x), y))
    aug_train_ds = aug_train_ds.map(lambda i, k: (rotation_layer(i), k))
    return aug_train_ds

aug_train_ds = data_augmentation(train_ds)
new_train_ds = train_ds.concatenate(aug_train_ds)

# Define input shape and num classes
num_classes = 4

#%%
model = Sequential()

model.add(Conv2D(input_shape=(256,256,1),filters=64, kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2))


# Pass data to dense layers
# Good idea to have fully connected layers at the end after flattening
model.add(Flatten())
model.add(Dense(units=512,activation="relu"))
model.add(Dense(units=4, activation="softmax"))

opt = Adam(learning_rate=0.001)
model.compile(
    optimizer=opt, 
    loss=categorical_crossentropy, 
    metrics=['accuracy', tf.keras.metrics.Recall()]
)
model.summary()


from keras.callbacks import ModelCheckpoint, EarlyStopping                                                                 

# checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')
#early = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=20, verbose=1, mode='auto')
results = model.fit(
    train_ds, 
    validation_data= val_ds, 
    validation_steps=10,
    epochs=5,
    steps_per_epoch = 1300 // 32
    #callbacks=[early],
)
history = model.fit(train_ds.repeat(),
                    steps_per_epoch=int(811/32),
                    epochs=25,
                    validation_data=val_ds.repeat(),
                    validation_steps=int(81/32))

#%%
plt.plot(results.history['loss'], label='train loss')
plt.plot(results.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')


# %%
