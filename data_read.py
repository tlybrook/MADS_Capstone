#%%
from PIL import Image, ImageOps
import os
import shutil
import pandas as pd
import matplotlib.pyplot  as plt
import numpy as np
import tensorflow as tf
import random

#%%
data_paths = "./Data"
folder_labels = ["adenocarcinoma", 'large.cell.carcinoma', 'normal', 'squamous.cell.carcinoma']
#This function is to take in the dataset from Kaggle, removes duplicates, and
# moves them into new folders to be ready for data spliting and preprocessing.
def data_preparation(data_path, folder_labels):
    #creating new directory to put files
    try:
        for i in folder_labels:
            os.makedirs(f"./final_dataset/{i}")
            print("Folder %s created!" % i)
    except FileExistsError:
        print("Folder %s already exists" % i)
        print("Oops! Please make sure to delete folders and try running again!")
        return
    
    imgs = []
    count = 0
    del_count = 0
    top_level_folders = os.listdir(data_path)
    for i in top_level_folders:
        folders = os.listdir(f"{data_path}/{i}")
        for k in folders:
            files = os.listdir(f"{data_path}/{i}/{k}")
            for file in files:
                im = Image.open(f"{data_path}/{i}/{k}/{file}")
                new_image = im.resize((512, 512))
                image_array  = np.asarray(ImageOps.grayscale(new_image)).astype('float32')

                if len(imgs) != 0 and np.any(np.all(image_array == imgs, axis=1)):
                    del_count += 1
                else:
                    imgs.append(image_array)
                    
                    index = next(i for i, e in enumerate(folder_labels) if k[:6] in e)
                    shutil.copyfile(f"{data_path}/{i}/{k}/{file}", f"./final_dataset/{folder_labels[index]}/file{str(count)}.jpg")
                    count += 1
            print(f"{count} images added from {i} {k}")
    print("Number of deleted duplicates: ", del_count)
    return

data_preparation(data_paths, folder_labels)

#%%
import pathlib
from sklearn.model_selection import train_test_split
data_dir = pathlib.Path('./final_dataset/')
def split_data(data_dir):

    batch_size = 32
    image_resize = (256, 256)

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        label_mode="int",
        color_mode="grayscale",
        subset="training",
        seed=42,
        image_size=image_resize,
        batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        label_mode="int",
        color_mode="grayscale",
        subset="validation",
        seed=42,
        image_size=image_resize,
        batch_size=batch_size)

    subset_size = int(0.5 * len(val_ds))

    test_ds = val_ds.take(subset_size)

    # Create a new dataset for the second subset
    val_ds = val_ds.skip(subset_size)
    
    return train_ds, val_ds, test_ds

train_ds, val_ds, test_ds = split_data(data_dir)

#%%
#Let's normalize the data
def normalize_data(ds):
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    normalized_ds = ds.map(lambda x, y: (normalization_layer(x), y))
    return normalized_ds

train_ds = normalize_data(train_ds)
val_ds = normalize_data(val_ds)
test_ds = normalize_data(test_ds)

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

#%%
def ds_to_array(ds):
    imgs = []
    labels = []
    for image_batch, label_batch in new_train_ds:
        labels.append(label_batch.numpy())
        imgs.append(image_batch.numpy())

    images_array = np.concatenate(imgs, axis=0)
    images_array = images_array.reshape(images_array.shape[0], -1)

    labels_array = np.concatenate(labels, axis=0)
    return images_array, labels_array

X_train, y_train = ds_to_array(new_train_ds)
X_test, y_test = ds_to_array(test_ds)

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

#Run logistic regression
clf = LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000).fit(X_train, y_train)
preds = clf.predict(X_test)
score = clf.score(X_test, y_test)
recall = recall_score(y_test, preds, average='macro')

print(score)
print(recall)

#%%
#note need to come back and rework a little bit
def determine_dim_resize(root_folder=root_folder, split_folders=split_folders, outcome_folders=outcome_folders):
    image_dict = {}
    for i in split_folders:
        for j in outcome_folders:
            folder_concat = root_folder + i + j
            files = os.listdir(folder_concat)
            for file in files:
                im = Image.open(f"{folder_concat}/{file}")
                width, height = im.size
                image_dict[str(file)] = (width, height)

    widths = [v[0] for v in image_dict.values()]
    heights = [v[1] for v in image_dict.values()]
    filenames = image_dict.keys()

    image_dims = pd.DataFrame(data={'FileName': filenames, 'Width': widths, 'Height': heights})

    #graph for determining image dimensions to resize
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    points = ax.scatter(image_dims.Width, image_dims.Height, color='blue', alpha=0.5, picker=True)
    ax.set_title("Image Dimensions")
    ax.set_xlabel("Width", size=14)
    ax.set_ylabel("Height", size=14)
    return

# %%
