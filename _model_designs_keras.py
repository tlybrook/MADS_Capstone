"""
Documentation file to store the code used to build prior model iterations.

THIS FILE IS NOT MEANT TO BE RUN.

We saved our initial model ideas here in this file and copied them over to a clean file to run.
Once we started working in PyTorch we used a pickle file to manage the different
model runs so we didn tneed to do this process anymore.
"""

#############################################
############## MODEL KEY = 0 ################
#############################################
model.add(Conv2D(input_shape=(256,256,1),filters=32, kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2))
model.add(Flatten())
model.add(Dense(units=512,activation="relu"))
model.add(Dense(units=4, activation="softmax"))

#############################################
############## MODEL KEY = 1 ################
#############################################
model.add(Conv2D(input_shape=(256,256,1),filters=32, kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=512,activation="relu"))
model.add(Dense(units=4, activation="softmax"))

#############################################
############## MODEL KEY = 2 ################
#############################################
model.add(Conv2D(input_shape=(256,256,1),filters=32, kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=1024,activation="relu"))
model.add(Dense(units=512,activation="relu"))
model.add(Dense(units=4, activation="softmax"))

#############################################
############## MODEL KEY = 3 ################
#############################################
model.add(Conv2D(input_shape=(256,256,1),filters=32, kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=1024,activation="relu"))
model.add(Dense(units=512,activation="relu"))
model.add(Dense(units=4, activation="softmax"))

#############################################
############## MODEL KEY = 4 ################
#############################################
model.add(Conv2D(input_shape=(256,256,1),filters=32, kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=1024,activation="relu"))
model.add(Dense(units=512,activation="relu"))
model.add(Dense(units=4, activation="softmax"))

#############################################
############## MODEL KEY = 5 ################
#############################################
model.add(Conv2D(input_shape=(256,256,1),filters=32, kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=1024,activation="relu"))
model.add(Dense(units=512,activation="relu"))
model.add(Dense(units=4, activation="softmax"))

#############################################
############## MODEL KEY = 6 ################
#############################################
model.add(Conv2D(input_shape=(256,256,1),filters=32, kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=1024,activation="relu"))
model.add(Dense(units=512,activation="relu"))
model.add(Dense(units=256,activation="relu"))
model.add(Dense(units=4, activation="softmax"))

#############################################
############## MODEL KEY = 7 ################
#############################################
model.add(Conv2D(input_shape=(256,256,1),filters=32, kernel_size=(5,5),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=(5,5), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=1024,activation="relu"))
model.add(Dense(units=512,activation="relu"))
model.add(Dense(units=256,activation="relu"))
model.add(Dense(units=4, activation="softmax"))

#############################################
############## MODEL KEY = 8 ################
#############################################
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

#############################################
############## MODEL KEY = 9 ################
#############################################
# This Model was the first one where we updated split sizes to 70/20/10. Future models will all use this split.
model.add(Conv2D(input_shape=(256,256,1),filters=32, kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=2048,activation="relu"))
model.add(Dense(units=1024,activation="relu"))
model.add(Dense(units=512,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=4, activation="softmax"))

##############################################
############## MODEL KEY = 10 ################
##############################################
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

##############################################
############## MODEL KEY = 11 ################
##############################################
# Same as model 11 but data augmentation has been updated to flip first and then rotate versus flipping and
# rotated at the same time. This doubled the training set size.

##############################################
############## MODEL KEY = 12 ################
##############################################
# Reverting to Model 10 - Same as model 10, except flips are vertical and horizontal instead of just horizontal.

##############################################
############## MODEL KEY = 13 ################
##############################################
# Reverted back to model 10 as that was better performing than model 11 and 12.

