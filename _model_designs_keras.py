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

# Pass data to dense layers
# Good idea to have fully connected layers at the end after flattening
model.add(Flatten())
model.add(Dense(units=512,activation="relu"))
# model.add(Dropout(0.2))
model.add(Dense(units=4, activation="softmax"))


#############################################
############## MODEL KEY = 1 ################
#############################################
model.add(Conv2D(input_shape=(256,256,1),filters=32, kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# Pass data to dense layers
# Good idea to have fully connected layers at the end after flattening
model.add(Flatten())
model.add(Dense(units=512,activation="relu"))
# model.add(Dropout(0.2))
model.add(Dense(units=4, activation="softmax"))


#############################################
############## MODEL KEY = 2 ################
#############################################
model.add(Conv2D(input_shape=(256,256,1),filters=32, kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# Pass data to dense layers
# Good idea to have fully connected layers at the end after flattening
model.add(Flatten())
model.add(Dense(units=1024,activation="relu"))
model.add(Dense(units=512,activation="relu"))
# model.add(Dropout(0.2))
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

# Pass data to dense layers
# Good idea to have fully connected layers at the end after flattening
model.add(Flatten())
model.add(Dense(units=1024,activation="relu"))
model.add(Dense(units=512,activation="relu"))
# model.add(Dropout(0.2))
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
# model.add(Dropout(0.2))
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
# model.add(Dropout(0.2))
model.add(Dense(units=4, activation="softmax"))



#############################################
############## MODEL KEY = 6 ################
#############################################
model.add(Conv2D(input_shape=(256,256,1),filters=32, kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# Pass data to dense layers
# Good idea to have fully connected layers at the end after flattening
model.add(Flatten())
model.add(Dense(units=1024,activation="relu"))
model.add(Dense(units=512,activation="relu"))
model.add(Dense(units=256,activation="relu"))
# model.add(Dropout(0.2))
model.add(Dense(units=4, activation="softmax"))



#############################################
############## MODEL KEY = 7 ################
#############################################
Model 7 was trash


#############################################
############## MODEL KEY = 8 ################
#############################################
model.add(Conv2D(input_shape=(256,256,1),filters=32, kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# Pass data to dense layers
# Good idea to have fully connected layers at the end after flattening
model.add(Flatten())
model.add(Dense(units=1024,activation="relu"))
model.add(Dense(units=512,activation="relu"))
model.add(Dense(units=256,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=4, activation="softmax"))



#############################################
############## MODEL KEY = 9 ################
#############################################
# This Model was the first one where we updated split sizes to 70/20/10
# Future models will all use this split
model.add(Conv2D(input_shape=(256,256,1),filters=32, kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=2))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

# Pass data to dense layers
# Good idea to have fully connected layers at the end after flattening
model.add(Flatten())
model.add(Dense(units=2048,activation="relu"))
model.add(Dense(units=1024,activation="relu"))
model.add(Dense(units=512,activation="relu"))
# model.add(Dense(units=256,activation="relu"))
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

# Pass data to dense layers
# Good idea to have fully connected layers at the end after flattening
model.add(Flatten())
# model.add(Dense(units=2048,activation="relu"))
model.add(Dense(units=1024,activation="relu"))
model.add(Dense(units=512,activation="relu"))
model.add(Dense(units=256,activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(units=4, activation="softmax"))


##############################################
############## MODEL KEY = 11 ################
##############################################
# Same as above but data aug has been updated to flip first and then rotate
# Instead of flip and rotate in 1 go
# Training set size is doubled! :)


##############################################
############## MODEL KEY = 12 ################
##############################################
# Reverting to Model 10 - Same as model 10
# Except flips are vertical and horizontal instead of just horizontal


##############################################
############## MODEL KEY = 13 ################
##############################################
# This model is the same as model 10
# Doesnt count as an additional model in the grid search


