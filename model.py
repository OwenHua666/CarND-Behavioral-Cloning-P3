# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 05:53:27 2017

@author: Owen Hua
Subject: Udacity Self-Driving Car Nanodegree
Assignment: Behavioral Cloning Traning Model
"""
# Import useful packages
import numpy as np
import cv2
import math
import sklearn
import csv

# Edit the path for reading the images in the right directory
def path_editor(local_path):
    filename = local_path.split("/")[-1]
    host_path = 'data/IMG/'+filename
    # print(host_path)
    return host_path

# Brightness Augmentation by adjusting the V channel
def augment_brightness(image):
    imageHSV = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    random = 0.2 + np.random.uniform()
    imageHSV[:,:,2] = imageHSV[:,:,2] * random
    imageout = cv2.cvtColor(imageHSV, cv2.COLOR_HSV2RGB)
    return imageout

# Translate the image and correspondingly adjust the steering angle to create
# extra images for training.
def trans_image(image, steer, trans_range):
    trans_x = trans_range * np.random.uniform() - trans_range/2 # To get positive and negative shift
    steer_ang = steer + trans_x / trans_range * 2 * 0.2
    trans_y = 10 * np.random.uniform() - 10/2
    Trans_m = np.float32([[1,0,trans_x],[0,1,trans_y]])
    image_trans = cv2.warpAffine(image, Trans_m, (image.shape[1], image.shape[0]))
    
    return image_trans, steer_ang, trans_x

# Defint the size of the new image
new_image_col = 64
new_image_row = 64

# Crop oup the unrelevent imformation in the image and resize it to 64x64
def preprocessImage(image, new_image_row, new_image_col):
    image = image[math.floor(image.shape[0] * 0.4):(image.shape[0]-35), 0:image.shape[1]]
    image = cv2.resize(image,(new_image_row, new_image_col), interpolation = cv2.INTER_AREA)
    return image

# Preprocess the images for training dataset and validation dataset
def preprocess_image_train(image_path, steer, new_image_row, new_image_col):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    steer_out = steer
    # image, steer_out, trans_x = trans_image(image, steer, 150)
    # image = augment_brightness(image)
    image = preprocessImage(image, new_image_row, new_image_col)
    image = np.array(image)
    # Flip the image and inverse the steer angle
    flip_token = np.random.randint(2)
    if flip_token == 0:
        image = cv2.flip(image,1)
        steer_out = -steer_out
    return image, steer_out
def preprocess_image_predict(path, steer, new_image_row, new_image_col):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = preprocessImage(image, new_image_row, new_image_col)
    image = np.array(image)
    return image, steer

# Read the image the split the image into training and validation dataset
samples = []
count=0
turn_angle_threshold = 0.15
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    camera = ['center', 'left', 'right']
    for line in reader:
        # Disregard the tags in .csv file
        if line[0] == 'center':
            continue
        
        # Drop 75% of images whose steering angle is less than 0.15
        token = np.random.uniform()
        if abs(float(line[3])) < turn_angle_threshold:
            if token > 0.75:
                for i in range(3):
                    count += 1
                    samples.append([camera[i], line[i], line[3]])
            else:
                continue
        else:
            for i in range(3):
                count += 1
                samples.append([camera[i], line[i], line[3]])

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

from random import shuffle


# Keras Generates 
def train_data_generator(samples, batch_size, new_image_row, new_image_col):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                image_name = path_editor(batch_sample[1])
                steer_i = float(batch_sample[2])
                # Adjust the steer angle based on the camera position
                camera_pose = batch_sample[0]
                if camera_pose == 'center':
                    steer_ang = steer_i
                elif camera_pose == 'left':
                    steer_ang = steer_i + 0.25
                else:
                    steer_ang = steer_i - 0.25
                image, ang = preprocess_image_train(image_name, steer_ang, new_image_row, new_image_col)
                images.append(image)
                angles.append(ang)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
            
            
def valid_data_generator(samples, batch_size, new_image_row, new_image_col):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                image_name = path_editor(batch_sample[1])
                steer_i = float(batch_sample[2])
                # Adjust the steer angle based on the camera position
                camera_pose = batch_sample[0]
                if camera_pose == 'center':
                    steer_ang = steer_i
                elif camera_pose == 'left':
                    steer_ang = steer_i + 0.25
                else:
                    steer_ang = steer_i - 0.25
                image, ang = preprocess_image_predict(image_name, steer_ang, new_image_row, new_image_col)
                images.append(image)
                angles.append(ang)
            X_valid = np.array(images)
            y_valid = np.array(angles)
            yield sklearn.utils.shuffle(X_valid, y_valid)
            
            
# KERAS neural network training model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Lambda
from keras.layers import ELU
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten

new_size_row = 64
new_size_col = 64

input_shape = (new_size_row, new_size_col, 3)

model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=input_shape,output_shape=input_shape))

## model.add(Conv2D(24, (5, 5), name='conv1', padding="valid"))
## model.add(Activation('relu'))
## model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(36, (5, 5), name='conv2'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
## model.add(Conv2D(48, (5, 5), name='conv3', padding='valid'))
## model.add(Activation('relu'))
## model.add(Conv2D(64, (3, 3), name='conv4'))
## model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(1,1)))
model.add(Conv2D(48, (3, 3), name='conv5', padding='valid'))


model.add(Flatten())

model.add(Dense(100, name="hidden1", kernel_initializer="he_normal"))
model.add(ELU())
model.add(Dropout(0.5))
model.add(Dense(50,name='hidden2', kernel_initializer="he_normal"))
model.add(ELU())
## model.add(Dropout(0.5))
model.add(Dense(10,name='hidden3',kernel_initializer="he_normal"))
model.add(ELU())
## model.add(Dropout(0.5))
model.add(Dense(1, name='output', kernel_initializer="he_normal"))
adam = Adam(lr=1e-4)

model.compile(optimizer=adam,
          loss='mse')


# Training
val_size = len(validation_samples)

batch_size = 256

i_best = 0
val_best = 1000
valid_generator = valid_data_generator(validation_samples, batch_size, new_size_col, new_size_row)
train_generator = train_data_generator(train_samples, batch_size, new_size_col, new_size_row)
model.fit_generator(train_generator, steps_per_epoch = np.round(len(train_samples)/batch_size), epochs=8, verbose=1, callbacks=None, validation_data=valid_generator,
                    validation_steps=np.round(val_size/batch_size), class_weight=None, workers=1, initial_epoch=0)
model.save('Model.h5')
print('The training is finished. Good Luck')