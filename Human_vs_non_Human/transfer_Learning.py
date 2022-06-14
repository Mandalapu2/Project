import os
import numpy as np
import shutil
num_classes = 2
CHANNELS = 3
IMAGE_RESIZE = 224
RESNET50_POOLING_AVERAGE = 'avg'
DENSE_LAYER_ACTIVATION = 'softmax'
OBJECTIVE_FUNCTION = 'categorical_crossentropy'
LOSS_METRICS = ['accuracy']

# EARLY_STOP_PATIENCE must be < NUM_EPOCHS

NUM_EPOCHS = 10

EARLY_STOP_PATIENCE = 3
# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively

# Training images processed in each step would be no.-of-train-images / STEPS_PER_EPOCH_TRAINING

STEPS_PER_EPOCH_TRAINING = 10

STEPS_PER_EPOCH_VALIDATION = 10

# These steps value should be proper FACTOR of no.-of-images in train & valid folders respectively

# NOTE that these BATCH* are for Keras ImageDataGenerator batching to fill epoch step input

BATCH_SIZE_TRAINING = 100

BATCH_SIZE_VALIDATION = 100

# Using 1 to easily manage mapping between test_generator & prediction for submission preparation

BATCH_SIZE_TESTING = 1

#Defining the model
import keras

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Flatten, GlobalAveragePooling2D

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg', weights='imagenet'))

my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False

my_new_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras import layers

from tensorflow.keras.optimizers import SGD

x = layers.Dropout(0.5)(my_new_model.output)

x = layers.Flatten()(x)

x = layers.Dropout(0.5)(x)

x = layers.Dense(128, activation='relu', kernel_initializer='he_uniform')(x)

x = layers.Dense(1, activation='sigmoid')(x)

opt = SGD(lr=0.001, momentum=0.9)

my_new_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["acc"])

my_new_model.summary()


#Generating the dataset
from keras.applications.resnet import preprocess_input

from keras.preprocessing.image import ImageDataGenerator

image_size = IMAGE_RESIZE

# preprocessing_function is applied on each image but only after re-sizing & augmentation (resize => augment => pre-process)

# Each of the keras.application.resnet* preprocess_input MOSTLY mean BATCH NORMALIZATION (applied on each batch) stabilize the in
#puts to nonlinear activation functions

# Batch Normalization helps in faster convergence

data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

# flow_From_directory generates batches of augmented data (where augmentation can be color conversion, etc)

# Both train & valid folders must have NUM_CLASSES sub-folders

train_generator = data_generator.flow_from_directory(

'C:/Users/jahna/OneDrive/Documents/GitHub/Project/classfication_human_vs_non_human/data/cropped/8x8/dataset_human_vs_non_human/train',

target_size=(image_size, image_size),

batch_size=BATCH_SIZE_TRAINING,

class_mode='categorical')

validation_generator = data_generator.flow_from_directory(

'C:/Users/jahna/OneDrive/Documents/GitHub/Project/classfication_human_vs_non_human/data/cropped/8x8/dataset_human_vs_non_human/test',

target_size=(image_size, image_size),

batch_size=BATCH_SIZE_VALIDATION,

class_mode='categorical')


#Training the model
(BATCH_SIZE_TRAINING, len(train_generator), BATCH_SIZE_VALIDATION, len(validation_generator))

my_new_model.fit_generator(

train_generator,

epochs=5,

steps_per_epoch=len(train_generator),

validation_steps=len(validation_generator),

validation_data=validation_generator)