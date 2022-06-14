import random
import os
import shutil
from glob import glob
from random import shuffle

# create directories
dataset_home = 'C:/Users/jahna/OneDrive/Documents/GitHub/Project/classfication_human_vs_non_human/data/cropped/8x8/dataset_human_vs_non_human/'
subdirs = ['train/', 'test/']
for subdir in subdirs:
	# create label subdirectories
	labeldirs = ['human/', 'non-human/']
	for labldir in labeldirs:
		newdir = dataset_home + subdir + labldir
		os.makedirs(newdir, exist_ok=True)

def load_images(img_loc: str):
    img_loc = "C:/Users/jahna/OneDrive/Documents/GitHub/Project/classfication_human_vs_non_human/data/cropped/8x8"
    img_s = {"new-human": {}, "new-non-human": {}}
    random.seed(1)
    val_ratio = 0.25
    for path, sub_dirs, files in os.walk(img_loc):
        req_path_list = list(img_s.keys())
        req_path = any(ele in path for ele in req_path_list)
        if req_path == True:
            print(path)
            for file in os.listdir(path):
                src = path + "/" + file
                dst_dir  = 'train/'
                if random.random()< val_ratio:
                    dst_dir = 'test/'
                if "new-human" in path:
                    dst = dataset_home + dst_dir + "human/" + file
                    shutil.copyfile(src, dst)
                elif "new-non-human" in path :
                    dst = dataset_home + dst_dir + "non-human/" + file
                    shutil.copyfile(src, dst)
                    
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
    
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(200, 200, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))
# compile model
opt = SGD(lr=0.001, momentum=0.9)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

datagen = ImageDataGenerator(rescale=1.0/255.0)
	# prepare iterators
train_it = datagen.flow_from_directory('dataset_human_vs_non_human/train/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
test_it = datagen.flow_from_directory('dataset_human_vs_non_human/test/',
		class_mode='binary', batch_size=64, target_size=(200, 200))
	# fit model
history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
		validation_data=test_it, validation_steps=len(test_it), epochs=20, verbose=0)

_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
print('> %.3f' % (acc * 100.0))
_, acc = model.evaluate_generator(train_it, steps=len(train_it), verbose=0)
print('> %.3f' % (acc * 100.0))
# def remove_white_image(img_loc: str):
#     img_loc = images_loc
#     img_s = {"new-human": {}, "new-non-human": {}}
#     for path, sub_dirs, files in os.walk(img_loc):
#         req_path_list = list(img_s.keys())
#         req_path = any(ele in path for ele in req_path_list)
#         if req_path == True:
#             print(path)
#             for file in fnmatch.filter(files, "*.png"):
#                 img = cv2.imread(path + '\\' + file)
#                 if round(np.mean(img)) == 255:
#                     os.remove(path + '\\' + file)

