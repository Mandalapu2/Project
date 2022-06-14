from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import KFold
import numpy as np
import cv2
import glob

# ============ Data Preparation ===============================================
ext = ['png', 'jpg', 'gif']    # Add image formats here
train_human_path = "C:/Users/jahna/OneDrive/Documents/GitHub/Project/classfication_human_vs_non_human/data/cropped/8x8/Dataset_human_vs_non_human/train/human/"
files = []
[files.extend(glob.glob(train_human_path + '*.' + e)) for e in ext]
train_human_img = [cv2.imread(file) for file in files]
train_human_img_ar = np.array(train_human_img)
train_target_human = np.array([[1]]*len(train_human_img_ar))
train_non_human_path = "C:/Users/jahna/OneDrive/Documents/GitHub/Project/classfication_human_vs_non_human/data/cropped/8x8/Dataset_human_vs_non_human/train/non-human/"
files = []
[files.extend(glob.glob(train_non_human_path + '*.' + e)) for e in ext]
train_non_human_img = [cv2.imread(file) for file in files]
train_non_human_img_ar = np.array(train_non_human_img)
train_target_non_human = np.array([[0]]*len(train_non_human_img_ar))

input_train = np.concatenate((train_human_img_ar, train_non_human_img_ar), axis=0)
train_target = np.concatenate((train_target_human, train_target_non_human), axis=0)

test_human_path = "C:/Users/jahna/OneDrive/Documents/GitHub/Project/classfication_human_vs_non_human/data/cropped/8x8/Dataset_human_vs_non_human/test/human/"
files = []
[files.extend(glob.glob(test_human_path + '*.' + e)) for e in ext]
test_human_img = [cv2.imread(file) for file in files]
test_human_img_ar = np.array(test_human_img)
test_target_human = np.array([[1]]*len(test_human_img_ar))
test_non_human_path = "C:/Users/jahna/OneDrive/Documents/GitHub/Project/classfication_human_vs_non_human/data/cropped/8x8/Dataset_human_vs_non_human/test/non-human/"
files = []
[files.extend(glob.glob(test_non_human_path + '*.' + e)) for e in ext]
test_non_human_img = [cv2.imread(file) for file in files]
test_non_human_img_ar = np.array(test_non_human_img)
test_target_non_human = np.array([[0]]*len(test_non_human_img_ar))

input_test = np.concatenate((test_human_img_ar, test_non_human_img_ar), axis=0)
test_target = np.concatenate((test_target_human, test_target_non_human), axis=0)

# Model configuration
batch_size = 50
img_width, img_height, img_num_channels = 240, 240, 3
loss_function = sparse_categorical_crossentropy
no_classes = 2
no_epochs = 25
optimizer = Adam()
verbosity = 1
num_folds = 10
# Determine shape of the data
input_shape = (img_width, img_height, img_num_channels)

# Parse numbers as floats
input_train = input_train.astype('float32')
input_test = input_test.astype('float32')


# Normalize data
input_train = input_train / 255
input_test = input_test / 255
acc_per_fold = []
loss_per_fold = []

# Merge inputs and targets
inputs = np.concatenate((input_train, input_test), axis=0)
targets = np.concatenate((train_target, test_target), axis=0)

# Define the K-fold Cross Validator
kfold = KFold(n_splits=num_folds, shuffle=True)

# K-fold Cross Validation model evaluation
fold_no = 1
for train, test in kfold.split(inputs, targets):
    # Define the model architecture
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', 
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(no_classes, activation='softmax'))
  
    # Compile the model
    model.compile(loss=loss_function,
                  optimizer=optimizer,
                  metrics=['accuracy'])
 # Generate a print
    print('------------------------------------------------------------------------')
    print(f'Training for fold {fold_no} ...')
  
    # Fit data to model
    history = model.fit(inputs[train], targets[train],
                batch_size=batch_size,
                epochs=no_epochs,
                verbose=verbosity)
  
    # Generate generalization metrics
    scores = model.evaluate(inputs[test], targets[test], verbose=0)
    print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
    acc_per_fold.append(scores[1] * 100)
    
    loss_per_fold.append(scores[0])
  
    # Increase fold number
    fold_no = fold_no + 1

# == Provide average scores ==
print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')

