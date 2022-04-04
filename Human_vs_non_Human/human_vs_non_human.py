import os
import fnmatch
import cv2
import random
from datetime import datetime
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop, Adam, SGD


def load_images(img_loc: str):
    img_s = {"human": {}, "non-human": {}}
    for path, sub_dirs, files in os.walk(img_loc):
        for file in fnmatch.filter(files, "*.png"):
            file_path = os.path.join(path, file)
            img = file_path.split("/")[-1].split("_")[0]
            if "non-human" in file_path:
                img_s['non-human'].update({img: img_s['non-human'].get(img, []) + [cv2.imread(file_path)]})
            else:
                img_s['human'].update({img: img_s['human'].get(img, []) + [cv2.imread(file_path)]})
    return img_s


def cross_val_list():
    random.seed(1)
    non_radial_rand = random.sample([i for i in images['non-human'].keys() if i not in list(images['human'].keys())],
                                    k=images['human'].__len__())
    dist = []
    for idx, i in enumerate(list(images['human'].keys())):
        choices = [i, non_radial_rand[idx]]
        rem = list(set(i for k, v in images.items() for i in v.keys() if i not in choices))
        dist.append([rem, choices])
    return dist


if __name__ == '__main__':
    images_loc = "data/cropped/4x4"
    # labels = ['radial', 'non-radial']
    images = load_images(images_loc)
    in_shape = images['human'][list(images['human'].keys())[0]][0].shape

    # HYPERPARAMETERS
    batch_size = 128
    epochs = 10
    verbose = 1
    learning_rate = 0.001
    input_shape = in_shape  # (120, 120, 3) if 16x16 images selected
    cross_vals = cross_val_list()

    # MODEL ARCHITECTURE
    cnn_model = Sequential([
        Conv2D(8, (3, 3), activation='relu', input_shape=in_shape, name="CONVOLUTION_2D_1"),
        MaxPooling2D((2, 2), name="MAXPOOL_2D_1"),
        Conv2D(16, (3, 3), activation='relu', name='CONVOLUTION_2D_2'),
        MaxPooling2D((2, 2), name="MAXPOOL_2D_2"),
        Conv2D(32, (3, 3), activation='relu', name='CONVOLUTION_2D_3'),
        MaxPooling2D((2, 2), name="MAXPOOL_2D_3"),
        Flatten(name="FLATTEN"),
        Dense(256, activation='relu', name="DENSE"),
        Dense(1, activation='sigmoid', name="SIGMOID")
    ], name='RADIAL_CNN')

    # COMPILE MODEL
    cnn_model.compile(
        loss='binary_crossentropy',
        optimizer=Adam(lr=learning_rate),
        metrics=['accuracy']
    )
    print(cnn_model.summary())

    acc_per_fold = []
    loss_per_fold = []
    history = []
    fold_no = 1

    for i in cross_vals:
        inputs, outputs = {"train": [], "test": []}, {"train": [], "test": []}
        for keys, vals in images.items():
            bool_val = 0 if keys == "radial" else 1
            for k, v in vals.items():
                var = 'train' if k in i[0] else 'test'
                inputs[var].extend(v)
                outputs[var].extend([bool_val] * len(v))
        for j in [inputs, outputs]:
            for k in j.keys():
                j[k] = np.stack(j[k], axis=0)
        inputs['train'], outputs['train'] = shuffle(inputs['train'], outputs['train'])
        inputs['test'], outputs['test'] = shuffle(inputs['test'], outputs['test'])
        inputs['train'], inputs['test'] = inputs['train'] / 255., inputs['test'] / 255.

        # Generate a print
        print('-----------------------------------------------------------------')
        print('Training for fold', fold_no, '...')

        # Fit data to model
        history.append(cnn_model.fit(inputs['train'], outputs['train'],
                                     batch_size=batch_size,
                                     epochs=epochs,
                                     verbose=verbose,
                                     #                                  callbacks=[TensorBoard(log_dir=logdir+str(fold_no))],
                                     validation_data=(inputs['test'], outputs['test'])
                                     )
                       )
        #     Generate generalization metrics
        scores = cnn_model.evaluate(inputs['test'], outputs['test'], verbose=0)
        print('Score for fold {0}: {1} of {2}; {3} of {4}%'.format(fold_no, cnn_model.metrics_names[0], scores[0],
                                                                   cnn_model.metrics_names[1], scores[1] * 100))
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no += 1

    # == Provide average scores ==
    print('=================================================================')
    print('Score per fold')
    for idx, i in enumerate(zip(loss_per_fold, acc_per_fold)):
        print('-----------------------------------------------------------------')
        print('> Fold {0} - Loss: {1:.6f} - Accuracy: {2:.6f}'.format(idx + 1, i[0], i[1]))
    print('=================================================================')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('=================================================================')

