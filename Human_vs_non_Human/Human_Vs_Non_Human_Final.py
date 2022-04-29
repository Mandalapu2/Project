import os
import fnmatch
import cv2
import random
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop, Adam, SGD

def remove_white_image(img_loc: str):
    img_loc = images_loc
    img_s = {"new-human": {}, "new-non-human": {}}
    for path, sub_dirs, files in os.walk(img_loc):
        req_path_list = list(img_s.keys())
        req_path = any(ele in path for ele in req_path_list)
        if req_path == True:
            print(path)
            for file in fnmatch.filter(files, "*.png"):
                img = cv2.imread(path + '\\' + file)
                if round(np.mean(img)) == 255:
                    os.remove(path + '\\' + file)
def load_images(img_loc: str):
    img_loc = images_loc
    img_s = {"new-human": {}, "new-non-human": {}}
    for path, sub_dirs, files in os.walk(img_loc):
        req_path_list = list(img_s.keys())
        req_path = any(ele in path for ele in req_path_list)
        if req_path == True:
            print(path)
            for file in fnmatch.filter(files, "*.png"):
                file_path = os.path.join(path, file)
                img = file_path.split("/")[-1].split("_")[0]
                if "new-non-human" in file_path:
                    img_s['new-non-human'].update({img: img_s['new-non-human'].get(img, []) + [cv2.imread(file_path)]})
                else:
                    img_s['new-human'].update({img: img_s['new-human'].get(img, []) + [cv2.imread(file_path)]})
    return img_s
def cross_val_list():
    random.seed(1)
    
# =============================================================================
#     non_radial_rand = random.choices([i for i in images['new-non-human'].keys() if i not in list(images['new-human'].keys())],
#                                     k=images['new-human'].__len__())
#     radial_rand = random.choices([i for i in images['new-human'].keys() if i not in list(images['new-non-human'].keys())],
#                                     k=images['new-non-human'].__len__())
# =============================================================================
    dist = []
    for idx, i in enumerate(list(images['new-human'].keys())):
        human_keys = list(images['new-human'].keys())
        non_human_keys = list(images['new-non-human'].keys())
        num_to_select = 2
        rand_human = random.sample(human_keys, num_to_select)
        rand_non_human = random.sample(non_human_keys, num_to_select)
        choices = rand_human + rand_non_human
        df_choices = pd.DataFrame({'keys':choices})
        df_choices = df_choices.join(df_choices['keys'].str.split('\\', expand=True))
        all_keys = human_keys + non_human_keys
        df_all_keys = pd.DataFrame({'keys':all_keys})
        df_all_keys = df_all_keys.join(df_all_keys['keys'].str.split('\\', expand=True))
        df_all_keys = df_all_keys[df_all_keys[2].isin(list(df_choices[2]))== False]
        choices = list(df_choices['keys'])
        all_keys = list(df_all_keys['keys'])
        dist.append([all_keys,choices])

        
# =============================================================================
#         print(idx,i)
#         choices = radial_rand[idx:idx+2] + non_radial_rand[idx:idx+2]
#         #choices = [i, non_radial_rand[idx]]
#         print(choices)
#         rem = list(set(i for k, v in images.items() for i in v.keys() if i not in choices))
#         x = [item for item in human_keys if item not in rem]
#         dist.append([rem, choices])
# =============================================================================
    return dist


    # random.seed(1)
    # non_radial_rand = random.choices([i for i in images['new-non-human'].keys() if i not in list(images['new-human'].keys())],
    #                                 k=images['new-human'].__len__())
    # radial_rand = random.choices([i for i in images['new-human'].keys() if i not in list(images['new-non-human'].keys())],
    #                                 k=images['new-non-human'].__len__())
    # dist = []
    # for idx, i in enumerate(list(images['new-human'].keys())):
    #     choices = radial_rand[idx:idx+3] + non_radial_rand[idx:idx+3]
    #     #choices = [i, non_radial_rand[idx]]
    #     print(choices)
    #     rem = list(set(i for k, v in images.items() for i in v.keys() if i not in choices))
    #     dist.append([rem, choices])
    # return dist


if __name__ == '__main__':
    images_loc = "C:/Users/jahna/OneDrive/Documents/GitHub/Project/classfication_human_vs_non_human/data/cropped/8x8"
    # labels = ['radial', 'non-radial']
    images = load_images(images_loc)
    in_shape = images['new-human'][list(images['new-human'].keys())[0]][0].shape

    # HYPERPARAMETERS 
    batch_size = 128
    epochs = 10
    verbose = 1
    learning_rate = 0.001
    input_shape = in_shape  # (120, 120, 3) if 16x16 images selected
    cross_vals = cross_val_list()

    # MODEL ARCHITECTURE
    cnn_model = Sequential([
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',input_shape=in_shape, name="CONVOLUTION_2D_1"),
        MaxPooling2D((2, 2), name="MAXPOOL_2D_1"),
        Conv2D(64, (3, 3), activation='relu', name='CONVOLUTION_2D_2'),
        MaxPooling2D((2, 2), name="MAXPOOL_2D_2"),
        Conv2D(128, (3, 3), activation='relu', name='CONVOLUTION_2D_3'),
        MaxPooling2D((2, 2), name="MAXPOOL_2D_3"),
        Flatten(name="FLATTEN"),
        Dense(128, activation='relu', name="DENSE"),
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
            bool_val = 0 if keys == "new-human" else 1
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
