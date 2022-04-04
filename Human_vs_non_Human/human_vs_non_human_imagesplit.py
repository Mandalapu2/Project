import os
import fnmatch
from math import ceil
import cv2
import numpy as np
import matplotlib.pyplot as plt


def create_dirs(dirs: list):
    for direc in dirs:
        if not os.path.exists(direc):
            os.makedirs(direc)
def get_images(dir_loc: str, categories: list):
    dir_loc = image_dir
    categories = labels
    img_s, groups = {}, categories
    for group in groups:
        img_s.update({group: []})
        img_dir = os.path.join(dir_loc, group)
        img_dir = img_dir.replace("\\", "/")
       
        for path, sub_dirs, files in os.walk(img_dir):
            for file in fnmatch.filter(files, "*.png"):
                img_s[group].append(os.path.join(path, file))
    return img_s
def split_images(img_s: dict, new_human_loc: str,
                 new_non_human_loc: str):
    for idx, img_path in enumerate(img_s['human']):
        img_name = img_path.split("\\")[-1]
        img = cv2.imread(img_path)
        non_mask_img_name = img_name.replace("_mask", "")
        non_mask_img_path = img_path.replace("_mask", "")
        non_mask_img = cv2.imread(non_mask_img_path)
        
        if np.mean(img) == 255:
            if not os.path.exists(new_non_human_loc):
                os.makedirs(new_non_human_loc)
            cv2.imwrite("{0}/{1}".format(new_non_human_loc, non_mask_img_name), non_mask_img)
            
            
        else:
            if not os.path.exists(new_human_loc):
                os.makedirs(new_human_loc)
            cv2.imwrite("{0}/{1}".format(new_human_loc, non_mask_img_name), non_mask_img)
            
    for idx, img_path in enumerate(img_s['non-human']):
        img_name = img_path.split("\\")[-1]
        img = cv2.imread(img_path)
        non_mask_img_name = img_name.replace("_mask", "")
        non_mask_img_path = img_path.replace("_mask", "")
        non_mask_img = cv2.imread(non_mask_img_path)
        if np.mean(img) == 255:
            if not os.path.exists(new_non_human_loc):
                os.makedirs(new_non_human_loc)
            cv2.imwrite("{0}/{1}".format(new_non_human_loc, non_mask_img_name), non_mask_img)
        else:
            if not os.path.exists(new_human_loc):
                os.makedirs(new_human_loc)
            cv2.imwrite("{0}/{1}".format(new_human_loc, non_mask_img_name), non_mask_img)

if __name__ == '__main__':
    image_dir = "C:/Users/jahna/OneDrive/Documents/GitHub/Project/classfication_human_vs_non_human/data/cropped/4x4"
    labels = ['human', 'non-human']
    x_div, y_div = 4, 4
    new_human = "C:/Users/jahna/OneDrive/Documents/GitHub/Project/classfication_human_vs_non_human/data/cropped/{}x{}/new_human".format(x_div, y_div)
    new_non_numan = "C:/Users/jahna/OneDrive/Documents/GitHub/Project/classfication_human_vs_non_human/data/cropped/{}x{}/new_non_human".format(x_div, y_div)

    images = get_images(image_dir, labels)
    split_images(images,  new_human, new_non_numan)    
    
