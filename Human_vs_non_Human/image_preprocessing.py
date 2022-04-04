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
    img_s, groups = {}, categories
    for group in groups:
        img_s.update({group: []})
        img_dir = os.path.join(dir_loc, group)
        img_dir = img_dir.replace('\\', "/")
        for path, sub_dirs, files in os.walk(img_dir):
            for file in fnmatch.filter(files, "*.png"):
                img_s[group].append(os.path.join(path, file))
    return img_s


def split_images(img_s: dict, x_break: int, y_break: int, new_radial_loc: str, new_non_radial_loc: str):
    create_dirs([new_radial_loc, new_non_radial_loc])

    for idx, img_path in enumerate(img_s['non-human']):
        img_name = img_path.split("/")[-1].split(".")[0]
        img_name = img_name.split('\\')[-1]
        img = cv2.imread(img_path)
        preview_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, depth = img.shape
        s_height = ceil(height / y_break)
        s_width = ceil(width / x_break)
        for i in range(y_break):
            for j in range(x_break):
                temp = img[i * s_height:(i + 1) * s_height, j * s_width:(j + 1) * s_width]
                white_pix = len(np.where(temp == 255)[0])
                if white_pix < 0.8 * temp.shape[0] * temp.shape[1]:
                    cv2.imwrite("{}/{}_{}_{}.png".format(new_non_radial_loc, img_name, i, j), temp)
                if white_pix > 0.8 * temp.shape[0] * temp.shape[1]:
                    cv2.imwrite("{}/{}_{}_{}.png".format(new_non_radial_loc, img_name, i, j), temp)
        
        print("Split completed for image:", img_path)

    for idx, img_path in enumerate(i for i in img_s['human']):
        img_name = img_path.split("/")[-1].split(".")[0]
        img_name = img_name.split('\\')[-1]
        img = cv2.imread(img_path)
        preview_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, depth = img.shape
        s_height = ceil(height / y_break)
        s_width = ceil(width / x_break)
        for i in range(y_break):
            for j in range(x_break):
                temp = img[i * s_height:(i + 1) * s_height, j * s_width:(j + 1) * s_width]
                white_pix = len(np.where(temp == 255)[0])
                if white_pix < 0.8 * temp.shape[0] * temp.shape[1]:
                    cv2.imwrite("{}/{}_{}_{}.png".format(new_radial_loc, img_name, i, j), temp)
                if white_pix > 0.8 * temp.shape[0] * temp.shape[1]:
                    cv2.imwrite("{}/{}_{}_{}.png".format(new_radial_loc, img_name, i, j), temp)
        print("Split completed for image:", img_path)
    print("Images created successfully")


if __name__ == '__main__':
    #image_dir = "C:\Users\jahna\neDrive\Documents\GitHub\Project\classfication_radial_vs_non_radial\data\original"
    image_dir = "C:/Users/jahna/OneDrive/Documents/GitHub/Project/classfication_human_vs_non_human/data/original"
    labels = ['human', 'non-human']
    x_div, y_div = 4, 4
    radial_folder = "C:/Users/jahna/OneDrive/Documents/GitHub/Project/classfication_human_vs_non_human/data/cropped/{}x{}/human".format(x_div, y_div)
    #radial_folder = "data/cropped/{0}x{0}/radial".format(x_div, y_div)
    non_radial_folder = "C:/Users/jahna/OneDrive/Documents/GitHub/Project/classfication_human_vs_non_human/data/cropped/{}x{}/non-human".format(x_div, y_div)
    #non_radial_folder = "data/cropped/{0}x{0}/non-radial".format(x_div, y_div)
    images = get_images(image_dir, labels)
    split_images(images, x_div, y_div, radial_folder, non_radial_folder)


