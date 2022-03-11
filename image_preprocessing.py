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
        for path, sub_dirs, files in os.walk(img_dir):
            for file in fnmatch.filter(files, "*.png"):
                img_s[group].append(os.path.join(path, file))
    return img_s


def split_images(img_s: dict, x_break: int, y_break: int, new_radial_loc: str, new_non_radial_loc: str):
    create_dirs([new_radial_loc, new_non_radial_loc])

    for idx, img_path in enumerate(img_s['non-radial']):
        img_name = img_path.split("/")[-1].split(".")[0]
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
                    cv2.imwrite("{0}/{1}_{2}_{3}.png".format(new_non_radial_loc, img_name, i, j), temp)
        print("Split completed for image:", img_path)

    for idx, img_path in enumerate([i for i in img_s['radial'] if "_mask" not in i]):
        img_name = img_path.split("/")[-1].split(".")[0]
        img_mask_path = img_path.replace(img_name, img_name + "_mask")
        img = cv2.imread(img_path)
        img_mask = cv2.imread(img_mask_path)
        preview_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, depth = img.shape
        s_height = ceil(height / y_break)
        s_width = ceil(width / x_break)
        for i in range(y_break):
            for j in range(x_break):
                temp = img[i * s_height:(i + 1) * s_height, j * s_width:(j + 1) * s_width]
                mask = img_mask[i * s_height:(i + 1) * s_height, j * s_width:(j + 1) * s_width]
                mask_pixels = len(np.where(mask == 0)[0])
                white_pix = len(np.where(temp == 255)[0])
                if mask_pixels > 0.2 * temp.shape[0] * temp.shape[1]:
                    if white_pix < 0.8 * temp.shape[0] * temp.shape[1]:
                        cv2.imwrite("{0}/{1}_{2}_{3}.png".format(new_radial_loc, img_name, i, j), temp)
                else:
                    if white_pix < 0.8 * temp.shape[0] * temp.shape[1]:
                        cv2.imwrite("{0}/{1}_{2}_{3}.png".format(new_non_radial_loc, img_name, i, j), temp)
        print("Split completed for image:", img_path)
    print("Images created successfully")


if __name__ == '__main__':
    image_dir = "data/original"
    labels = ['radial', 'non-radial']
    x_div, y_div = 16, 16

    radial_folder = "data/cropped/{0}x{0}/radial".format(x_div, y_div)
    non_radial_folder = "data/cropped/{0}x{0}/non-radial".format(x_div, y_div)
    images = get_images(image_dir, labels)
    split_images(images, x_div, y_div, radial_folder, non_radial_folder)


