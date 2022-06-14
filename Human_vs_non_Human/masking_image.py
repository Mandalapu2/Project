import cv2
import math
import numpy as np
import scipy.ndimage
import os
import fnmatch


def orientated_non_max_suppression(mag, ang):
    ang_quant = np.round(ang / (np.pi/4)) % 4
    winE = np.array([[0, 0, 0],[1, 1, 1], [0, 0, 0]])
    winSE = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    winS = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
    winSW = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    magE = non_max_suppression(mag, winE)
    magSE = non_max_suppression(mag, winSE)
    magS = non_max_suppression(mag, winS)
    magSW = non_max_suppression(mag, winSW)

    mag[ang_quant == 0] = magE[ang_quant == 0]
    mag[ang_quant == 1] = magSE[ang_quant == 1]
    mag[ang_quant == 2] = magS[ang_quant == 2]
    mag[ang_quant == 3] = magSW[ang_quant == 3]
    return mag

def non_max_suppression(data, win):
    data_max = scipy.ndimage.filters.maximum_filter(data, footprint=win, mode='constant')
    data_max[data != data_max] = 0
    return data_max

def get_images(dir_loc: str, categories: list):
    img_s, groups = {}, categories
    for group in groups:
        img_s.update({group: []})
        img_dir = os.path.join(dir_loc, group)
        for path, sub_dirs, files in os.walk(img_dir):
            for file in fnmatch.filter(files, "*.png"):
                img_s[group].append(os.path.join(path, file))
    return img_s



image_dir = "C:/Users/jahna/OneDrive/Documents/GitHub/Project/classfication_human_vs_non_human/data/original"
labels = ['human', 'non_human']
images = get_images(image_dir, labels)

for i in images:
    for idx, img_path in enumerate(j for j in images[i]):
        gray_image = cv2.imread(images[i][idx],0)
        with_nmsup = True #apply non-maximal suppression
        fudgefactor = 0.8 #with this threshold you can play a little bit
        sigma = 21 #for Gaussian Kernel
        kernel = 2*math.ceil(2*sigma)+1 #Kernel size

        gray_image = gray_image/255.0
        blur = cv2.GaussianBlur(gray_image, (kernel, kernel), sigma)
        gray_image = cv2.subtract(gray_image, blur)

        # compute sobel response
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        mag = np.hypot(sobelx, sobely)
        ang = np.arctan2(sobely, sobelx)

        # threshold
        threshold = 4 * fudgefactor * np.mean(mag)
        mag[mag < threshold] = 0

        #either get edges directly
        if with_nmsup is False:
            mag = cv2.normalize(mag, 0, 255, cv2.NORM_MINMAX)
            kernel = np.ones((5,5),np.uint8)
            result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)
            imagem = cv2.bitwise_not(result)
            filename = images[i][idx].split("\\")[0] +"/" + i + \
                          "/" + images[i][idx].split("\\")[-1].split(".")[0]\
                            +  "_mask.png"
            cv2.imwrite(filename, imagem)

        #or apply a non-maximal suppression
        else:

            # non-maximal suppression
            mag = orientated_non_max_suppression(mag, ang)
            # create mask
            mag[mag > 0] = 255
            mag = mag.astype(np.uint8)

            kernel = np.ones((5,5),np.uint8)
            result = cv2.morphologyEx(mag, cv2.MORPH_CLOSE, kernel)
            imagem = cv2.bitwise_not(result)
            filename = images[i][idx].split("\\")[0] +"/" + i + \
                      "/" + images[i][idx].split("\\")[-1].split(".")[0]\
                        +  "_mask.png"
            cv2.imwrite(filename, imagem)




