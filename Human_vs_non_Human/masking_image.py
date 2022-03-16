import cv2
import numpy as np

# Load image, create mask, and draw white circle on mask
image_path = 'C:/Users/jahna/OneDrive/Documents/GitHub/Project/classfication_human_vs_non_human/data/original/human/1.png'
image = cv2.imread(image_path)
mask = np.zeros(image.shape, dtype=np.uint8)
mask = cv2.circle(mask, (260, 300), 225, (255,255,255), -1) 

# Mask input image with binary mask
result = cv2.bitwise_and(image, mask)
# Color background white
result[mask==0] = 255 # Optional

cv2.imshow('image', image)
cv2.imshow('mask', mask)
cv2.imshow('result', result)
cv2.waitKey()


# load image whose you want to create mask
img = cv2.imread(image_path)

# convert to graky
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
