import cv2
from binary_image import *
import numpy as np
def cut_by_x(img):
    image = pre_transfer(img)
    projection = np.sum(image == 0, axis=0)
    start_x = -1 
    segments = [] 
    
    for x in range(len(projection)):
        if projection[x] > 0:
            if start_x == -1: 
                start_x = x
        else: 
            if start_x != -1:
                segments.append(image[:, start_x:x].copy())
                start_x = -1
    if start_x != -1:
        segments.append(image[:, start_x:].copy())
    
    return segments
def cut_white_edge(img):
    image = cv2.Canny(img,128,200)
    image = img
    height, width = image.shape
    
    top = 0
    while top < height and np.all(image[top, :] == 255):
        top += 1
    
    bottom = height - 1
    while bottom >= 0 and np.all(image[bottom, :] == 255):
        bottom -= 1
    

    if top > bottom:
        return [np.array([], dtype=image.dtype).reshape(0, 0)]
    
    left = 0
    while left < width and np.all(image[:, left] == 255):
        left += 1
    
    right = width - 1
    while right >= 0 and np.all(image[:, right] == 255):
        right -= 1
    cropped = img[top:bottom+1, left:right+1].copy()
    
    return cropped

