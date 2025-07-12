import cv2
import matplotlib.pyplot as plt
import numpy as np
def resize(img:str):#resize a image into 128*128 resolution

    img = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
    ratio = img.shape[1]/img.shape[0]
    size = (128,int(128/ratio))
    sigma_x,sigma_y = int(img.shape[0]/100),int(img.shape[1]/100)
    size_of_filiter  = [7*sigma_x,7*sigma_y]
    for i in range(0,len(size_of_filiter)):
        if size_of_filiter[i]%2 == 0:
            size_of_filiter[i]+=1

    if img.shape[0] * img.shape[1] < size[0] * size[1]:
        img = cv2.resize(img,size)
        edge = cv2.Laplacian(img,cv2.CV_64F)
        ret,img=cv2.threshold(edge, 16, 255, cv2.THRESH_BINARY)
        return img
    else:
        img = cv2.GaussianBlur(img,size_of_filiter,sigma_x,sigma_y)
        img = cv2.resize(img,size)
        edge = cv2.Laplacian(img,cv2.CV_64F)
        ret,img=cv2.threshold(edge, 16, 255, cv2.THRESH_BINARY)
        return img
    

    