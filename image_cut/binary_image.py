import cv2
import matplotlib.pyplot as plt
import numpy as np
def resize(img:str):#resize a image into 128*128 resolution

    ratio = img.shape[1]/img.shape[0]
    size = (64,int(64/ratio))
    sigma_x,sigma_y = int(img.shape[0]/100),int(img.shape[1]/100)
    size_of_filiter  = [7*sigma_x,7*sigma_y]
    for i in range(0,len(size_of_filiter)):
        if size_of_filiter[i]%2 == 0:
            size_of_filiter[i]+=1

    if img.shape[0] * img.shape[1] < size[0] * size[1]:
        img = cv2.resize(img,size)
        ret,img=cv2.threshold(img, 16, 255, cv2.THRESH_BINARY)
        return img
    else:
        img = cv2.GaussianBlur(img,size_of_filiter,sigma_x,sigma_y)
        img = cv2.resize(img,size)
        ret,img=cv2.threshold(img, 16, 255, cv2.THRESH_BINARY)
        return img

def pre_transfer(img):
    h,w = img.shape
    img1 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,4)
    img1 = cv2.medianBlur(img1,3)
    img2 = cv2.Canny(img,128,255)
    return img2+img1    

    
