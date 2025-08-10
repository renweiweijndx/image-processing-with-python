# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#openCV is a library of programming functions mainly aimed at computer vision.
#Very good for images and videos, especially real time videos.
#It is used extensively for facial recognition, object recognition, motion tracking,
#optical character recognition, segmentation, and even for artificial neural netwroks. 

"""
You can import images in color, grey scale or unchanged usingindividual commands 
cv2.IMREAD_COLOR : Loads a color image. Any transparency of image will be neglected. It is the default flag.
cv2.IMREAD_GRAYSCALE : Loads image in grayscale mode
cv2.IMREAD_UNCHANGED : Loads image as such including alpha channel
Instead of these three flags, you can simply pass integers 1, 0 or -1 respectively.

"""

#import library
import cv2
import glob
import matplotlib.pyplot as plt

##read images
grey_img = cv2.imread("/home/ren/Downloads/test_img/1-555/rescue1-5551.lsm", 0)
color_img = cv2.imread("/home/ren/Downloads/test_img/1-555/rescue1-5551.lsm", 1)

#images opened using cv2 are numpy arrays
#print(type(grey_img)) 
#print(type(color_img)) 

# Use the function cv2.imshow() to display an image in a window. 
# First argument is the window name which is a string. second argument is our image. 

cv2.imshow("pic", grey_img)
cv2.imshow("color pic", color_img)

# Maintain output window until 
# user presses a key or 1000 ms (1s)
cv2.waitKey(0)          

#destroys all windows created
cv2.destroyAllWindows() 

#OpenCV imread, imwrite and imshow all work with the BGR order, not RGB
#but there is no need to change the order when you read an image with 
#cv2.imread and then want to show it with cv2.imshow
#if you use matplotlib, it uses RGB. 


plt.imshow(color_img)  

#OpenCV represents RGB images as multi-dimensional NumPy arrays, but as BGR.

#we can convert the images from BGR to RGB
plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB))

#We can also change color spaces from RGB to HSV..
plt.imshow(cv2.cvtColor(color_img, cv2.COLOR_BGR2HSV))


##batch process images
#select the path
path = "/home/ren/Downloads/test_img/1-555/*.*"
for file in glob.glob(path):
    print(file)     #just stop here to see all file names printed
    a= cv2.imread(file)  #now, we can read each file since we have the full path
    print(a)  #print numpy arrays for each file

#let us look at each file
   # cv2.imshow('Original Image', a)
   # cv2.waitKey(0)
   # cv2.destroyAllWindows()
    
#process each image - change color from BGR to RGB.
    c = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
    cv2.imshow('Color image', c)
#wait for 1 second
    k = cv2.waitKey(0)
#destroy the window
    cv2.destroyAllWindows()

#######################################################################################