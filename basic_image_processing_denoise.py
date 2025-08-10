#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 13:32:12 2025

@author: ren
"""


import os
import tifffile as tiff
import cv2

# ---------- User settings ----------
in_folder    = r"/home/ren/Downloads/test_img/1-555/split/Ch1"                             # folder with your ~10 images
out_folder     = r"/home/ren/Downloads/test_img/1-555/split/Ch1_denoise"                       # None -> save next to input folder

# -----------------------------------

for file in os.listdir(in_folder):
        file_path = os.path.join(in_folder, file)    
        # Read the image
        img = tiff.imread(file_path)
        # Denoise images
        bilateral_blur = cv2.bilateralFilter(img,7,50,50)
        
        #cv2.imshow("denoised image", bilateral_blur)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        # Extract file name without extension
        file_name = os.path.splitext(file)[0]
        full_path = os.path.join(out_folder, file_name)
        
        tiff.imwrite(full_path, bilateral_blur)
