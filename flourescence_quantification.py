#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  9 20:45:08 2025

@author: ren
"""

import os
import tifffile as tiff
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

## enhance contrast with CLAHE
#Histogram Equalization considers the global contrast of the image, may not give good results.
#Adaptive histogram equalization divides images into small tiles and performs hist. eq.
#Contrast limiting is also applied to minimize aplification of noise.
#Together the algorithm is called: Contrast Limited Adaptive Histogram Equalization (CLAHE)
#plt.hist(cl_img.flat, bins =100, range=(0,255))

#cv2.imshow("Otsu", th)
#cv2.waitKey(0)          
#cv2.destroyAllWindows() 
# ---------- User settings ----------

in_folder    = r"/home/ren/Downloads/test_img/1-555/split/Ch1_denoise"      # folder with input images
out_folder   = r"/home/ren/Downloads/test_img/1-555/split/Ch1_masked"
csv_out      = r"/home/ren/Downloads/test_img/1-555/split/Ch1_masked/result.csv"
results      = []
clahe        = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))  #Define tile size and clip limit.
#kernel      = np.ones((3,3),np.uint8)   # kernels are used for morphology filters. 

# -----------------------------------

for file in os.listdir(in_folder):
        file_path = os.path.join(in_folder, file)    
        # Read the image
        img = tiff.imread(file_path)
        # Enhance image contrast with CLAHE
        cl_img = clahe.apply(img)
        # Threshold the image with Otsu filter
        ret,th = cv2.threshold(cl_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # Multiply binary mask and original image
        binary_mask = th.astype(bool)
        masked_img = img * binary_mask
        # Write masked images in new folder
        file_name = os.path.splitext(file)[0]
        full_path = os.path.join(out_folder, file_name)        
        tiff.imwrite(full_path, masked_img)
        
        
        ## Quantification of flourescent intensity
        # Integrated intensity = sum of pixel values in the mask
        integrated = masked_img.sum(dtype=np.float64)
        # Area (in pixels) = count of True in mask
        area = int(binary_mask.sum())
        # Mean intensity over bright area
        mean_int = np.nan if area == 0 else integrated / area
        # Extract file name without extension
        file_name = os.path.splitext(file)[0]
        full_path = os.path.join(out_folder, file_name)
        
        results.append({
        "id": os.path.basename(full_path),
        "integrated_intensity": integrated,
        "area_pixels": area,
        "intensity_per_area": mean_int
        })
results = pd.DataFrame(results)
results.to_csv(csv_out, index=False)
