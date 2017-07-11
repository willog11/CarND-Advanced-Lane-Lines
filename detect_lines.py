# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 08:01:20 2017

@author: wogrady
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob
import os


def correct_image(img, visualize=False):
        
    # Read in the saved camera matrix and distortion coefficients
    cam_pickle = pickle.load( open( "cam_pickle.p", "rb" ) )
    mtx = cam_pickle["mtx"]
    dist = cam_pickle["dist"]
    
    undistort_img = cv2.undistort(img, mtx, dist, None, mtx)
    
    # Creating mask area
    ROI_TOP_LEFT_X = 575
    ROI_TOP_RIGHT_X = 570
    ROI_BOTTOM_LEFT_X = 255
    ROI_BOTTOM_RIGHT_X = 220
    ROI_TOP_Y = 460
    ROI_BOTTOM_Y = 50
    imshape = undistort_img.shape
    src = np.float32([(ROI_BOTTOM_LEFT_X,imshape[0]-ROI_BOTTOM_Y),
                      (ROI_TOP_LEFT_X, ROI_TOP_Y), 
                      (imshape[1]-ROI_TOP_RIGHT_X, ROI_TOP_Y), 
                      (imshape[1]-ROI_BOTTOM_RIGHT_X,imshape[0]-ROI_BOTTOM_Y)])
    
    offset = 250
    img_size = (imshape[1],imshape[0])
    
    dst = np.float32([[offset, imshape[0]],
                      [offset, 0], 
                      [imshape[1]-offset, 0], 
                      [imshape[1]-offset, imshape[0]]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped_img = cv2.warpPerspective(undistort_img, M, img_size, flags=cv2.INTER_LINEAR)
    
    if visualize == True:
        # Visualize before and after undistortion is applied
        rows = 1
        cols = 2
        idx = 1
        plt.figure(figsize=(10, 10))
        
        verticies = np.int32(src)
        #masked_img = region_of_interest(undistort_img, [verticies])
        masked_img = cv2.polylines(undistort_img, [verticies], True, (255,0,0), 5)
        plt.subplot(rows, cols, idx)
        plt.imshow(masked_img)
        plt.title('Masked Image')
        
        idx+=1
        plt.subplot(rows, cols, idx)
        verticies = np.int32(dst)
        warped_img = cv2.polylines(warped_img, [verticies], True, (255,0,0), 5)
        plt.imshow(warped_img)
        plt.title('Corrected Image')
        plt.tight_layout()
        
#        output_path = 'output_images/corrected_image.jpg'
#        if (os.path.isfile(output_path)):
#                os.remove(output_path)
#        plt.savefig(output_path, bbox_inches='tight')

# Make a list of calibration images
images = glob.glob('test_images/straight_lines*.jpg')

# Step through the list and search for chessboard corners
for img_count, fname in enumerate(images):
    img = mpimg.imread(fname)
    correct_image(img, True)