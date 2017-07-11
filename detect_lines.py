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
    corrected_img = cv2.warpPerspective(undistort_img, M, img_size, flags=cv2.INTER_LINEAR)
    
    if visualize == True:
        # Visualize before and after undistortion is applied
        if len(imshape) > 2:
            color_map='brg'
            line_color = (255,0,0)
        else:
            color_map='gray'
            line_color = (1)
        rows = 1
        cols = 2
        idx = 1
        plt.figure(figsize=(10, 10))
        
        plt.subplot(rows, cols, idx)
        plt.imshow(img, cmap=color_map)
        plt.title('Original Image')
        
        idx+=1
        plt.subplot(rows, cols, idx)
        plt.imshow(undistort_img, cmap=color_map)
        plt.title('Distrorted Image')
        plt.tight_layout()

        idx = 1
        plt.figure(figsize=(10, 10))
        
        plt.subplot(rows, cols, idx)
        verticies = np.int32(src)
        masked_img = cv2.polylines(undistort_img, [verticies], True, line_color , 5)
        plt.imshow(masked_img, cmap=color_map)
        plt.title('Precorrection Image')
        
        idx+=1
        plt.subplot(rows, cols, idx)
        verticies = np.int32(dst)
        corrected_img = cv2.polylines(corrected_img, [verticies], True, line_color, 5)
        plt.imshow(corrected_img, cmap=color_map)
        plt.title('Corrected Image')
        plt.tight_layout()
           
    return corrected_img

def white_color_enhance(img):
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(img, lower, upper)
    return  white_mask

def yellow_color_enhance(img):
    lower = np.uint8([  10, 0, 100])
    upper = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(img, lower, upper)
    return yellow_mask

def white_yellow_enhance(img, white_mask, yellow_mask):
    overall_mask = cv2.bitwise_or(white_mask, yellow_mask)
    return overall_mask

def extract_hls_features(img, visualize=False):
    """Applies the HSL transform"""
    hls_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    white_mask = white_color_enhance(hls_image)
    yellow_mask = yellow_color_enhance(hls_image)
    overall_mask = white_yellow_enhance(hls_image, white_mask, yellow_mask)
    overall_mask[(overall_mask == 255)] = 1
    if visualize == True:
        rows = 2
        cols = 2
        idx = 1
        plt.figure(figsize=(10, 10))
        
        plt.subplot(rows, cols, idx)
        plt.imshow(img)
        plt.title('Original image')
        
        idx+=1
        plt.subplot(rows, cols, idx)
        plt.imshow(white_mask, cmap='gray')
        plt.title('Ehanced white colors')
        
        idx+=1
        plt.subplot(rows, cols, idx)
        plt.imshow(yellow_mask, cmap='gray')
        plt.title('Ehanced yellow colors')
        
        idx+=1
        plt.subplot(rows, cols, idx)
        plt.imshow(overall_mask, cmap='gray')
        plt.title('Ehanced white and yellow colors combined')
        plt.tight_layout()
    
    return overall_mask


def grad_threshold(img, orient='x', thresh_min=0, thresh_max=255):

    # Apply x or y gradient with the OpenCV Sobel() function and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(img, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Return the result
    return binary_output

def mag_threshold(img, sobel_kernel=3, mag_thresh=(0, 255)):
    
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    mag = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255*mag/np.max(mag))

    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1

    return binary_output

def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)

    dir_grad = np.arctan2(abs_sobely, abs_sobelx)
    binary_output = np.zeros_like(img)
    binary_output[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1

    return binary_output

def get_thresholded_img(img, visualize=False):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    grad_binary = grad_threshold(gray, orient='x', thresh_min=20, thresh_max=100)
    mag_binary = mag_threshold(gray, sobel_kernel=3, mag_thresh=(30, 100))
    dir_binary = dir_threshold(gray, sobel_kernel=11, thresh=(0.7, 1.3))
    
    combined = np.zeros_like(dir_binary)
    combined[(grad_binary==1) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    
    if visualize == True:
        rows = 3
        cols = 2
        idx = 1
        plt.figure(figsize=(10, 10))
        
        plt.subplot(rows, cols, idx)
        plt.imshow(img)
        plt.title('Original image')
        
        idx+=1
        plt.subplot(rows, cols, idx)
        plt.imshow(grad_binary, cmap='gray')
        plt.title('Gradient Thresholded Image')
        
        idx+=1
        plt.subplot(rows, cols, idx)
        plt.imshow(mag_binary, cmap='gray')
        plt.title('Magnitude Thresholded Image')
        
        idx+=1
        plt.subplot(rows, cols, idx)
        plt.imshow(dir_binary, cmap='gray')
        plt.title('Direction Thresholded Image')
        
        idx+=1
        plt.subplot(rows, cols, idx)
        plt.imshow(combined, cmap='gray')
        plt.title('Combined Thresholded Image')
    
    return combined
                

visualize = True
# Make a list of test images
images = glob.glob('test_images/straight_lines*.jpg')

for img_count, fname in enumerate(images):
    img = mpimg.imread(fname)
    color_binary = extract_hls_features(img, False)
    thresh_binary = get_thresholded_img(img, False)
    
    combined = np.zeros_like(thresh_binary)
    combined[(color_binary == 1) | (thresh_binary == 1)] = 1
    
    corrected_img = correct_image(combined, False)
    
    if visualize == True:
        rows = 1
        cols = 3
        idx = 1
        plt.figure(figsize=(10, 10))
        
        plt.subplot(rows, cols, idx)
        plt.imshow(img)
        plt.title('Original image')
        
        idx+=1
        plt.subplot(rows, cols, idx)
        plt.imshow(combined, cmap='gray')
        plt.title('Color & Thresholded Image')
        
        idx+=1
        plt.subplot(rows, cols, idx)
        plt.imshow(corrected_img, cmap='gray')
        plt.title('Resulting Corrected Image')
    