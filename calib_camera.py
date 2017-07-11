# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 17:04:31 2017

@author: wogrady
"""
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
import pickle

def getObjImgPoints(visualize=True):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,9,0)
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    # Make a list of calibration images
    images = glob.glob('camera_cal/calibration*.jpg')
    
    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    
        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
    
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (9,6), corners, ret)

            if visualize == True:
                cv2.imshow('Distorted Image', img)
                cv2.waitKey(500)
    if visualize == True:
        cv2.destroyAllWindows()
    return objpoints, imgpoints

def calibCamera(objpoints, imgpoints, visualize=False):
    
    # Test undistortion on an image
    img = cv2.imread('camera_cal/calibration1.jpg')
    img_size = (img.shape[1], img.shape[0])
    
    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None,None)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    
    # Visualize before and after undistortion is applied
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(dst)
    plt.title('Undistorted Image')
    
    # Save resulting image
    output_path = "output_images/undistored_image.jpg"
    if (os.path.isfile(output_path)):
            os.remove(output_path)
    plt.savefig(output_path, bbox_inches='tight')   
    
    return ret, mtx, dist, rvecs, tvecs



objpoints = []
imgpoints = []
visualize = True
objpoints, imgpoints = getObjImgPoints(visualize)
ret, mtx, dist, rvecs, tvecs = calibCamera(objpoints, imgpoints, visualize)


# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
cam_pickle = {}
cam_pickle["mtx"] = mtx
cam_pickle["dist"] = dist
pickle.dump( cam_pickle, open( "cam_pickle.p", "wb" ) )
