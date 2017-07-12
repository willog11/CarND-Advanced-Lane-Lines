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

def undistort_image(img, visualize=False):
    # Read in the saved camera matrix and distortion coefficients
    cam_pickle = pickle.load( open( "cam_pickle.p", "rb" ) )
    mtx = cam_pickle["mtx"]
    dist = cam_pickle["dist"]

    undistort_img = cv2.undistort(img, mtx, dist, None, mtx)
    
    if visualize == True:
        idx = 1
        plt.figure(figsize=(10, 10))
        
        plt.subplot(rows, cols, idx)
        plt.imshow(img)
        plt.title('Original Image')
        
        idx+=1
        plt.subplot(rows, cols, idx)
        plt.imshow(undistort_img)
        plt.title('Undistrorted Image')
        plt.tight_layout()
        
    return undistort_img

def correct_image(img, visualize=False):
            
    # Creating mask area
    ROI_TOP_LEFT_X = 575
    ROI_TOP_RIGHT_X = 570
    ROI_BOTTOM_LEFT_X = 255
    ROI_BOTTOM_RIGHT_X = 220
    ROI_TOP_Y = 460
    ROI_BOTTOM_Y = 50
    imshape = img.shape

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
    corrected_img = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    if visualize == True:
        # Visualize before and after undistortion is applied
        line_color = (255,0,0)
        rows = 1
        cols = 2
        idx = 1
        plt.figure(figsize=(10, 10))
        
        plt.subplot(rows, cols, idx)
        verticies = np.int32(src)
        undistort_img = np.dstack((img, img, img))*255
        masked_img = cv2.polylines(undistort_img, [verticies], True, line_color , 5)
        plt.imshow(masked_img)
        plt.title('Precorrection Image')
        
        idx+=1
        plt.subplot(rows, cols, idx)
        verticies = np.int32(dst)
        corrected_img_vis = np.dstack((corrected_img, corrected_img, corrected_img))*255
        corrected_img_vis = cv2.polylines(corrected_img_vis, [verticies], True, line_color, 5)
        plt.imshow(corrected_img_vis)
        plt.title('Corrected Image')
        plt.tight_layout()
           
    return corrected_img, M

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

def find_initial_lanes(img, visualize=False):

    # Take a histogram of the bottom half of the image
    histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((img, img, img))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(out_img.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = out_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    
    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = out_img.shape[0] - (window+1)*window_height
        win_y_high = out_img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    
    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    ploty = np.linspace(0, out_img.shape[0]-1, out_img.shape[0] )
    
    if visualize == True:
        #Generate x and y values for plotting

        plt.figure(figsize=(10, 10))      
        plt.title("Histogram of points (lower half of image)")
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        plt.plot(histogram)

        plt.figure(figsize=(10, 10)) 
        plt.title("Lanes found")
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
    
    return left_fit, right_fit, ploty

def find_next_lanes(img, left_fit, right_fit, visualize=False):
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  
    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    if visualize == True:
        plt.figure(figsize=(10, 10))
        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((img, img, img))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        
        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))
        
        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
    
    return left_fit, right_fit, ploty

def calc_curvature_offset(left_fit, right_fit, ploty):
    
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    left_fit = np.asarray(left_fit)
    y_eval = np.max(ploty)
    #left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    #right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    #print(left_curverad, 'm', right_curverad, 'm')
   
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    
    # Fit new polynomials to x,y in world space
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
    
    #print("Min: "+str(np.min(right_fitx)*xm_per_pix))
    #print("Max: "+str(np.max(right_fitx)*xm_per_pix))
    # Calculate the new radii of curvature
    left_curverad_m = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad_m = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters

    curve_mean = ((left_curverad_m + right_curverad_m) / 2)
        
    img_width = 1280
    center_img_m = (img_width/2 + 0.5) * xm_per_pix
    
    left_lane_m = np.max(left_fitx)* xm_per_pix
    right_lane_m = np.max(right_fitx)*xm_per_pix
    lane_midpt   = (left_lane_m + right_lane_m) / 2
    veh_pos_rel_center = lane_midpt - center_img_m

    return curve_mean, veh_pos_rel_center



def draw_result_raw_image(img, binary_img, M, left_fit, right_fit, ploty, mean_curve_m, lat_off_rel_ctr, visualize=False):
    
    Minv = np.linalg.inv(M)
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(img, 1, newwarp, 0.3, 0)
    # Write some Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result,'Mean Radius of Curvature: {0:.2f}m'.format(mean_curve_m),(350,50), font, 1,(255,255,255),2)
    cv2.putText(result,' Lateral Offset Rel. Center {0:.2f}m'.format(lat_off_rel_ctr),(360,100), font, 1,(255,255,255),2)
    
    if visualize == True:
        plt.figure(figsize=(10, 10))
        plt.imshow(result)
    
def pipeline(img):
    visualize_preproc_steps = False
    
    undistort_img                       = undistort_image(img, False)
    color_binary                        = extract_hls_features(undistort_img, False)
    thresh_binary                       = get_thresholded_img(undistort_img, False)
    
    combined                            = np.zeros_like(thresh_binary)
    combined[(color_binary == 1) | (thresh_binary == 1)] = 1
    
    corrected_img, M                    = correct_image(combined, False)
    left_fit, right_fit, ploty          = find_initial_lanes(corrected_img, False)
    left_fit, right_fit, ploty          = find_next_lanes(corrected_img, left_fit, right_fit, False)
    mean_curve_m, lat_off_rel_ctr       = calc_curvature_offset(left_fit, right_fit, ploty)
    
    draw_result_raw_image(img, corrected_img, 
                          M, left_fit, right_fit, ploty, 
                          mean_curve_m, lat_off_rel_ctr, True)
    
    if visualize_preproc_steps == True:
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
    
    
# Make a list of test images
images = glob.glob('test_images/*.jpg')

for img_count, fname in enumerate(images):
    img = mpimg.imread(fname)
    pipeline(img)
    
    